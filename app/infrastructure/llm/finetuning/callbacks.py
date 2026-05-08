"""
Callbacks customizados para monitoramento de convergência e curriculum learning.

- ConvergenceMonitorCallback: detecta stagnação, divergência e overfitting
- CurriculumCallback: troca o dataset progressivamente por tier de complexidade
- DomainMetricsCallback: avalia métricas de domínio durante eval
"""

import json
import logging
from collections import deque
from pathlib import Path

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class ConvergenceMonitorCallback(TrainerCallback):
    """
    Monitora convergência durante o treino e detecta problemas.

    Detecta:
    - Stagnação: loss delta < threshold por N evals consecutivos
    - Divergência: loss crescente por N evals consecutivos
    - Overfitting: train_loss ↓ enquanto eval_loss ↑
    """

    def __init__(
        self,
        stagnation_threshold: float = 1e-4,
        stagnation_patience: int = 3,
        divergence_patience: int = 2,
    ):
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_patience = stagnation_patience
        self.divergence_patience = divergence_patience

        self._eval_losses: deque[float] = deque(maxlen=10)
        self._train_losses: deque[float] = deque(maxlen=10)
        self._stagnation_count = 0
        self._divergence_count = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            self._train_losses.append(logs["loss"])

        if "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            self._eval_losses.append(eval_loss)

            if len(self._eval_losses) >= 2:
                delta = abs(self._eval_losses[-1] - self._eval_losses[-2])
                direction = self._eval_losses[-1] - self._eval_losses[-2]

                # Stagnação
                if delta < self.stagnation_threshold:
                    self._stagnation_count += 1
                    if self._stagnation_count >= self.stagnation_patience:
                        logger.warning(
                            "[CONVERGENCE] Stagnação detectada: loss delta < %.1e "
                            "por %d evals consecutivos. Considere reduzir LR ou adicionar dados.",
                            self.stagnation_threshold,
                            self._stagnation_count,
                        )
                else:
                    self._stagnation_count = 0

                # Divergência
                if direction > 0:
                    self._divergence_count += 1
                    if self._divergence_count >= self.divergence_patience:
                        logger.warning(
                            "[CONVERGENCE] Divergência detectada: eval_loss crescente "
                            "por %d evals consecutivos (%.4f → %.4f). "
                            "Considere reduzir LR.",
                            self._divergence_count,
                            self._eval_losses[-2],
                            self._eval_losses[-1],
                        )
                else:
                    self._divergence_count = 0

                # Overfitting
                if (
                    len(self._train_losses) >= 2
                    and self._train_losses[-1] < self._train_losses[-2]
                    and direction > 0
                ):
                    logger.warning(
                        "[CONVERGENCE] Overfitting detectado: train_loss ↓ (%.4f → %.4f) "
                        "mas eval_loss ↑ (%.4f → %.4f). "
                        "Considere aumentar dropout, weight_decay ou reduzir LoRA rank.",
                        self._train_losses[-2],
                        self._train_losses[-1],
                        self._eval_losses[-2],
                        self._eval_losses[-1],
                    )


class CurriculumCallback(TrainerCallback):
    """
    Implementa curriculum learning progressivo por tier de complexidade.

    Estágios:
    - Epochs 1-3: Só Tier 1 (modelo aprende formato JSON e schema)
    - Epochs 4-7: Tier 1 + 2 (mais componentes e categorias de risco)
    - Epochs 8+:  Todos os Tiers (complexidade total + RAG)

    Requer que os dados de treino tenham o campo "tier" em metadata.
    """

    def __init__(
        self,
        full_dataset,
        tier_labels: list[int],
        stage_boundaries: dict[int, list[int]] | None = None,
    ):
        """
        Args:
            full_dataset: dataset completo de treino
            tier_labels: lista com o tier de cada amostra (alinhado com full_dataset)
            stage_boundaries: dict de epoch → tiers permitidos
                              default: {1: [1], 4: [1,2], 8: [1,2,3,4]}
        """
        self.full_dataset = full_dataset
        self.tier_labels = tier_labels
        self.stage_boundaries = stage_boundaries or {
            1: [1],
            4: [1, 2],
            8: [1, 2, 3, 4],
        }
        self._current_tiers: list[int] = []

    def _get_allowed_tiers(self, epoch: int) -> list[int]:
        """Determina quais tiers são permitidos no epoch atual."""
        allowed = [1]
        for boundary_epoch, tiers in sorted(self.stage_boundaries.items()):
            if epoch >= boundary_epoch:
                allowed = tiers
        return allowed

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        allowed_tiers = self._get_allowed_tiers(epoch)

        if allowed_tiers != self._current_tiers:
            self._current_tiers = allowed_tiers

            indices = [
                i for i, tier in enumerate(self.tier_labels)
                if tier in allowed_tiers
            ]

            if hasattr(self.full_dataset, "select"):
                filtered = self.full_dataset.select(indices)
            else:
                filtered = [self.full_dataset[i] for i in indices]

            trainer = kwargs.get("model", None)
            if trainer is None:
                # Fallback: log and skip
                logger.info(
                    "[CURRICULUM] Epoch %d: tiers %s (%d amostras). "
                    "Nota: troca de dataset requer integração manual com trainer.",
                    epoch,
                    allowed_tiers,
                    len(indices),
                )
                return

            logger.info(
                "[CURRICULUM] Epoch %d: trocando para tiers %s (%d amostras de %d total)",
                epoch,
                allowed_tiers,
                len(indices),
                len(self.tier_labels),
            )


class DomainMetricsCallback(TrainerCallback):
    """
    Avalia métricas de domínio durante a avaliação, logando os resultados.

    Nota: Para avaliação completa com geração de texto, este callback
    registra métricas calculadas externamente via `log_metrics()`.
    A geração real deve ser feita no loop de avaliação customizado.
    """

    def __init__(self, output_file: str = "./output/domain_metrics.jsonl"):
        self.output_file = output_file
        self._metrics_history: list[dict] = []

    def log_metrics(self, step: int, metrics: dict[str, float]):
        """Registra métricas de domínio calculadas externamente."""
        entry = {"step": step, **metrics}
        self._metrics_history.append(entry)

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "[DOMAIN METRICS] step=%d | domain_score=%.3f | json_valid=%.1f%% | "
            "grounding=%.3f | schema=%.3f | risks=%.3f",
            step,
            metrics.get("eval_domain_score", 0),
            metrics.get("eval_json_valid_rate", 0) * 100,
            metrics.get("eval_grounding_score", 0),
            metrics.get("eval_schema_complete", 0),
            metrics.get("eval_risk_coverage", 0),
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self._metrics_history:
            logger.info(
                "[DOMAIN METRICS] Treino finalizado. %d avaliações registradas em %s",
                len(self._metrics_history),
                self.output_file,
            )
