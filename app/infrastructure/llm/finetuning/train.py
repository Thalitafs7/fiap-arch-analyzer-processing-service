"""
Script de treino QLoRA — executa FORA do Docker, em máquina com GPU.

Recomendado: Google Colab Pro (A100), RunPod, Lambda Labs ou qualquer GPU com >= 16GB VRAM.

Dependências (instalar via finetuning-requirements.txt):
    pip install -r finetuning-requirements.txt

Uso:
    # 1. Gerar dados
    python -m app.finetuning.data_generator --samples 500 --output ./data/raw_pairs.jsonl

    # 2. Formatar
    python -m app.finetuning.data_formatter --input ./data/raw_pairs.jsonl --output ./data

    # 3. Treinar
    python -m app.finetuning.train

    # 4. (Opcional) Enviar para HuggingFace Hub
    python -m app.finetuning.train --push-to-hub --hub-model-id "seu-usuario/report-lora"
"""

import argparse
import json
import sys
from pathlib import Path

from app.infrastructure.llm.finetuning.config import (
    LoRAConfig,
    QuantizationConfig,
    TrainingConfig,
)


def _check_dependencies():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets", "bitsandbytes"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"ERRO: Dependências de GPU não instaladas: {missing}\n"
            "Execute: pip install -r finetuning-requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)


def train(
    base_model_id: str = "",
    train_file: str = "",
    val_file: str = "",
    output_dir: str = "",
    hub_model_id: str = "",
    push_to_hub: bool = False,
    num_epochs: int = 0,
    batch_size: int = 0,
    grad_accum: int = 0,
    learning_rate: float = 0,
    max_seq_length: int = 0,
    use_curriculum: bool = False,
):
    """Executa o fine-tuning QLoRA com monitoramento de convergência."""
    _check_dependencies()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from trl import SFTTrainer

    from app.infrastructure.llm.finetuning.callbacks import (
        ConvergenceMonitorCallback,
        DomainMetricsCallback,
    )

    # Carregar configurações dos dataclasses (single source of truth)
    lora_cfg = LoRAConfig()
    quant_cfg = QuantizationConfig()
    train_cfg = TrainingConfig()

    # CLI args override config defaults (0/empty = use config)
    base_model_id = base_model_id or train_cfg.base_model_id
    train_file = train_file or train_cfg.train_file
    val_file = val_file or train_cfg.val_file
    output_dir = output_dir or train_cfg.output_dir
    num_epochs = num_epochs or train_cfg.num_train_epochs
    batch_size = batch_size or train_cfg.per_device_train_batch_size
    grad_accum = grad_accum or train_cfg.gradient_accumulation_steps
    learning_rate = learning_rate or train_cfg.learning_rate
    max_seq_length = max_seq_length or train_cfg.max_seq_length

    print(f"GPU disponível: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("AVISO: Sem GPU detectada. O treino será extremamente lento.", file=sys.stderr)

    # ── Quantização 4-bit (QLoRA) ──────────────────────────────────
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )

    # ── Carregar modelo base ────────────────────────────────────────
    print(f"Carregando modelo base: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── LoRA (usando config centralizado) ──────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=lora_cfg.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ─────────────────────────────────────────────────────
    for path in [train_file, val_file]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}\n"
                "Execute data_generator.py e data_formatter.py primeiro."
            )

    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
    )
    print(f"Dataset: {len(dataset['train'])} treino, {len(dataset['validation'])} val")

    # Análise de token lengths para diagnóstico
    _analyze_token_lengths(dataset["train"], tokenizer, max_seq_length)

    # ── Argumentos de treino (otimizados para convergência) ────────
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        optim=train_cfg.optim,
        eval_strategy=train_cfg.eval_strategy,
        eval_steps=train_cfg.eval_steps,
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        logging_steps=train_cfg.logging_steps,
        logging_first_step=train_cfg.logging_first_step,
        report_to=train_cfg.report_to,
        logging_dir=train_cfg.logging_dir,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    # ── Callbacks de convergência ───────────────────────────────────
    convergence_cb = ConvergenceMonitorCallback(
        stagnation_threshold=1e-4,
        stagnation_patience=3,
        divergence_patience=2,
    )
    domain_metrics_cb = DomainMetricsCallback(
        output_file=str(Path(output_dir) / "domain_metrics.jsonl"),
    )
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=3,
    )

    callbacks = [convergence_cb, domain_metrics_cb, early_stopping_cb]

    # ── SFTTrainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        max_seq_length=max_seq_length,
        callbacks=callbacks,
    )

    print("Iniciando treino...")
    print(f"  Hiperparâmetros: LR={learning_rate}, LoRA r={lora_cfg.r}, "
          f"alpha={lora_cfg.lora_alpha}, dropout={lora_cfg.lora_dropout}")
    print(f"  Batch efetivo: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"  Steps/epoch estimado: ~{len(dataset['train']) // (batch_size * grad_accum)}")
    print(f"  Convergence monitoring: ON | Early stopping: patience=3")
    print(f"  Report to: {train_cfg.report_to}")

    trainer.train()

    # ── Salvar adapter ───────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter salvo em: {output_dir}")

    # Salvar metadados do treino
    meta = {
        "base_model_id": base_model_id,
        "num_epochs": num_epochs,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
        "lora_r": lora_cfg.r,
        "lora_alpha": lora_cfg.lora_alpha,
        "lora_dropout": lora_cfg.lora_dropout,
        "learning_rate": learning_rate,
        "batch_size_effective": batch_size * grad_accum,
        "warmup_ratio": train_cfg.warmup_ratio,
        "weight_decay": train_cfg.weight_decay,
        "max_seq_length": max_seq_length,
        "compute_dtype": "bf16" if use_bf16 else "fp16",
    }
    with open(Path(output_dir) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if push_to_hub and hub_model_id:
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print(f"Adapter publicado no Hub: {hub_model_id}")

    return output_dir


def _analyze_token_lengths(dataset, tokenizer, max_seq_length: int):
    """Analisa distribuição de token lengths para diagnóstico."""
    lengths = []
    for example in dataset:
        messages = example.get("messages", [])
        text = " ".join(m.get("content", "") for m in messages)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

    if not lengths:
        return

    lengths.sort()
    n = len(lengths)
    p50 = lengths[n // 2]
    p90 = lengths[int(n * 0.9)]
    p95 = lengths[int(n * 0.95)]
    p99 = lengths[min(int(n * 0.99), n - 1)]
    max_len = lengths[-1]

    print(f"Token lengths: p50={p50}, p90={p90}, p95={p95}, p99={p99}, max={max_len}")

    truncated = sum(1 for l in lengths if l > max_seq_length)
    if truncated > 0:
        print(f"  AVISO: {truncated}/{n} amostras ({truncated/n*100:.1f}%) excedem "
              f"max_seq_length={max_seq_length} e serão truncadas.")

    if p95 < max_seq_length // 2:
        print(f"  DICA: p95={p95} < {max_seq_length // 2}. Considere reduzir "
              f"max_seq_length para {p95 + 256} para economizar memória.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA para geração de relatórios")
    parser.add_argument("--model",          default="")
    parser.add_argument("--train-file",     default="")
    parser.add_argument("--val-file",       default="")
    parser.add_argument("--output-dir",     default="")
    parser.add_argument("--hub-model-id",   default="")
    parser.add_argument("--push-to-hub",    action="store_true")
    parser.add_argument("--epochs",         type=int,   default=0)
    parser.add_argument("--batch-size",     type=int,   default=0)
    parser.add_argument("--grad-accum",     type=int,   default=0)
    parser.add_argument("--lr",             type=float, default=0)
    parser.add_argument("--max-seq-length", type=int,   default=0)
    parser.add_argument("--curriculum",     action="store_true", help="Ativar curriculum learning")
    args = parser.parse_args()

    train(
        base_model_id=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        use_curriculum=args.curriculum,
    )
