"""
Data Formatter — converte pares brutos para JSONL no formato chat instruction-tuning.

Formato de saída (compatível com SFTTrainer do trl e maioria dos LLMs):
{
  "messages": [
    {"role": "system",    "content": "<system prompt>"},
    {"role": "user",      "content": "<extraction + risks + rag formatados>"},
    {"role": "assistant", "content": "<JSON do relatório>"}
  ]
}

Uso:
    python -m app.finetuning.data_formatter \
        --input  ./data/raw_pairs.jsonl \
        --output ./data \
        --split  0.9
"""

import argparse
import json
import random
from pathlib import Path

from app.infrastructure.llm.finetuning.prompts import SYSTEM_PROMPT, build_user_message


def _validate_pair(pair: dict) -> bool:
    """Valida que o par tem todos os campos obrigatórios e JSON parseable."""
    try:
        report = pair.get("report", {})
        required_keys = {"components_identified", "architectural_risks", "recommendations", "executive_summary"}
        if not required_keys.issubset(report.keys()):
            return False
        if not report.get("components_identified"):
            return False
        if not report.get("recommendations"):
            return False
        summary = report.get("executive_summary", "")
        if not summary or len(summary) < 50:
            return False
        # Valida que o report é serializável em JSON
        json.dumps(report, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


def _load_jsonl(path: str) -> tuple[list[dict], int]:
    """Carrega e valida pares de um arquivo JSONL. Retorna (pares_válidos, skipped)."""
    pairs = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                if _validate_pair(pair):
                    pairs.append(pair)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    return pairs, skipped


def format_pairs(
    input_path: str,
    output_dir: str,
    train_split: float = 0.9,
    seed: int = 42,
    extra_sources: list[str] | None = None,
) -> tuple[int, int]:
    """
    Lê o JSONL bruto, converte para formato chat e grava train.jsonl e val.jsonl.

    Suporta múltiplas fontes de dados: o arquivo principal (sintético) +
    fontes extras (ex: dataset HuggingFace adaptado via hf_dataset_adapter.py).

    Args:
        input_path:    Caminho do arquivo raw_pairs.jsonl (sintético).
        output_dir:    Diretório de saída para train.jsonl e val.jsonl.
        train_split:   Fração dos dados para treino (padrão: 0.9).
        seed:          Seed para shuffle reproduzível.
        extra_sources: Lista de caminhos JSONL extras para combinar.

    Returns:
        Tupla (n_train, n_val).
    """
    raw_pairs: list[dict] = []
    total_skipped = 0

    # Carregar fonte principal
    if Path(input_path).exists():
        pairs, skipped = _load_jsonl(input_path)
        raw_pairs.extend(pairs)
        total_skipped += skipped
        print(f"[sintético] {len(pairs)} válidos, {skipped} ignorados ← {input_path}")

    # Carregar fontes extras (ex: HuggingFace adaptado)
    for source_path in (extra_sources or []):
        if Path(source_path).exists():
            pairs, skipped = _load_jsonl(source_path)
            raw_pairs.extend(pairs)
            total_skipped += skipped
            print(f"[extra]     {len(pairs)} válidos, {skipped} ignorados ← {source_path}")
        else:
            print(f"[extra]     AVISO: arquivo não encontrado — {source_path}")

    print(f"\nTotal: {len(raw_pairs)} pares válidos, {total_skipped} ignorados")

    random.seed(seed)
    random.shuffle(raw_pairs)

    split_idx = int(len(raw_pairs) * train_split)
    train_pairs = raw_pairs[:split_idx]
    val_pairs = raw_pairs[split_idx:]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_path = Path(output_dir) / "train.jsonl"
    val_path = Path(output_dir) / "val.jsonl"

    for path, pairs in [(train_path, train_pairs), (val_path, val_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                rag_result = pair.get("rag_context")
                example = {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": build_user_message(
                            pair["extraction"], pair["risks"], rag_result
                        )},
                        {"role": "assistant", "content": json.dumps(
                            pair["report"], ensure_ascii=False
                        )},
                    ]
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✅ Treino:  {len(train_pairs)} exemplos → {train_path}")
    print(f"✅ Val:     {len(val_pairs)} exemplos  → {val_path}")
    return len(train_pairs), len(val_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formata dados brutos para fine-tuning")
    parser.add_argument("--input",  default="./data/raw_pairs.jsonl", help="JSONL bruto (sintético)")
    parser.add_argument("--output", default="./data",                  help="Diretório de saída")
    parser.add_argument("--split",  type=float, default=0.9,           help="Fração treino")
    parser.add_argument("--seed",   type=int,   default=42,            help="Seed para shuffle")
    parser.add_argument("--extra",  nargs="*",  default=[],            help="JSONLs extras (ex: HF adaptado)")
    args = parser.parse_args()

    format_pairs(args.input, args.output, args.split, args.seed, extra_sources=args.extra)
