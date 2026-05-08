"""
Módulo de fine-tuning para geração de relatórios técnicos.

Estrutura:
  config.py          — hiperparâmetros de treino
  data_generator.py  — geração de dados sintéticos via LLM (professor)
  data_formatter.py  — conversão para JSONL formato chat
  train.py           — script QLoRA (roda em GPU externa, ex: Colab)
  inference.py       — cliente de inferência (HuggingFace API ou local)
"""
