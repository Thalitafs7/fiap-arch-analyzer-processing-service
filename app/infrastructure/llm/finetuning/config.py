"""
Configurações de hiperparâmetros para o fine-tuning QLoRA.
Separado do config principal (pydantic-settings) pois este módulo
roda como script standalone em GPU, fora do Docker.
"""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    r: int = 8                           # rank reduzido para evitar overfitting com poucos dados
    lora_alpha: int = 16                 # escala (alpha/r = 2 mantido)
    lora_dropout: float = 0.1            # dropout aumentado para regularização
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Camadas alvo — padrão para modelos decoder-only (Mistral, LLaMA, Phi, Qwen)
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"    # "bfloat16" se GPU suportar
    bnb_4bit_quant_type: str = "nf4"           # NormalFloat4 — melhor para QLoRA
    bnb_4bit_use_double_quant: bool = True      # quantização dupla economiza ~0.4 bits/param


@dataclass
class TrainingConfig:
    # Modelo base — troque pelo modelo escolhido pelo time
    base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Saída
    output_dir: str = "./output/report-lora-adapter"
    hub_model_id: str = ""                     # ex: "seu-usuario/report-lora" (HuggingFace Hub)
    push_to_hub: bool = False

    # Dados
    train_file: str = "./data/train.jsonl"
    val_file: str = "./data/val.jsonl"
    max_seq_length: int = 4096

    # Treino — otimizado para convergência
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2       # batch menor = mais steps por epoch
    gradient_accumulation_steps: int = 2       # batch efetivo = 2 × 2 = 4 (mais updates)
    learning_rate: float = 5e-5                # LR conservadora para QLoRA em modelo instruction-tuned
    warmup_ratio: float = 0.1                  # warmup real (>1 step)
    weight_decay: float = 0.01                 # regularização L2
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"            # otimizador eficiente para QLoRA

    # Avaliação e salvamento — monitoramento frequente
    eval_strategy: str = "steps"
    eval_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Logging — visibilidade total da convergência
    logging_steps: int = 1
    logging_first_step: bool = True
    report_to: str = "tensorboard"
    logging_dir: str = "./output/logs"


@dataclass
class DataGenerationConfig:
    num_synthetic_samples: int = 500
    train_split: float = 0.9                   # 90% treino, 10% validação
    output_dir: str = "./data"
    rag_sample_ratio: float = 0.3              # 30% das amostras incluem contexto RAG sintético

    # Templates organizados por tier de complexidade (curriculum learning)
    tier_1_simple: list = field(default_factory=lambda: [
        # 5-7 componentes, 1-2 categorias de risco
        "static_website_cdn",
        "simple_crud_api",
        "single_container_app",
        "basic_queue_worker",
        "wordpress_lamp",
    ])
    tier_2_intermediate: list = field(default_factory=lambda: [
        # 8-12 componentes, 3-4 categorias de risco
        "microservices_api_gateway",
        "monolith_single_db",
        "event_driven_kafka",
        "serverless_lambda_dynamodb",
        "kubernetes_service_mesh",
        "cqrs_event_sourcing",
        "bff_mobile_web",
        "data_pipeline_etl",
        "hexagonal_clean_arch",
        "multi_region_failover",
        "saga_pattern_distributed",
        "api_composition_gateway",
        "strangler_fig_migration",
        "blue_green_deployment",
        "feature_flag_service",
    ])
    tier_3_complex: list = field(default_factory=lambda: [
        # 12-20 componentes, 5-6 categorias de risco
        "multi_cloud_hybrid",
        "streaming_platform_kafka_flink",
        "ml_inference_pipeline",
        "zero_trust_network",
        "event_mesh_choreography",
        "polyglot_persistence",
        "global_edge_computing",
    ])
    tier_4_expert: list = field(default_factory=lambda: [
        # 20+ componentes, todas as 6 categorias, com RAG
        "banking_core_modernization",
        "healthcare_hipaa_platform",
        "autonomous_vehicle_platform",
    ])

    # Variações por tier
    variations_tier_1: int = 20
    variations_tier_2: int = 15
    variations_tier_3: int = 10
    variations_tier_4: int = 10

    @property
    def all_templates(self) -> list:
        return self.tier_1_simple + self.tier_2_intermediate + self.tier_3_complex + self.tier_4_expert

    def get_tier(self, template: str) -> int:
        if template in self.tier_1_simple:
            return 1
        if template in self.tier_2_intermediate:
            return 2
        if template in self.tier_3_complex:
            return 3
        if template in self.tier_4_expert:
            return 4
        return 2  # default

    def get_variations(self, template: str) -> int:
        tier = self.get_tier(template)
        return {1: self.variations_tier_1, 2: self.variations_tier_2,
                3: self.variations_tier_3, 4: self.variations_tier_4}[tier]
