"""
Inference Client — abstração para usar o LLM fine-tunado como backend do report_agent.

Dois modos de operação:
  - HuggingFaceAPIClient : usa HuggingFace Inference API (recomendado para produção/hackathon)
  - LocalModelClient     : carrega modelo + adapter localmente (requer GPU)

Uso no report_agent.py:
    from app.infrastructure.llm.finetuning.inference import get_report_client
    client = get_report_client(settings)
    result = client.generate_report(extraction_result, risk_result, rag_result)
"""

import json
from abc import ABC, abstractmethod

from app.shared.logging import get_logger
from app.infrastructure.llm.finetuning.prompts import SYSTEM_PROMPT, build_user_message

logger = get_logger(__name__)


def _parse_json_response(raw: str) -> dict:
    """Remove blocos markdown e parseia o JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(raw)


# ──────────────────────────────────────────────
# Interface abstrata
# ──────────────────────────────────────────────

class ReportModelClient(ABC):
    @abstractmethod
    def generate_report(
        self,
        extraction_result: dict,
        risk_result: dict,
        rag_result: dict | None = None,
    ) -> dict:
        """
        Gera o relatório técnico estruturado.

        Returns:
            dict com: components_identified, architectural_risks,
                      recommendations, executive_summary, rag_used
        Raises:
            ReportGenerationError se o LLM retornar JSON inválido ou incompleto.
        """


# ──────────────────────────────────────────────
# HuggingFace Inference API
# ──────────────────────────────────────────────

class HuggingFaceAPIClient(ReportModelClient):
    """
    Usa um endpoint de inferência no HuggingFace (modelo fine-tunado hospedado).
    Compatível com: Inference Endpoints, Inference API (serverless) e TGI.
    """

    def __init__(self, api_token: str, endpoint_url: str, timeout: float = 120.0):
        self._token = api_token
        self._endpoint_url = endpoint_url.rstrip("/")
        self._timeout = timeout

    def generate_report(
        self,
        extraction_result: dict,
        risk_result: dict,
        rag_result: dict | None = None,
    ) -> dict:
        from app.shared.exceptions import ReportGenerationError

        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError("huggingface_hub não instalado. Execute: pip install huggingface_hub")

        client = InferenceClient(
            model=self._endpoint_url,
            token=self._token,
            timeout=self._timeout,
        )

        user_message = build_user_message(extraction_result, risk_result, rag_result)

        logger.info("finetuned.inference.start", backend="huggingface", endpoint=self._endpoint_url[:60])

        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=4096,
                temperature=0.1,    # baixa temperatura → saída mais determinística e JSON-friendly
            )
            raw = response.choices[0].message.content
        except Exception as exc:
            raise ReportGenerationError(
                f"Erro na chamada HuggingFace Inference API: {exc}",
                step="report",
            )

        try:
            result = _parse_json_response(raw)
        except json.JSONDecodeError as exc:
            raise ReportGenerationError(
                f"LLM fine-tunado retornou JSON inválido: {exc}\nRaw: {raw[:300]}",
                step="report",
            )

        logger.info(
            "finetuned.inference.done",
            backend="huggingface",
            components=len(result.get("components_identified", [])),
        )
        return result


# ──────────────────────────────────────────────
# Modelo local (GPU)
# ──────────────────────────────────────────────

class LocalModelClient(ReportModelClient):
    """
    Carrega o modelo base + LoRA adapter localmente.
    Requer GPU com VRAM suficiente (≥ 16 GB para 7B em 4-bit).
    """

    def __init__(self, model_path: str, base_model_id: str = ""):
        self._model_path = model_path
        self._base_model_id = base_model_id
        self._pipeline = None

    def _load(self):
        """Carrega o pipeline lazy (apenas na primeira chamada)."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        except ImportError:
            raise RuntimeError(
                "Dependências de GPU não instaladas.\n"
                "Execute: pip install -r finetuning-requirements.txt"
            )

        logger.info("finetuned.local.loading", model_path=self._model_path)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_id = self._base_model_id or self._model_path
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, self._model_path)

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True,
        )
        logger.info("finetuned.local.loaded")

    def generate_report(
        self,
        extraction_result: dict,
        risk_result: dict,
        rag_result: dict | None = None,
    ) -> dict:
        from app.shared.exceptions import ReportGenerationError

        self._load()

        user_message = build_user_message(extraction_result, risk_result, rag_result)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        logger.info("finetuned.inference.start", backend="local")

        try:
            output = self._pipeline(messages)
            raw = output[0]["generated_text"][-1]["content"]
        except Exception as exc:
            raise ReportGenerationError(f"Erro na inferência local: {exc}", step="report")

        try:
            result = _parse_json_response(raw)
        except json.JSONDecodeError as exc:
            raise ReportGenerationError(
                f"LLM local retornou JSON inválido: {exc}\nRaw: {raw[:300]}",
                step="report",
            )

        logger.info("finetuned.inference.done", backend="local")
        return result


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def get_report_client(settings) -> ReportModelClient:
    """
    Retorna o client de inferência configurado via settings.

    REPORT_MODEL_BACKEND:
      "finetuned_api"   → HuggingFaceAPIClient (padrão para hackathon)
      "finetuned_local" → LocalModelClient (requer GPU local)
    """
    backend = getattr(settings, "report_model_backend", "langchain")

    if backend == "finetuned_api":
        token = getattr(settings, "huggingface_api_token", "")
        endpoint = getattr(settings, "huggingface_endpoint_url", "")
        if not token or not endpoint:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN e HUGGINGFACE_ENDPOINT_URL são obrigatórios "
                "quando REPORT_MODEL_BACKEND=finetuned_api"
            )
        return HuggingFaceAPIClient(api_token=token, endpoint_url=endpoint)

    if backend == "finetuned_local":
        model_path = getattr(settings, "local_model_path", "")
        base_model = getattr(settings, "base_model_id", "")
        if not model_path:
            raise ValueError(
                "LOCAL_MODEL_PATH é obrigatório quando REPORT_MODEL_BACKEND=finetuned_local"
            )
        return LocalModelClient(model_path=model_path, base_model_id=base_model)

    raise ValueError(
        f"Backend desconhecido: '{backend}'. "
        "Valores válidos: 'finetuned_api', 'finetuned_local'"
    )
