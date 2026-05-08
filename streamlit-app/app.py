"""
Streamlit Frontend — Hackathon FIAP
Interface para análise de diagramas de arquitetura com progresso em tempo real.
Consome o ia-service via SSE (POST /analyze/stream) sem afetar o consumer SQS.
"""

import os
import json
import time
import traceback
import httpx
import streamlit as st

IA_SERVICE_URL = os.getenv("IA_SERVICE_URL", "http://ia-service:8000")
REPORT_API_URL = os.getenv("REPORT_API_URL", "http://report-api:8001")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))

SUPPORTED_TYPES = ["png", "jpg", "jpeg", "gif", "webp", "pdf"]

STEP_LABELS = {
    "ingestion": ("📁", "Ingestão"),
    "extraction": ("🔍", "Extração"),
    "rag": ("🔗", "RAG"),
    "report": ("📝", "Relatório + Riscos"),
    "qa": ("✅", "Validação QA"),
}

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisador de Arquitetura",
    page_icon="🏗️",
    layout="wide",
)

# ── Session state ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def _add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def _check_health() -> dict | None:
    try:
        r = httpx.get(f"{IA_SERVICE_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return None


def _fetch_history(limit: int = 10) -> list[dict]:
    try:
        r = httpx.get(
            f"{REPORT_API_URL}/reports",
            params={"limit": limit},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("items", [])
    except Exception:
        pass
    return []


def _format_step_summary(step: str, data: dict) -> str:
    """Formata o resumo de cada etapa para exibição."""
    if step == "ingestion":
        return f"{data.get('file_type', '?').upper()}, {data.get('file_size_kb', 0):.0f} KB"
    if step == "extraction":
        patterns = data.get("patterns", [])
        p = f", padrões: {', '.join(patterns[:3])}" if patterns else ""
        return f"{data.get('components_count', 0)} componentes, {data.get('relationships_count', 0)} relacionamentos{p}"
    if step == "rag":
        if data.get("has_context"):
            return f"{data.get('similar_count', 0)} análise(s) similar(es) encontrada(s)"
        return "Sem histórico, skip"
    if step == "report":
        sev = data.get("severity", {})
        risk_str = f"{sev.get('high', 0)} alto, {sev.get('medium', 0)} médio, {sev.get('low', 0)} baixo"
        return f"{data.get('risks_count', 0)} riscos ({risk_str}), {data.get('recommendations_count', 0)} recomendações"
    if step == "qa":
        score = data.get("completeness_score", 0)
        valid = "aprovado" if data.get("is_valid") else "rejeitado"
        return f"Score {score:.0%} — {valid}"
    return ""


def _async_analysis(file_bytes: bytes, file_name: str, status_container):
    """
    Análise assíncrona em 2 fases:
    Fase 1: submete o diagrama via POST /analyze/async → recebe job_id.
    Fase 2: assina SSE via GET /jobs/{job_id}/events → acompanha progresso.
    """
    # ── Fase 1: Submit ──
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{IA_SERVICE_URL}/analyze/async",
            files={"file": (file_name, file_bytes)},
        )
        if resp.status_code != 202:
            raise Exception(f"Erro ao submeter: {resp.status_code} - {resp.text}")
        job = resp.json()
        job_id = job["job_id"]

    # ── Fase 2: Subscribe SSE ──
    result = None
    completed_steps = []
    current_step = None

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        with client.stream(
            "GET",
            f"{IA_SERVICE_URL}/jobs/{job_id}/events",
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Erro ao conectar SSE: {response.status_code}")

            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    for line in raw_event.strip().split("\n"):
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        step = event.get("step", "")
                        event_status = event.get("status", "")
                        data = event.get("data", {})

                        if step == "done" and event_status == "complete":
                            result = data
                            continue

                        if event_status == "error":
                            error_msg = data.get("error", "Erro desconhecido")
                            server_tb = data.get("traceback", "")
                            icon, label = STEP_LABELS.get(step, ("❌", step))
                            completed_steps.append(f"❌ **{label}** — {error_msg}")
                            _redraw_steps(status_container, completed_steps, None)
                            exc = Exception(error_msg)
                            exc._server_traceback = server_tb
                            raise exc

                        if event_status == "running":
                            icon, label = STEP_LABELS.get(step, ("⏳", step))
                            current_step = f"⏳ **{label}** — Processando..."
                            _redraw_steps(status_container, completed_steps, current_step)

                        elif event_status == "done":
                            icon, label = STEP_LABELS.get(step, ("✅", step))
                            elapsed = data.get("elapsed", 0)
                            summary = _format_step_summary(step, data)
                            completed_steps.append(
                                f"✅ **{label}** ({elapsed}s) — {summary}"
                            )
                            current_step = None
                            _redraw_steps(status_container, completed_steps, current_step)

    return result


def _redraw_steps(container, completed: list[str], current: str | None):
    """Redesenha a lista de etapas no container do st.status."""
    container.empty()
    with container.container():
        for line in completed:
            st.markdown(line)
        if current:
            st.markdown(current)


def _render_severity_badge(severity: str) -> str:
    s = severity.upper()
    if s == "ALTO":
        return "🔴 ALTO"
    if s == "MÉDIO":
        return "🟡 MÉDIO"
    return "🟢 BAIXO"


def _render_report(result: dict):
    """Renderiza o relatório de análise."""
    report = result.get("report", {})
    qa = result.get("qa", {})

    summary = report.get("executive_summary", "")
    if summary:
        st.markdown("### 📋 Resumo Executivo")
        st.markdown(summary)
        st.divider()

    components = report.get("components_identified", [])
    if components:
        st.markdown("### 🧩 Componentes Identificados")
        cols = st.columns(3)
        for i, comp in enumerate(components):
            cols[i % 3].markdown(f"- **{comp}**")
        st.divider()

    risks = report.get("architectural_risks", [])
    if risks:
        st.markdown("### ⚠️ Riscos Arquiteturais")
        for risk in risks:
            severity = _render_severity_badge(risk.get("severity", "BAIXO"))
            risk_type = risk.get("type", "N/A")
            with st.expander(f"{severity} — {risk_type}: {risk.get('description', '')[:80]}"):
                st.markdown(f"**Tipo:** {risk_type}")
                st.markdown(f"**Severidade:** {severity}")
                st.markdown(f"**Descrição:** {risk.get('description', '')}")
                affected = risk.get("affected_components", [])
                if affected:
                    st.markdown(f"**Componentes afetados:** {', '.join(affected)}")
                mitigation = risk.get("mitigation", "")
                if mitigation:
                    st.markdown(f"**Mitigação:** {mitigation}")
        st.divider()

    recommendations = report.get("recommendations", [])
    if recommendations:
        st.markdown("### 💡 Recomendações")
        for rec in recommendations:
            prefix = "🔗 " if "[RAG]" in rec else "➡️ "
            st.markdown(f"{prefix} {rec}")
        st.divider()

    if qa:
        score = qa.get("completeness_score", 0)
        is_valid = qa.get("is_valid", False)
        st.markdown("### ✅ Qualidade do Relatório")
        col1, col2 = st.columns(2)
        col1.metric("Score de Completude", f"{score:.0%}")
        col2.metric("Válido", "Sim ✓" if is_valid else "Não ✗")

        issues = qa.get("issues_found", [])
        if issues:
            st.warning(f"Problemas encontrados: {', '.join(issues)}")

    if report.get("rag_used"):
        st.info("🔗 Este relatório foi enriquecido com contexto histórico via RAG.")

    st.download_button(
        label="📥 Baixar relatório (JSON)",
        data=json.dumps(result, indent=2, ensure_ascii=False, default=str),
        file_name=f"relatorio_{result.get('analysis_id', 'unknown')}.json",
        mime="application/json",
    )


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Analisador de Arquitetura")
    st.caption("Hackathon FIAP — IA para análise de diagramas")

    st.divider()

    health = _check_health()
    if health and health.get("status") == "healthy":
        st.success("🟢 IA Service online")
    else:
        st.error("🔴 IA Service offline")

    st.divider()

    st.markdown("### 📚 Histórico")
    history = _fetch_history(limit=10)
    if history:
        for item in history:
            aid = item.get("analysis_id", "")[:8]
            fname = item.get("file_name", "N/A")
            created = str(item.get("analysis_created_at", ""))[:16]
            st.caption(f"🗂️ `{aid}…` — {fname} ({created})")
    else:
        st.caption("Nenhuma análise anterior encontrada.")

    st.divider()
    st.caption("Pipeline: Ingestão → Extração → RAG → Relatório+Riscos → QA")


# ── Main area ───────────────────────────────────────────────────
st.title("🏗️ Análise de Diagramas de Arquitetura")
st.markdown(
    "Envie um diagrama de arquitetura (PNG, JPEG, PDF) e receba um relatório "
    "técnico com componentes, riscos e recomendações."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_file = st.file_uploader(
    "📎 Anexe seu diagrama de arquitetura",
    type=SUPPORTED_TYPES,
    help="Formatos aceitos: PNG, JPEG, GIF, WebP, PDF (máx. 20MB)",
)

if uploaded_file is not None:
    if uploaded_file.type and uploaded_file.type.startswith("image/"):
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    if st.button("🚀 Analisar Diagrama", type="primary", use_container_width=True):
        file_bytes = uploaded_file.getvalue()

        _add_message("user", f"📎 Enviando diagrama: **{uploaded_file.name}** ({len(file_bytes) / 1024:.1f} KB)")

        with st.chat_message("user"):
            st.markdown(f"📎 Enviando diagrama: **{uploaded_file.name}** ({len(file_bytes) / 1024:.1f} KB)")

        with st.chat_message("assistant"):
            with st.status("🔄 Analisando diagrama...", expanded=True) as status:
                step_display = st.empty()
                start = time.time()

                try:
                    result = _async_analysis(file_bytes, uploaded_file.name, step_display)
                    elapsed = time.time() - start

                    status.update(
                        label=f"✅ Análise concluída em {elapsed:.1f}s",
                        state="complete",
                        expanded=True,
                    )

                    st.session_state.analysis_result = result
                    _add_message(
                        "assistant",
                        f"✅ Análise concluída! ID: `{result.get('analysis_id', 'N/A')}`",
                    )

                except Exception as e:
                    elapsed = time.time() - start
                    status.update(
                        label=f"❌ Erro após {elapsed:.1f}s",
                        state="error",
                    )
                    st.error(f"**{type(e).__name__}:** {e}")

                    # Traceback do servidor (via SSE)
                    server_tb = getattr(e, "_server_traceback", "")
                    if server_tb:
                        with st.expander("🔍 Stack trace (servidor)", expanded=True):
                            st.code(server_tb, language="python")

                    # Traceback local do Streamlit
                    local_tb = traceback.format_exc()
                    if local_tb and local_tb.strip() != "NoneType: None":
                        with st.expander("🔍 Stack trace (local)", expanded=False):
                            st.code(local_tb, language="python")

                    _add_message("assistant", f"❌ Erro na análise: {e}")

if st.session_state.analysis_result:
    st.divider()
    _render_report(st.session_state.analysis_result)
