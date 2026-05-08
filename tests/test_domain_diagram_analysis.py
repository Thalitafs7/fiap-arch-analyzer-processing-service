"""Testes unitários para o Bounded Context de análise de diagrama."""

import base64
import pytest

from app.domain.shared.analysis_id import AnalysisId
from app.domain.diagram_analysis.analysis import AnalysisAggregate
from app.domain.diagram_analysis.analysis_status import AnalysisStatus
from app.domain.diagram_analysis.component import Component, Relationship, ArchitecturalPattern
from app.domain.diagram_analysis.diagram_file import DiagramFile
from app.domain.diagram_analysis.extraction_result import ExtractionResult
from app.domain.diagram_analysis.file_type import FileType

PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# FileType
# ---------------------------------------------------------------------------

class TestFileType:
    @pytest.mark.parametrize("mime,expected", [
        ("image/png", FileType.PNG),
        ("image/jpeg", FileType.JPEG),
        ("image/jpg", FileType.JPEG),
        ("image/gif", FileType.GIF),
        ("image/webp", FileType.WEBP),
        ("application/pdf", FileType.PDF),
    ])
    def test_from_mime_supported(self, mime, expected):
        assert FileType.from_mime(mime) == expected

    def test_from_mime_unsupported_raises(self):
        with pytest.raises(ValueError, match="não suportado"):
            FileType.from_mime("text/plain")

    def test_mime_type_roundtrip(self):
        for ft in FileType:
            assert FileType.from_mime(ft.mime_type) == ft


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class TestComponent:
    def test_valid_component(self):
        c = Component(name="API Gateway")
        assert str(c) == "API Gateway"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            Component(name="")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            Component(name="   ")

    def test_matches_case_insensitive(self):
        a = Component("Lambda")
        b = Component("lambda")
        assert a.matches(b)

    def test_not_matches_different_name(self):
        assert not Component("Lambda").matches(Component("SQS"))

    def test_frozen(self):
        c = Component("API")
        with pytest.raises((AttributeError, TypeError)):
            c.name = "outro"


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

class TestRelationship:
    def test_from_string_arrow_unicode(self):
        r = Relationship.from_string("API Gateway → Lambda: invoca")
        assert r.source == "API Gateway"
        assert r.target == "Lambda"
        assert r.description == "invoca"

    def test_from_string_arrow_ascii(self):
        r = Relationship.from_string("A -> B: chama")
        assert r.source == "A"
        assert r.target == "B"
        assert r.description == "chama"

    def test_from_string_no_description(self):
        r = Relationship.from_string("A → B")
        assert r.source == "A"
        assert r.target == "B"
        assert r.description == ""

    def test_from_string_unparseable_falls_back(self):
        raw = "sem formato valido"
        r = Relationship.from_string(raw)
        assert r.description == raw

    def test_str_with_source_and_target(self):
        r = Relationship(source="A", target="B", description="usa")
        assert str(r) == "A → B: usa"

    def test_str_without_source(self):
        r = Relationship(source="", target="", description="texto livre")
        assert str(r) == "texto livre"


# ---------------------------------------------------------------------------
# ArchitecturalPattern
# ---------------------------------------------------------------------------

class TestArchitecturalPattern:
    def test_str(self):
        p = ArchitecturalPattern(name="Event-driven")
        assert str(p) == "Event-driven"


# ---------------------------------------------------------------------------
# DiagramFile
# ---------------------------------------------------------------------------

class TestDiagramFile:
    def test_create_from_png_bytes(self):
        df = DiagramFile.create(PNG_1PX, "diagrama.png")
        assert df.file_type == FileType.PNG
        assert df.media_type == "image/png"
        assert df.content_base64 == base64.standard_b64encode(PNG_1PX).decode()
        assert df.file_size_kb > 0

    def test_create_rejects_oversized_file(self):
        big = b"x" * (21 * 1024 * 1024)
        with pytest.raises(ValueError, match="limite"):
            DiagramFile.create(big, "grande.png")

    def test_create_rejects_unsupported_mime(self):
        with pytest.raises(ValueError):
            DiagramFile.create(b"data", "arquivo.txt")

    def test_to_dict_keys(self):
        df = DiagramFile.create(PNG_1PX, "d.png")
        d = df.to_dict()
        assert d["status"] == "recebido"
        assert d["file_type"] == "png"
        assert "content_base64" in d


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------

class TestExtractionResult:
    BASE = {
        "components": ["API Gateway", "Lambda"],
        "relationships": ["API Gateway → Lambda: invoca"],
        "patterns": ["Serverless"],
        "raw_description": "Arquitetura serverless.",
    }

    def test_from_dict(self):
        er = ExtractionResult.from_dict(self.BASE)
        assert len(er.components) == 2
        assert len(er.relationships) == 1
        assert len(er.patterns) == 1
        assert er.raw_description == "Arquitetura serverless."

    def test_component_names(self):
        er = ExtractionResult.from_dict(self.BASE)
        assert "API Gateway" in er.component_names

    def test_has_component_case_insensitive(self):
        er = ExtractionResult.from_dict(self.BASE)
        assert er.has_component("lambda")
        assert not er.has_component("S3")

    def test_to_dict_status(self):
        er = ExtractionResult.from_dict(self.BASE)
        d = er.to_dict()
        assert d["status"] == "em_processamento"
        assert "API Gateway" in d["components"]

    def test_from_dict_empty_lists(self):
        er = ExtractionResult.from_dict({})
        assert er.components == []
        assert er.relationships == []
        assert er.patterns == []


# ---------------------------------------------------------------------------
# AnalysisAggregate
# ---------------------------------------------------------------------------

class TestAnalysisAggregate:
    def _make(self, file_name="d.png", file_type="png"):
        return AnalysisAggregate.create(
            analysis_id=AnalysisId.generate(),
            file_name=file_name,
            file_type=file_type,
        )

    def _diagram_file(self):
        return DiagramFile.create(PNG_1PX, "d.png")

    def _extraction(self):
        return ExtractionResult.from_dict({
            "components": ["API Gateway", "Lambda"],
            "relationships": ["API Gateway → Lambda: invoca"],
            "patterns": ["Serverless"],
            "raw_description": "desc",
        })

    def test_create_status_is_received(self):
        agg = self._make()
        assert agg.status == AnalysisStatus.RECEIVED

    def test_create_emits_diagram_received_event(self):
        agg = self._make()
        events = agg.pull_events()
        assert len(events) == 1
        assert events[0].event_name == "DiagramReceivedEvent"

    def test_pull_events_clears_list(self):
        agg = self._make()
        agg.pull_events()
        assert agg.pull_events() == []

    def test_start_ingestion_transitions_to_processing(self):
        agg = self._make()
        agg.pull_events()
        agg.start_ingestion(self._diagram_file())
        assert agg.status == AnalysisStatus.PROCESSING

    def test_start_ingestion_emits_ingested_event(self):
        agg = self._make()
        agg.pull_events()
        agg.start_ingestion(self._diagram_file())
        events = agg.pull_events()
        assert any(e.event_name == "DiagramIngestedEvent" for e in events)

    def test_start_ingestion_wrong_state_raises(self):
        agg = self._make()
        agg.pull_events()
        agg.start_ingestion(self._diagram_file())
        with pytest.raises(ValueError, match="start_ingestion"):
            agg.start_ingestion(self._diagram_file())

    def test_complete_extraction_stores_result(self):
        agg = self._make()
        agg.pull_events()
        agg.start_ingestion(self._diagram_file())
        agg.complete_extraction(self._extraction())
        assert agg.extraction_result is not None

    def test_complete_extraction_emits_event(self):
        agg = self._make()
        agg.pull_events()
        agg.start_ingestion(self._diagram_file())
        agg.pull_events()
        agg.complete_extraction(self._extraction())
        events = agg.pull_events()
        assert any(e.event_name == "ComponentsExtractedEvent" for e in events)

    def test_complete_marks_analyzed(self):
        agg = self._make()
        agg.pull_events()
        agg.start_ingestion(self._diagram_file())
        agg.complete_extraction(self._extraction())
        agg.complete(qa_score=0.9)
        assert agg.status == AnalysisStatus.ANALYZED

    def test_complete_wrong_state_raises(self):
        agg = self._make()
        with pytest.raises(ValueError, match="complete"):
            agg.complete(qa_score=0.9)

    def test_fail_sets_error_state(self):
        agg = self._make()
        agg.fail(step="extraction", error_message="timeout")
        assert agg.status == AnalysisStatus.ERROR
        assert agg.error_message == "timeout"

    def test_fail_from_any_state(self):
        agg = self._make()
        agg.pull_events()
        agg.start_ingestion(self._diagram_file())
        agg.fail(step="extraction", error_message="erro")
        assert agg.status == AnalysisStatus.ERROR

    def test_fail_emits_failed_event(self):
        agg = self._make()
        agg.pull_events()
        agg.fail(step="ingestion", error_message="bad file")
        events = agg.pull_events()
        assert any(e.event_name == "AnalysisFailedEvent" for e in events)

    def test_full_happy_path_event_sequence(self):
        agg = self._make()
        all_events = []
        all_events.extend(agg.pull_events())

        agg.start_ingestion(self._diagram_file())
        all_events.extend(agg.pull_events())

        agg.complete_extraction(self._extraction())
        all_events.extend(agg.pull_events())

        agg.complete(qa_score=0.85)
        all_events.extend(agg.pull_events())

        names = [e.event_name for e in all_events]
        assert names == [
            "DiagramReceivedEvent",
            "DiagramIngestedEvent",
            "ComponentsExtractedEvent",
            "AnalysisCompletedEvent",
        ]
