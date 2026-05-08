"""Testes unitários para o Shared Kernel do domínio (IDs e eventos)."""

import uuid
import pytest
from datetime import datetime, timezone

from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.report_id import ReportId
from app.domain.shared.events.domain_event import DomainEvent


class TestAnalysisId:
    def test_generate_creates_valid_uuid(self):
        aid = AnalysisId.generate()
        uuid.UUID(aid.value)

    def test_from_string_valid_uuid(self):
        raw = str(uuid.uuid4())
        aid = AnalysisId.from_string(raw)
        assert aid.value == raw

    def test_from_string_invalid_uuid_raises(self):
        with pytest.raises(ValueError):
            AnalysisId("not-a-uuid")

    def test_str_returns_value(self):
        raw = str(uuid.uuid4())
        assert str(AnalysisId(raw)) == raw

    def test_equality(self):
        raw = str(uuid.uuid4())
        assert AnalysisId(raw) == AnalysisId(raw)

    def test_inequality(self):
        assert AnalysisId.generate() != AnalysisId.generate()

    def test_is_frozen(self):
        aid = AnalysisId.generate()
        with pytest.raises((AttributeError, TypeError)):
            aid.value = "outro"


class TestReportId:
    def test_generate_creates_valid_uuid(self):
        rid = ReportId.generate()
        uuid.UUID(rid.value)

    def test_from_string_valid(self):
        raw = str(uuid.uuid4())
        assert ReportId.from_string(raw).value == raw

    def test_invalid_uuid_raises(self):
        with pytest.raises(ValueError):
            ReportId("bad")

    def test_str_representation(self):
        raw = str(uuid.uuid4())
        assert str(ReportId(raw)) == raw


class TestDomainEvent:
    def test_event_name_is_class_name(self):
        event = DomainEvent()
        assert event.event_name == "DomainEvent"

    def test_occurred_at_is_utc(self):
        event = DomainEvent()
        assert event.occurred_at.tzinfo == timezone.utc

    def test_occurred_at_is_recent(self):
        before = datetime.now(timezone.utc)
        event = DomainEvent()
        after = datetime.now(timezone.utc)
        assert before <= event.occurred_at <= after
