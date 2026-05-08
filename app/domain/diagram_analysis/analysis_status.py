from enum import Enum


class AnalysisStatus(str, Enum):
    RECEIVED = "recebido"
    PROCESSING = "em_processamento"
    ANALYZED = "analisado"
    ERROR = "erro"
