/**
 * Helper para chamadas API diretas (sem Streamlit).
 */

const IA_SERVICE_URL = 'http://localhost:8000';
const REPORT_API_URL = 'http://localhost:8001';

export interface HealthResponse {
  status: string;
  db: string;
}

export interface AnalysisStatusResponse {
  analysis_id: string;
  status: string;
  file_name: string;
  created_at: string;
  error_message: string | null;
}

export interface ReportListResponse {
  total: number;
  limit: number;
  offset: number;
  items: Record<string, unknown>[];
}

export async function getIAServiceHealth(): Promise<HealthResponse> {
  const res = await fetch(`${IA_SERVICE_URL}/health`);
  return res.json() as Promise<HealthResponse>;
}

export async function getReportAPIHealth(): Promise<HealthResponse> {
  const res = await fetch(`${REPORT_API_URL}/health`);
  return res.json() as Promise<HealthResponse>;
}

export async function analyzeDiagram(file: Buffer, fileName: string): Promise<Response> {
  const formData = new FormData();
  formData.append('file', new Blob([file]), fileName);
  return fetch(`${IA_SERVICE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });
}

export async function getAnalysisStatus(analysisId: string): Promise<Response> {
  return fetch(`${IA_SERVICE_URL}/analyses/${analysisId}/status`);
}

export async function getReports(limit = 20, offset = 0): Promise<Response> {
  return fetch(`${REPORT_API_URL}/reports?limit=${limit}&offset=${offset}`);
}

export async function getReport(analysisId: string): Promise<Response> {
  return fetch(`${REPORT_API_URL}/reports/${analysisId}`);
}
