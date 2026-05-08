import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join } from 'path';
import {
  analyzeDiagram,
  getAnalysisStatus,
  getReports,
  getReport,
  ReportListResponse,
} from '../helpers/api-client';

const DIAGRAM_PATH = join(__dirname, '..', 'fixtures', 'test-diagram.png');

test.describe('Report API', () => {
  test('GET /reports retorna lista com paginacao', async () => {
    const res = await getReports(10, 0);
    expect(res.status).toBe(200);

    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toHaveProperty('total');
    expect(body).toHaveProperty('limit');
    expect(body).toHaveProperty('offset');
    expect(body).toHaveProperty('items');
    expect(Array.isArray(body.items)).toBe(true);
  });

  test('GET /reports/{id} retorna relatorio completo', async () => {
    // Get first report from the list (from prior test runs)
    const listRes = await getReports(1, 0);
    const listBody = (await listRes.json()) as ReportListResponse;

    if (listBody.total === 0) {
      test.skip(true, 'No reports in database');
      return;
    }

    const analysisId = (listBody.items[0] as Record<string, unknown>).analysis_id as string;
    const res = await getReport(analysisId);
    expect(res.status).toBe(200);

    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toHaveProperty('analysis_id');
    expect(body).toHaveProperty('report');
  });

  test('GET /reports/{id} retorna 404 para ID inexistente', async () => {
    const fakeId = '00000000-0000-0000-0000-000000000000';
    const res = await getReport(fakeId);
    expect(res.status).toBe(404);
  });

  test('Paginacao retorna resultados consistentes', async () => {
    const res1 = await getReports(1, 0);
    const res2 = await getReports(1, 1);

    const body1 = (await res1.json()) as ReportListResponse;
    const body2 = (await res2.json()) as ReportListResponse;

    if (body1.total > 1) {
      expect(body1.items[0]).not.toEqual(body2.items[0]);
    }
  });

  test('GET /analyses/{id}/status retorna status', async () => {
    // Get an analysis ID from the reports list
    const listRes = await getReports(1, 0);
    const listBody = (await listRes.json()) as ReportListResponse;

    if (listBody.total === 0) {
      test.skip(true, 'No reports in database');
      return;
    }

    const analysisId = (listBody.items[0] as Record<string, unknown>).analysis_id as string;
    const res = await getAnalysisStatus(analysisId);
    expect(res.status).toBe(200);

    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toHaveProperty('analysis_id');
    expect(body).toHaveProperty('status');
    expect(['recebido', 'em_processamento', 'analisado', 'erro']).toContain(body.status);
  });
});
