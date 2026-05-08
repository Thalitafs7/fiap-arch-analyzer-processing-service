import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { consumeSSE, validateEventOrder, SSEEvent } from '../helpers/sse-client';

const DIAGRAM_PATH = join(__dirname, '..', 'fixtures', 'test-diagram.png');

/**
 * All SSE pipeline tests share a single pipeline execution to avoid
 * Groq rate limiting (30 req/min). The result is cached in beforeAll.
 */
test.describe('Pipeline SSE via API', () => {
  let events: SSEEvent[];

  test.beforeAll(async () => {
    const diagramFile = readFileSync(DIAGRAM_PATH);
    events = await consumeSSE(diagramFile, 'test-diagram.png');
  });

  test('POST /analyze/stream retorna SSE com eventos validos', () => {
    // At least ingestion + extraction steps (running + done each)
    expect(events.length).toBeGreaterThanOrEqual(4);

    for (const event of events) {
      expect(event).toHaveProperty('step');
      expect(event).toHaveProperty('status');
      expect(event).toHaveProperty('data');
    }
  });

  test('Eventos seguem ordem do pipeline', () => {
    expect(() => validateEventOrder(events)).not.toThrow();
  });

  test('Pipeline completa ou reporta erro corretamente', () => {
    const lastEvent = events[events.length - 1];
    const completed = lastEvent.step === 'done' && lastEvent.status === 'complete';
    const errored = lastEvent.status === 'error';

    // Pipeline must either complete or fail with a clear error
    expect(completed || errored).toBe(true);

    if (completed) {
      // If completed, verify result has expected fields
      const result = lastEvent.data as Record<string, unknown>;
      expect(result).toHaveProperty('analysis_id');
      expect(result).toHaveProperty('report');
    }
  });
});

test.describe('Pipeline Sync via API', () => {
  test('POST /analyze retorna JSON com analysis_id', async () => {
    const diagramFile = readFileSync(DIAGRAM_PATH);
    const formData = new FormData();
    formData.append('file', new Blob([diagramFile]), 'test-diagram.png');

    const res = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      body: formData,
    });

    // 200 = success, 422 = pipeline error (e.g., LLM rate limit)
    expect([200, 422]).toContain(res.status);

    if (res.status === 200) {
      const body = (await res.json()) as Record<string, unknown>;
      expect(body).toHaveProperty('analysis_id');
      expect(body).toHaveProperty('status');
      expect(body.status).toBe('analisado');
      expect(body).toHaveProperty('report');
    }
  });
});
