/**
 * Helper para consumir Server-Sent Events do /analyze/stream.
 */

const IA_SERVICE_URL = 'http://localhost:8000';

export interface SSEEvent {
  step: string;
  status: string;
  data: Record<string, unknown>;
}

/**
 * Envia arquivo para /analyze/stream e coleta todos os eventos SSE.
 * Retorna array de eventos na ordem recebida.
 */
export async function consumeSSE(
  file: Buffer,
  fileName: string,
  timeoutMs = 180_000,
): Promise<SSEEvent[]> {
  const formData = new FormData();
  formData.append('file', new Blob([file]), fileName);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${IA_SERVICE_URL}/analyze/stream`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }

    const events: SSEEvent[] = [];
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr) as SSEEvent;
          events.push(event);
        } catch {
          // Ignore malformed SSE lines
        }
      }
    }

    return events;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Valida que os eventos SSE seguem a ordem esperada do pipeline.
 * Se o pipeline errored, valida apenas os passos até o erro.
 * Se completou, exige todos os passos.
 */
export function validateEventOrder(events: SSEEvent[]): void {
  const hasError = events.some((e) => e.status === 'error');
  const hasComplete = events.some((e) => e.step === 'done' && e.status === 'complete');

  const allExpected = ['ingestion', 'extraction', 'rag', 'report', 'qa', 'done'];
  const stepOrder = events
    .map((e) => e.step)
    .filter((s, i, arr) => arr.indexOf(s) === i); // unique, preserving order

  // If pipeline completed, all steps must be present
  // If pipeline errored, only validate steps that appeared are in order
  const expectedSteps = hasComplete
    ? allExpected
    : allExpected.filter((s) => stepOrder.includes(s));

  for (let i = 0; i < expectedSteps.length; i++) {
    const idx = stepOrder.indexOf(expectedSteps[i]);
    if (idx === -1) {
      throw new Error(`Step "${expectedSteps[i]}" not found in events`);
    }
    if (i > 0) {
      const prevIdx = stepOrder.indexOf(expectedSteps[i - 1]);
      if (idx <= prevIdx) {
        throw new Error(
          `Step "${expectedSteps[i]}" appeared before "${expectedSteps[i - 1]}"`,
        );
      }
    }
  }

  // At minimum, ingestion and extraction should always appear
  if (!stepOrder.includes('ingestion')) {
    throw new Error('Step "ingestion" not found in events');
  }
  if (!stepOrder.includes('extraction')) {
    throw new Error('Step "extraction" not found in events');
  }
}
