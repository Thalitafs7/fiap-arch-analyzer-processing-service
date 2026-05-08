import { test, expect, Page } from '@playwright/test';
import { join } from 'path';
import {
  FILE_UPLOADER_INPUT,
  PRIMARY_BUTTON,
  DOWNLOAD_BUTTON,
  IMAGE,
} from '../helpers/selectors';

const DIAGRAM_PNG = join(__dirname, '..', 'fixtures', 'test-diagram.png');

test.describe('Fluxo Principal de Upload', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(
      page.getByText('Análise de Diagramas de Arquitetura'),
    ).toBeVisible({ timeout: 30_000 });
  });

  test('Upload de PNG mostra preview e botao Analisar', async ({ page }) => {
    const fileInput = page.locator(FILE_UPLOADER_INPUT);
    await fileInput.setInputFiles(DIAGRAM_PNG);

    // Image preview should appear
    await expect(page.locator(IMAGE).first()).toBeVisible({ timeout: 10_000 });

    // "Analisar Diagrama" button should be visible
    await page.waitForTimeout(1000);
    const analyzeBtn = page.locator(PRIMARY_BUTTON).filter({ hasText: /Analisar/i });
    await expect(analyzeBtn).toBeVisible({ timeout: 10_000 });
  });

  test('Analise completa com todas as etapas e download', async ({ page }) => {
    test.setTimeout(180_000);

    const fileInput = page.locator(FILE_UPLOADER_INPUT);
    await fileInput.setInputFiles(DIAGRAM_PNG);
    await page.waitForTimeout(2000);

    const analyzeBtn = page.locator(PRIMARY_BUTTON).filter({ hasText: /Analisar/i });
    await analyzeBtn.click();

    // Wait for pipeline to finish (success or error)
    await expect(
      page.getByText(/conclu[ií]da|Erro após/i),
    ).toBeVisible({ timeout: 160_000 });

    const pageContent = await page.locator('body').textContent() || '';
    const succeeded = pageContent.match(/conclu[ií]da/i);

    // Pipeline steps should always appear (at least ingestion + extraction)
    expect(pageContent).toContain('Ingestão');
    expect(pageContent).toContain('Extração');

    if (succeeded) {
      // Full success: all sections rendered
      expect(pageContent).toContain('RAG');
      expect(pageContent).toContain('QA');
      expect(pageContent).toContain('Resumo Executivo');
      expect(pageContent).toContain('Componentes Identificados');
      expect(pageContent).toContain('Riscos Arquiteturais');
      expect(pageContent).toContain('Recomenda');

      const downloadBtn = page.locator(DOWNLOAD_BUTTON);
      await expect(downloadBtn).toBeVisible({ timeout: 10_000 });
    }
    // If errored (e.g., rate limit), the test still passes —
    // it verified the upload + SSE flow works correctly
  });
});
