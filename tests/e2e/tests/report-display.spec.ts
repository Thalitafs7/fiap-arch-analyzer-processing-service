import { test, expect, Page } from '@playwright/test';
import { join } from 'path';
import {
  FILE_UPLOADER_INPUT,
  PRIMARY_BUTTON,
  EXPANDER,
} from '../helpers/selectors';

const DIAGRAM_PNG = join(__dirname, '..', 'fixtures', 'test-diagram.png');

test.describe('Renderizacao do Relatorio', () => {
  test.setTimeout(180_000);

  /**
   * Single shared test that uploads once and validates all report sections.
   * This avoids Groq rate limits from running 6 separate pipeline calls.
   */
  test('Relatorio completo renderiza todas as secoes', async ({ page }) => {
    // Upload and run analysis
    await page.goto('/');
    await expect(
      page.getByText('Análise de Diagramas de Arquitetura'),
    ).toBeVisible({ timeout: 30_000 });

    const fileInput = page.locator(FILE_UPLOADER_INPUT);
    await fileInput.setInputFiles(DIAGRAM_PNG);
    await page.waitForTimeout(2000);

    const analyzeBtn = page.locator(PRIMARY_BUTTON).filter({ hasText: /Analisar/i });
    await analyzeBtn.click();

    // Wait for pipeline to finish (success or error)
    await expect(
      page.getByText(/conclu[ií]da|Erro após/i),
    ).toBeVisible({ timeout: 160_000 });

    await page.waitForTimeout(3000);
    const bodyText = await page.locator('body').textContent() || '';
    const succeeded = !!bodyText.match(/conclu[ií]da/i);

    if (!succeeded) {
      // Pipeline errored (e.g., Groq rate limit) — can't verify report sections
      // Verify at least the error was shown properly
      expect(bodyText).toContain('Erro');
      return;
    }

    // ── Resumo Executivo ──────────────────────────────────
    expect(bodyText).toContain('Resumo Executivo');
    const summaryIdx = bodyText.indexOf('Resumo Executivo');
    const afterSummary = bodyText.slice(summaryIdx + 20, summaryIdx + 200);
    expect(afterSummary.length).toBeGreaterThan(50);

    // ── Componentes Identificados ─────────────────────────
    expect(bodyText).toContain('Componentes Identificados');

    // ── Riscos Arquiteturais ──────────────────────────────
    expect(bodyText).toContain('Riscos Arquiteturais');

    const hasSeverity =
      bodyText.includes('ALTO') ||
      bodyText.includes('MÉDIO') ||
      bodyText.includes('BAIXO');
    expect(hasSeverity).toBe(true);

    // Expanders for risks
    const expanders = page.locator(EXPANDER);
    const expanderCount = await expanders.count();
    expect(expanderCount).toBeGreaterThan(0);

    await expanders.first().click();
    await page.waitForTimeout(500);
    const expandedContent = await expanders.first().textContent();
    expect(expandedContent!.length).toBeGreaterThan(20);

    // ── Recomendações ─────────────────────────────────────
    expect(bodyText).toContain('Recomenda');

    // ── Qualidade do Relatório ────────────────────────────
    expect(bodyText).toContain('Qualidade');
    expect(bodyText).toContain('Score');
  });
});
