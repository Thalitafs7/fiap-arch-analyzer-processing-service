import { test, expect } from '@playwright/test';
import { getIAServiceHealth, getReportAPIHealth } from '../helpers/api-client';
import { SIDEBAR } from '../helpers/selectors';

test.describe('Health Checks', () => {
  test('ia-service /health retorna healthy', async () => {
    const health = await getIAServiceHealth();
    expect(health.status).toBe('healthy');
    expect(health.db).toBe('connected');
  });

  test('report-api /health retorna healthy', async () => {
    const health = await getReportAPIHealth();
    expect(health.status).toBe('healthy');
  });

  test('Streamlit carrega sem erro', async ({ page }) => {
    await page.goto('/');
    await expect(
      page.getByText('Análise de Diagramas de Arquitetura'),
    ).toBeVisible({ timeout: 30_000 });
  });

  test('Sidebar mostra status online', async ({ page }) => {
    await page.goto('/');
    const sidebar = page.locator(SIDEBAR);
    await expect(sidebar.getByText('online', { exact: false })).toBeVisible({
      timeout: 15_000,
    });
  });
});
