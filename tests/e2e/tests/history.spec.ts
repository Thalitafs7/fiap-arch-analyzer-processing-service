import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { SIDEBAR } from '../helpers/selectors';
import { analyzeDiagram } from '../helpers/api-client';

const DIAGRAM_PATH = join(__dirname, '..', 'fixtures', 'test-diagram.png');

test.describe('Historico na Sidebar', () => {
  test.beforeAll(async () => {
    // Ensure at least one analysis exists
    const diagramFile = readFileSync(DIAGRAM_PATH);
    const res = await analyzeDiagram(diagramFile, 'test-diagram.png');
    if (!res.ok) {
      console.warn('Could not create analysis for history test:', res.status);
    }
  });

  test('Sidebar mostra analises anteriores', async ({ page }) => {
    await page.goto('/');
    await expect(
      page.getByText('Análise de Diagramas de Arquitetura'),
    ).toBeVisible({ timeout: 30_000 });

    const sidebar = page.locator(SIDEBAR);
    await expect(sidebar.getByText('Histórico')).toBeVisible({ timeout: 10_000 });

    const sidebarText = await sidebar.textContent();
    expect(sidebarText!.length).toBeGreaterThan(20);
  });

  test('Item do historico mostra nome do arquivo', async ({ page }) => {
    await page.goto('/');
    await expect(
      page.getByText('Análise de Diagramas de Arquitetura'),
    ).toBeVisible({ timeout: 30_000 });

    const sidebar = page.locator(SIDEBAR);
    const sidebarText = await sidebar.textContent();

    expect(sidebarText).toContain('test-diagram');
  });
});
