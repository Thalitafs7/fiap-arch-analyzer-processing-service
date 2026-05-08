import { test, expect } from '@playwright/test';
import { join } from 'path';
import { readFileSync } from 'fs';
import { PRIMARY_BUTTON } from '../helpers/selectors';
import { analyzeDiagram } from '../helpers/api-client';

const INVALID_FILE = join(__dirname, '..', 'fixtures', 'invalid-file.txt');

test.describe('Cenarios de Erro', () => {
  test('Sem upload, botao Analisar nao aparece', async ({ page }) => {
    await page.goto('/');
    await expect(
      page.getByText('Análise de Diagramas de Arquitetura'),
    ).toBeVisible({ timeout: 30_000 });

    const analyzeBtn = page.locator(PRIMARY_BUTTON).filter({ hasText: /Analisar/i });
    await expect(analyzeBtn).not.toBeVisible({ timeout: 5_000 });
  });

  test('API rejeita arquivo invalido (.txt) com erro 400', async () => {
    const invalidFile = readFileSync(INVALID_FILE);
    const res = await analyzeDiagram(invalidFile, 'invalid-file.txt');
    expect(res.status).toBe(400);
  });

  test('API rejeita arquivo vazio com erro', async () => {
    const emptyFile = Buffer.alloc(0);
    const res = await analyzeDiagram(emptyFile, 'empty.png');
    expect([400, 422]).toContain(res.status);
  });

  test('API retorna 404 para analise inexistente', async () => {
    const fakeId = '00000000-0000-0000-0000-000000000000';
    const res = await fetch(`http://localhost:8000/analyses/${fakeId}/status`);
    expect(res.status).toBe(404);
  });
});
