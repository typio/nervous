import { expect, test } from '@playwright/test';

test('index page has expected nav', async ({ page }) => {
    await page.goto('/');
    expect(await page.textContent('nav')).toBe('Demos');
});
