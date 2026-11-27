// Screenshot docs/results.html with headless Chromium (Playwright).
// Run from the Playwright install dir so `playwright` resolves:
//   cd ~/workspace/.scratch/shot && node /home/hatch/workspace/projects/permission-rag/scripts/capture_screenshot.mjs
// Serves docs/ on :8002 itself, captures viewport + full-page shots.
import { spawn } from 'node:child_process';
import path from 'node:path';
import { createRequire } from 'node:module';

// Resolve Playwright from the existing install at ~/workspace/.scratch/shot
// (ESM imports resolve relative to this file, not the cwd).
const require = createRequire('/home/hatch/workspace/.scratch/shot/package.json');
const { chromium } = require('playwright');

const ROOT = '/home/hatch/workspace/projects/permission-rag';
const PORT = 8002;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], {
  cwd: path.join(ROOT, 'docs'), stdio: 'pipe',
});

async function waitForServer(tries = 40) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/results.html`);
      if (res.ok) return;
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error('results server did not come up');
}

try {
  await waitForServer();
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(`http://127.0.0.1:${PORT}/results.html`, { waitUntil: 'networkidle' });
  await sleep(500);
  await page.screenshot({ path: path.join(ROOT, 'docs', 'screenshots', 'demo-top.png') });
  await page.screenshot({ path: path.join(ROOT, 'docs', 'screenshots', 'demo-full.png'), fullPage: true });
  await browser.close();
  console.log('screenshots written to docs/screenshots/');
} finally {
  server.kill();
}
