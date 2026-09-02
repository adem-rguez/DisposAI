// download-binaries.mjs
//
// Downloads and stages the native inference binaries (llama-server.exe,
// sd.exe) and the standalone `uv` Python tool that DisposAI needs at
// runtime but does not bundle in the installer, in order to keep the
// installer small. Binaries are fetched from their upstream GitHub
// Releases at first-run time instead of being packaged.
//
// This module is designed to run in the Electron MAIN process (it needs
// full fs/https/child_process-level access) and is intended to be called
// either directly from electron-main.js or from a first-run setup wizard
// renderer via IPC (wizard not built yet — this module is standalone and
// testable on its own).

import fs from 'node:fs';
import https from 'node:https';
import path from 'node:path';
import os from 'node:os';
import extractZip from 'extract-zip';

const USER_AGENT = 'disposai-download-binaries';

/**
 * Fetch and JSON.parse a URL via Node's built-in https module.
 * GitHub's API requires a User-Agent header or it will reject the request.
 */
function fetchJson(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': USER_AGENT, Accept: 'application/vnd.github+json' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        resolve(fetchJson(res.headers.location));
        return;
      }
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`GET ${url} failed with status ${res.statusCode}`));
        return;
      }
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (err) {
          reject(new Error(`Failed to parse JSON from ${url}: ${err.message}`));
        }
      });
    }).on('error', reject);
  });
}

/**
 * Download a URL to a local file, following up to `maxRedirects` redirect
 * hops (GitHub asset downloads 302-redirect to S3). Reports byte progress
 * via onProgress({ bytesDownloaded, totalBytes }).
 */
function downloadFile(url, destPath, onProgress, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    const request = (currentUrl, redirectsLeft) => {
      https.get(currentUrl, { headers: { 'User-Agent': USER_AGENT } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          if (redirectsLeft <= 0) {
            reject(new Error(`Too many redirects downloading ${url}`));
            return;
          }
          request(res.headers.location, redirectsLeft - 1);
          return;
        }
        if (res.statusCode !== 200) {
          res.resume();
          reject(new Error(`GET ${currentUrl} failed with status ${res.statusCode}`));
          return;
        }

        const totalBytes = Number(res.headers['content-length']) || 0;
        let bytesDownloaded = 0;
        const fileStream = fs.createWriteStream(destPath);

        res.on('data', (chunk) => {
          bytesDownloaded += chunk.length;
          if (onProgress) onProgress({ bytesDownloaded, totalBytes });
        });

        res.pipe(fileStream);

        fileStream.on('finish', () => {
          fileStream.close((err) => {
            if (err) reject(err);
            else resolve();
          });
        });

        fileStream.on('error', reject);
        res.on('error', reject);
      }).on('error', reject);
    };

    request(url, maxRedirects);
  });
}

/**
 * Extract a zip to a temp scratch directory, recursively search for a file
 * named `exeName` (case-insensitive), copy it to destDir/exeName, then
 * clean up the scratch directory and the zip file itself.
 */
async function extractAndFindExe(zipPath, exeName, destDir) {
  const scratchDir = fs.mkdtempSync(path.join(os.tmpdir(), 'disposai-extract-'));
  try {
    await extractZip(zipPath, { dir: scratchDir });

    const found = findFileRecursive(scratchDir, exeName);
    if (!found) {
      throw new Error(`Could not find ${exeName} inside extracted archive ${zipPath}`);
    }

    const destPath = path.join(destDir, exeName);
    fs.copyFileSync(found, destPath);
    return destPath;
  } finally {
    fs.rmSync(scratchDir, { recursive: true, force: true });
    fs.rmSync(zipPath, { force: true });
  }
}

/** Recursively search `dir` for a file named `targetName` (case-insensitive). */
function findFileRecursive(dir, targetName) {
  const lowerTarget = targetName.toLowerCase();
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findFileRecursive(fullPath, targetName);
      if (found) return found;
    } else if (entry.name.toLowerCase() === lowerTarget) {
      return fullPath;
    }
  }

  return null;
}

/** Pick the best matching Windows CUDA asset from a release's assets array. */
function pickWindowsCudaAsset(assets, { requireCuda = true } = {}) {
  const cudaAsset = assets.find((a) => {
    const name = a.name.toLowerCase();
    return name.includes('win') && name.includes('cuda') && name.endsWith('.zip');
  });
  if (cudaAsset) return cudaAsset;

  if (!requireCuda) return null;

  const cpuAsset = assets.find((a) => {
    const name = a.name.toLowerCase();
    return name.includes('win') && name.includes('cpu') && name.endsWith('.zip');
  });
  return cpuAsset || null;
}

/**
 * Downloads a prebuilt Windows CUDA build of llama-server.exe from the
 * latest llama.cpp GitHub release and stages it at destDir/llama-server.exe.
 */
export async function downloadLlamaServer(destDir, onProgress) {
  const phase = 'llama-server';
  try {
    fs.mkdirSync(destDir, { recursive: true });

    const release = await fetchJson('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest');
    const asset = pickWindowsCudaAsset(release.assets || []);
    if (!asset) {
      throw new Error('Could not find a Windows CUDA (or CPU fallback) build asset in the latest llama.cpp release');
    }

    const zipPath = path.join(os.tmpdir(), `disposai-llama-server-${Date.now()}.zip`);
    await downloadFile(asset.browser_download_url, zipPath, ({ bytesDownloaded, totalBytes }) => {
      if (onProgress) onProgress({ phase, status: 'running', bytesDownloaded, totalBytes });
    });

    await extractAndFindExe(zipPath, 'llama-server.exe', destDir);

    if (onProgress) onProgress({ phase, status: 'done' });
  } catch (err) {
    if (onProgress) onProgress({ phase, status: 'error', message: err.message });
    throw err;
  }
}

/**
 * Downloads a prebuilt Windows CUDA build of sd.exe from the latest
 * stable-diffusion.cpp GitHub release and stages it at destDir/sd.exe.
 */
export async function downloadSdBinary(destDir, onProgress) {
  const phase = 'sd';
  try {
    fs.mkdirSync(destDir, { recursive: true });

    const release = await fetchJson('https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest');
    const asset = pickWindowsCudaAsset(release.assets || []);
    if (!asset) {
      throw new Error('Could not find a Windows CUDA (or CPU fallback) build asset in the latest stable-diffusion.cpp release');
    }

    const zipPath = path.join(os.tmpdir(), `disposai-sd-${Date.now()}.zip`);
    await downloadFile(asset.browser_download_url, zipPath, ({ bytesDownloaded, totalBytes }) => {
      if (onProgress) onProgress({ phase, status: 'running', bytesDownloaded, totalBytes });
    });

    await extractAndFindExe(zipPath, 'sd.exe', destDir);

    if (onProgress) onProgress({ phase, status: 'done' });
  } catch (err) {
    if (onProgress) onProgress({ phase, status: 'error', message: err.message });
    throw err;
  }
}

/**
 * Ensures uv.exe is present at destDir/uv.exe, downloading the standalone
 * Windows x64 build from the latest astral-sh/uv GitHub release if needed.
 * Does not use the piped PowerShell installer script, to avoid trusting
 * and executing a remote shell script during install.
 */
export async function ensureUv(destDir, onProgress) {
  const phase = 'uv';
  const uvPath = path.join(destDir, 'uv.exe');

  try {
    fs.mkdirSync(destDir, { recursive: true });

    if (fs.existsSync(uvPath)) {
      if (onProgress) onProgress({ phase, status: 'done', skipped: true });
      return uvPath;
    }

    const release = await fetchJson('https://api.github.com/repos/astral-sh/uv/releases/latest');
    const asset = (release.assets || []).find((a) => {
      const name = a.name.toLowerCase();
      return name.includes('x86_64-pc-windows-msvc') && name.endsWith('.zip');
    });
    if (!asset) {
      throw new Error('Could not find a Windows x64 uv build asset in the latest uv release');
    }

    const zipPath = path.join(os.tmpdir(), `disposai-uv-${Date.now()}.zip`);
    await downloadFile(asset.browser_download_url, zipPath, ({ bytesDownloaded, totalBytes }) => {
      if (onProgress) onProgress({ phase, status: 'running', bytesDownloaded, totalBytes });
    });

    const resultPath = await extractAndFindExe(zipPath, 'uv.exe', destDir);

    if (onProgress) onProgress({ phase, status: 'done' });
    return resultPath;
  } catch (err) {
    if (onProgress) onProgress({ phase, status: 'error', message: err.message });
    throw err;
  }
}
