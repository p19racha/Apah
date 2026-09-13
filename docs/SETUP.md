# GitHub Pages Setup & Verification Guide

This document describes how to configure GitHub Pages hosting for the Apah one-line installer scripts and landing page.

---

## 1. Enabling GitHub Pages in Repository Settings

1. Open your repository on GitHub: [`https://github.com/p19racha/Apah`](https://github.com/p19racha/Apah).
2. Click on the **Settings** tab.
3. In the left sidebar navigation, click on **Pages** (under Code and automation).
4. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`.
   - **Branch**: Select `main` and choose `/docs` as the folder directory.
5. Click **Save**.

GitHub Pages will build and deploy the site at `https://p19racha.github.io/Apah/`.

---

## 2. Verifying Plain Script Serving (Security & Content Type)

Before testing the piped execution (`curl ... | sh`), verify that GitHub Pages serves `install.sh` as raw plain text:

```bash
# Verify HTTP 200 status and raw script contents
curl -fsSL https://p19racha.github.io/Apah/install.sh | head -n 10
```

Expected output should display the script header:
```sh
#!/bin/sh
# Apah POSIX One-Line Installer for Linux and macOS
```

If it returns HTML or a 404 error page, do NOT pipe to `sh`. Re-check that the `/docs` folder has been published on `main`.

---

## 3. One-Line Install Verification Commands

- **Linux & macOS**:
  ```bash
  curl -fsSL https://p19racha.github.io/Apah/install.sh | sh
  ```

- **Windows PowerShell**:
  ```powershell
  irm https://p19racha.github.io/Apah/install.ps1 | iex
  ```
