#!/usr/bin/env bash
# Build Script for packaging Apah into a standalone Linux AppImage using appimagetool
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APPDIR="${REPO_ROOT}/build/AppDir"

echo "===================================================="
echo " Building Apah Standalone AppImage"
echo "===================================================="

if ! command -v appimagetool &> /dev/null; then
    echo "ERROR: 'appimagetool' command not found in PATH."
    echo "Please download appimagetool from https://github.com/AppImage/AppImageKit/releases"
    echo "and add it to your PATH before running this script."
    exit 1
fi

echo "[1/4] Running PyInstaller build..."
cd "${REPO_ROOT}"
pyinstaller packaging/apah.spec --clean --noconfirm

echo "[2/4] Preparing AppDir structure..."
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/256x256/apps"

cp dist/apah "${APPDIR}/usr/bin/apah"
cp packaging/linux/apah.desktop "${APPDIR}/apah.desktop"
cp apah/dashboard/frontend/assets/apah-logo.svg "${APPDIR}/apah.svg"
cp apah/dashboard/frontend/assets/apah-logo.svg "${APPDIR}/usr/share/icons/hicolor/256x256/apps/apah.svg"

echo "[3/4] Creating AppRun launcher..."
cat << 'EOF' > "${APPDIR}/AppRun"
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/apah" "$@"
EOF
chmod +x "${APPDIR}/AppRun"

echo "[4/4] Generating AppImage with appimagetool..."
ARCH=x86_64 appimagetool "${APPDIR}" "${REPO_ROOT}/dist/Apah-x86_64.AppImage"

echo "===================================================="
echo " Successfully created AppImage: dist/Apah-x86_64.AppImage"
echo "===================================================="
