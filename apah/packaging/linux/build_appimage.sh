#!/usr/bin/env bash
# Build script to generate Apah AppImage on Linux using appimagetool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Building Apah AppImage ==="

# Step 1: Run PyInstaller build
cd "$ROOT_DIR"
echo "1. Running PyInstaller..."
pyinstaller --clean apah/packaging/apah.spec

# Step 2: Prepare AppDir structure
APPDIR="$ROOT_DIR/build/Apah.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/64x64/apps"

echo "2. Populating AppDir..."
cp "$ROOT_DIR/dist/apah-app" "$APPDIR/usr/bin/apah-app"
cp "$ROOT_DIR/apah/packaging/linux/apah.desktop" "$APPDIR/usr/share/applications/apah.desktop"
cp "$ROOT_DIR/apah/packaging/linux/apah.desktop" "$APPDIR/apah.desktop"

# Create simple AppRun entrypoint script
cat << 'EOF' > "$APPDIR/AppRun"
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "$HERE/usr/bin/apah-app" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# Step 3: Check appimagetool dependency
if ! command -v appimagetool &> /dev/null; then
    echo "ERROR: 'appimagetool' is not installed or not in PATH."
    echo "Please download appimagetool from https://github.com/AppImage/AppImageKit/releases and make it executable."
    exit 1
fi

# Step 4: Run appimagetool
echo "3. Invoking appimagetool..."
ARCH=x86_64 appimagetool "$APPDIR" "$ROOT_DIR/dist/Apah-x86_64.AppImage"

echo "=== AppImage successfully created at dist/Apah-x86_64.AppImage ==="
