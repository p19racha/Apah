#!/bin/sh
# Apah Offline / Air-Gapped Local Bundle Installer
# Usage: ./scripts/install-offline.sh /path/to/apah-offline-bundle

set -eu

if [ -t 1 ]; then
  BOLD="$(tput bold 2>/dev/null || echo '')"
  GREEN="$(tput setaf 2 2>/dev/null || echo '')"
  YELLOW="$(tput setaf 3 2>/dev/null || echo '')"
  RED="$(tput setaf 1 2>/dev/null || echo '')"
  RESET="$(tput sgr0 2>/dev/null || echo '')"
else
  BOLD=""
  GREEN=""
  YELLOW=""
  RED=""
  RESET=""
fi

log_info() {
  printf "%s[INFO]%s %s\n" "${GREEN}" "${RESET}" "$1"
}

log_error() {
  printf "%s[ERROR]%s %s\n" "${RED}" "${RESET}" "$1"
}

printf "%s=====================================================================%s\n" "${BOLD}" "${RESET}"
printf "%s             APAH AIR-GAPPED / OFFLINE BUNDLE INSTALLER             %s\n" "${BOLD}" "${RESET}"
printf "%s=====================================================================%s\n" "${BOLD}" "${RESET}"

if [ $# -lt 1 ]; then
  log_error "Missing required bundle directory argument."
  printf "Usage: %s /path/to/apah-offline-bundle-directory\n" "$0"
  printf "The bundle directory must contain pre-downloaded wheel files (.whl).\n"
  exit 1
fi

BUNDLE_DIR="$1"

if [ ! -d "${BUNDLE_DIR}" ]; then
  log_error "Bundle directory '${BUNDLE_DIR}' does not exist or is not a directory."
  exit 1
fi

# Check for presence of wheel files
WHEEL_COUNT="$(find "${BUNDLE_DIR}" -maxdepth 2 -name "*.whl" 2>/dev/null | wc -l || echo 0)"
if [ "${WHEEL_COUNT}" -eq 0 ]; then
  log_error "No Python wheel (.whl) files found in bundle directory '${BUNDLE_DIR}'."
  log_error "Ensure you downloaded dependencies on a connected machine via 'pip download apah -d ${BUNDLE_DIR}'."
  exit 1
fi

log_info "Found ${WHEEL_COUNT} offline wheel package(s) in '${BUNDLE_DIR}'."

# Check for python3
if ! command -v python3 >/dev/null 2>&1; then
  log_error "Python 3 is required but not found on PATH."
  exit 1
fi

INSTALL_DIR="${APAH_INSTALL_DIR:-/usr/local/lib/apah}"
BIN_DIR="${APAH_BIN_DIR:-/usr/local/bin}"

if [ "$(id -u)" -ne 0 ] && [ ! -w "$(dirname "${INSTALL_DIR}")" ]; then
  INSTALL_DIR="${HOME}/.local/lib/apah"
  BIN_DIR="${HOME}/.local/bin"
fi

mkdir -p "${INSTALL_DIR}"
mkdir -p "${BIN_DIR}"

VENV_DIR="${INSTALL_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
  log_info "Creating isolated virtual environment in ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
fi

log_info "Installing Apah offline without internet connectivity (--no-index)..."
"${VENV_DIR}/bin/pip" install --no-index --find-links="${BUNDLE_DIR}" apah

APAH_EXEC="${VENV_DIR}/bin/apah"
TARGET_LINK="${BIN_DIR}/apah"

if [ -f "${APAH_EXEC}" ]; then
  rm -f "${TARGET_LINK}"
  ln -s "${APAH_EXEC}" "${TARGET_LINK}"
  log_info "Successfully created symlink: ${TARGET_LINK} -> ${APAH_EXEC}"
else
  log_error "Apah binary not found after offline installation."
  exit 1
fi

printf "\n%s[SUCCESS] Apah installed in 100%% air-gapped mode!%s\n" "${GREEN}${BOLD}" "${RESET}"
"${TARGET_LINK}" --version || true
