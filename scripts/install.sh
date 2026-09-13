#!/bin/sh
# Apah POSIX One-Line Installer for Linux and macOS
# Usage: curl -fsSL https://p19racha.github.io/Apah/install.sh | sh

set -eu

# Color formatting helpers
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

log_warn() {
  printf "%s[WARNING]%s %s\n" "${YELLOW}" "${RESET}" "$1"
}

log_error() {
  printf "%s[ERROR]%s %s\n" "${RED}" "${RESET}" "$1"
}

banner() {
  printf "%s=====================================================================%s\n" "${BOLD}" "${RESET}"
  printf "%s                 APAH INFERENCE RUNTIME INSTALLER                    %s\n" "${BOLD}" "${RESET}"
  printf "%s=====================================================================%s\n" "${BOLD}" "${RESET}"
}

banner

# 1. Detect System Architecture and OS
OS="$(uname -s 2>/dev/null || echo 'Unknown')"
ARCH="$(uname -m 2>/dev/null || echo 'Unknown')"

log_info "Detected operating system: ${OS} (${ARCH})"

case "${OS}" in
  Linux*)  IS_LINUX=1; IS_MACOS=0 ;;
  Darwin*) IS_LINUX=0; IS_MACOS=1 ;;
  *)
    log_error "Unsupported operating system: ${OS}. Apah supports Linux and macOS."
    exit 1
    ;;
esac

# 2. Check for Python 3 (3.10+)
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
fi

if [ -z "${PYTHON_CMD}" ]; then
  log_error "Python 3 is required but was not found on your system PATH."
  log_error "Please install Python 3.10 or higher (e.g. 'sudo apt install python3 python3-venv' or 'brew install python')."
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_CMD} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
PYTHON_MAJOR="$(${PYTHON_CMD} -c 'import sys; print(sys.version_info[0])')"
PYTHON_MINOR="$(${PYTHON_CMD} -c 'import sys; print(sys.version_info[1])')"

log_info "Detected Python version: ${PYTHON_VERSION} (using '${PYTHON_CMD}')"

if [ "${PYTHON_MAJOR}" -lt 3 ] || { [ "${PYTHON_MAJOR}" -eq 3 ] && [ "${PYTHON_MINOR}" -lt 10 ]; }; then
  log_error "Apah requires Python 3.10 or higher. Found Python ${PYTHON_VERSION}."
  log_error "Please upgrade Python to 3.10+ before running this installer."
  exit 1
fi

# 3. Check NVIDIA GPU Telemetry on Linux
if [ "${IS_LINUX}" -eq 1 ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo 'NVIDIA GPU')"
    log_info "Detected NVIDIA Hardware Acceleration: ${BOLD}${GPU_NAME}${RESET}"
  else
    log_warn "nvidia-smi was not found. NVIDIA GPU acceleration will not be available locally."
    log_warn "The Apah CLI can still be installed to manage remote or staging Apah instances."
  fi
fi

# 4. Determine Installation & Binary Directories
# Default to /usr/local/lib/apah and /usr/local/bin if writable or root, else fallback to ~/.local
DEFAULT_INSTALL_DIR="/usr/local/lib/apah"
DEFAULT_BIN_DIR="/usr/local/bin"

if [ "$(id -u)" -ne 0 ] && [ ! -w "$(dirname "${DEFAULT_INSTALL_DIR}")" ]; then
  DEFAULT_INSTALL_DIR="${HOME}/.local/lib/apah"
  DEFAULT_BIN_DIR="${HOME}/.local/bin"
fi

INSTALL_DIR="${APAH_INSTALL_DIR:-${DEFAULT_INSTALL_DIR}}"
BIN_DIR="${APAH_BIN_DIR:-${DEFAULT_BIN_DIR}}"
VERSION="${APAH_VERSION:-latest}"

log_info "Installation target directory: ${INSTALL_DIR}"
log_info "Binary symlink target directory: ${BIN_DIR}"

mkdir -p "${INSTALL_DIR}"
mkdir -p "${BIN_DIR}"

# 5. Create Virtual Environment and Install / Upgrade Apah
VENV_DIR="${INSTALL_DIR}/venv"

if [ ! -d "${VENV_DIR}" ]; then
  log_info "Creating isolated Virtual Environment in ${VENV_DIR}..."
  if ! ${PYTHON_CMD} -m venv "${VENV_DIR}"; then
    log_error "Failed to create python venv. Make sure 'python3-venv' or 'python3-virtualenv' package is installed."
    exit 1
  fi
fi

log_info "Upgrading pip and installing Apah package (${VERSION})..."
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip setuptools wheel

if [ "${VERSION}" = "latest" ]; then
  "${VENV_DIR}/bin/pip" install --quiet --upgrade git+https://github.com/p19racha/Apah.git
else
  "${VENV_DIR}/bin/pip" install --quiet --upgrade "git+https://github.com/p19racha/Apah.git@${VERSION}"
fi

# 6. Symlink Apah Executable to Binary Directory
APAH_EXEC="${VENV_DIR}/bin/apah"
TARGET_LINK="${BIN_DIR}/apah"

if [ -f "${APAH_EXEC}" ]; then
  rm -f "${TARGET_LINK}"
  ln -s "${APAH_EXEC}" "${TARGET_LINK}"
  log_info "Successfully created executable symlink: ${TARGET_LINK} -> ${APAH_EXEC}"
else
  log_error "Apah binary not found at ${APAH_EXEC} after installation."
  exit 1
fi

# 7. Optional Linux Systemd Service Unit Setup
if [ "${IS_LINUX}" -eq 1 ] && command -v systemctl >/dev/null 2>&1 && [ -d /etc/systemd/system ] && [ "$(id -u)" -eq 0 ]; then
  SERVICE_FILE="/etc/systemd/system/apah.service"
  log_info "Installing systemd unit at ${SERVICE_FILE}..."
  cat <<EOF > "${SERVICE_FILE}"
[Unit]
Description=Apah Custom LLM Inference Runtime (Sovereign AI Workbench)
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=${SUDO_USER:-root}
Environment="APAH_AIRGAP_MODE=1"
ExecStart=${TARGET_LINK} serve --host 127.0.0.1 --port 11500
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload >/dev/null 2>&1 || true
  log_info "Systemd service 'apah.service' created successfully."
fi

# 8. Success Output & Next Steps
printf "\n%s[SUCCESS] Apah LLM Inference Runtime installed successfully!%s\n" "${GREEN}${BOLD}" "${RESET}"
printf "%sVerified CLI Version:%s " "${BOLD}" "${RESET}"
"${TARGET_LINK}" --version || true
printf "\n"

printf "%sNext Steps:%s\n" "${BOLD}" "${RESET}"
printf "  1. Launch Apah Server:     %sapah serve%s\n" "${GREEN}" "${RESET}"
printf "  2. Pull a Model:           %sapah pull Qwen/Qwen2.5-0.5B-Instruct%s\n" "${GREEN}" "${RESET}"
printf "  3. Run Interactive Chat:   %sapah run Qwen/Qwen2.5-0.5B-Instruct%s\n" "${GREEN}" "${RESET}"
printf "\n"
