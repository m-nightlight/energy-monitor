#!/usr/bin/env bash
# install.sh – Set up the energy monitor on a Raspberry Pi
#
# Run as root (or with sudo):
#   sudo bash scripts/install.sh
#
# What this does:
#   1. Installs system dependencies (ssocr, libcamera, python3-venv)
#   2. Creates a Python virtualenv and installs packages
#   3. Installs and enables the systemd service + timer

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SERVICE_USER:-pi}"
VENV_DIR="${REPO_DIR}/venv"
SYSTEMD_DIR="/etc/systemd/system"

echo "=== Energy Monitor Installer ==="
echo "Repo:        ${REPO_DIR}"
echo "Run as user: ${SERVICE_USER}"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/4] Installing system packages..."
apt-get update -qq
apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-libcamera \
    libcamera-tools \
    ssocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# ── 2. Python virtualenv ──────────────────────────────────────────────────────
echo "[2/4] Creating Python virtualenv at ${VENV_DIR}..."
python3 -m venv --system-site-packages "${VENV_DIR}"
# --system-site-packages lets the venv use the system picamera2 package
# (picamera2 installs differently and is best used from system packages)

echo "[2/4] Installing Python packages..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -r "${REPO_DIR}/requirements.txt" --quiet

# ── 3. Prepare data directory ─────────────────────────────────────────────────
echo "[3/4] Preparing data directory..."
mkdir -p "${REPO_DIR}/data" "${REPO_DIR}/debug"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${REPO_DIR}/data" "${REPO_DIR}/debug"

# ── 4. Systemd service + timer ────────────────────────────────────────────────
echo "[4/4] Installing systemd units..."

# Patch the WorkingDirectory and User in the service file to use this install
SERVICE_SRC="${REPO_DIR}/scripts/energy-monitor.service"
TIMER_SRC="${REPO_DIR}/scripts/energy-monitor.timer"

# Substitute the repo path and username
sed \
    -e "s|WorkingDirectory=.*|WorkingDirectory=${REPO_DIR}|" \
    -e "s|ExecStart=.*|ExecStart=${VENV_DIR}/bin/python -m src.pipeline|" \
    -e "s|^User=.*|User=${SERVICE_USER}|" \
    -e "s|^Group=.*|Group=${SERVICE_USER}|" \
    "${SERVICE_SRC}" > "${SYSTEMD_DIR}/energy-monitor.service"

cp "${TIMER_SRC}" "${SYSTEMD_DIR}/energy-monitor.timer"

systemctl daemon-reload
systemctl enable energy-monitor.timer
systemctl start  energy-monitor.timer

echo ""
echo "=== Installation complete ==="
echo ""
echo "The pipeline will run at the top of every hour."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status energy-monitor.timer     # Timer status"
echo "  sudo systemctl status energy-monitor.service   # Last run status"
echo "  sudo journalctl -u energy-monitor.service -f   # Follow logs"
echo "  sudo systemctl start energy-monitor.service    # Trigger a reading now"
echo ""
echo "Test the pipeline manually (no camera needed):"
echo "  ${VENV_DIR}/bin/python -m src.pipeline --image energy_reading.png --debug"
echo ""
echo "Calibrate the display region:"
echo "  ${VENV_DIR}/bin/python -m src.pipeline --image energy_reading.png --calibrate"
echo ""
echo "View stored readings:"
echo "  ${VENV_DIR}/bin/python -m src.pipeline --list"
