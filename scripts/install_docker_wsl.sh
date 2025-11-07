#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: Required command '$1' not found. Please install it first."
    exit 1
  fi
}

if [[ $EUID -ne 0 ]]; then
  log "This script must be run with sudo. Example:"
  log "  sudo bash $0"
  exit 1
fi

require_cmd curl
require_cmd gpg

DISTRO="bookworm"
ARCH="$(dpkg --print-architecture)"
KEYRING_DIR="/etc/apt/keyrings"
DOCKER_GPG="${KEYRING_DIR}/docker.gpg"
DOCKER_LIST="/etc/apt/sources.list.d/docker.list"

log "Updating apt package index..."
apt-get update

log "Installing prerequisites..."
apt-get install -y --no-install-recommends ca-certificates curl gnupg

log "Ensuring keyring directory exists at ${KEYRING_DIR}..."
install -m 0755 -d "${KEYRING_DIR}"

log "Fetching Docker GPG key..."
curl -fsSL "https://download.docker.com/linux/debian/gpg" | gpg --dearmor -o "${DOCKER_GPG}"
chmod a+r "${DOCKER_GPG}"

log "Configuring Docker apt repository..."
cat > "${DOCKER_LIST}" <<EOF
deb [arch=${ARCH} signed-by=${DOCKER_GPG}] https://download.docker.com/linux/debian ${DISTRO} stable
EOF

log "Updating apt package index (Docker repo added)..."
apt-get update

log "Installing Docker Engine and plugins..."
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

log "Enabling Docker to start under systemd (if available)..."
if command -v systemctl >/dev/null 2>&1; then
  systemctl enable docker || log "systemctl enable docker failed (this is expected on some WSL setups)."
fi

log "Docker installation complete."
log "IMPORTANT: Add your user to the docker group and restart your shell:"
log "  sudo usermod -aG docker ${SUDO_USER:-$USER}"
log "  newgrp docker  # or log out/in"

log "Start the Docker service before using it (on WSL sessions):"
log "  sudo service docker start"

log "Test your installation once the service is running:"
log "  docker run hello-world"
