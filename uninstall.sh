#!/usr/bin/env bash
set -euo pipefail

PKG="yandex-speech-cli"

if command -v pipx >/dev/null 2>&1; then
  echo "[INFO] Удаляю через pipx..."
  pipx uninstall "$PKG" || true
fi

echo "[INFO] Удаляю через pip..."
python3 -m pip uninstall -y "$PKG" || true

echo "[OK] Удаление завершено."

