#!/usr/bin/env bash
set -euo pipefail

# Check ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[INFO] ffmpeg не найден. Установите его (на macOS: brew install ffmpeg)." >&2
fi

# Prefer pipx if available
if command -v pipx >/dev/null 2>&1; then
  echo "[INFO] Устанавливаю через pipx..."
  pipx install .
else
  echo "[INFO] pipx не найден. Устанавливаю через pip (в текущую среду). Рекомендуется использовать venv."
  python3 -m pip install .
fi

echo "[OK] Установка завершена. Команда 'tts-synthesize' должна быть доступна в PATH."

