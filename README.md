# Yandex & Google Speech Synthesis • CLI и Web

Инструмент синтеза речи с двумя интерфейсами:
- CLI‑утилита `voice` (алиас `tts-synthesize`) для локального использования
- Веб‑сервис (Flask) для ввода текста/файла и получения MP3

Проект соответствует принципам из `CLI_DESIGN_PRINCIPLES.md` и операционной модели из `OPERATIONAL_MODEL_PROPOSAL.md`.

## Установка CLI

Рекомендуется `pipx` (изолированная установка). При его отсутствии можно использовать `pip`.

```bash
# В корне репозитория
pipx install .   # или: python3 -m pip install .

# проверка
voice --help
```

Если видите предупреждение о PATH, добавьте в `~/.zshrc` строку (путь может отличаться):
```bash
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH"
```

## Креденшлы

### Google Cloud Text‑to‑Speech
Поддерживаются ADC и ключ JSON сервсчёта:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-key.json"
```

### Yandex SpeechKit
Рекомендуемый способ — переменные окружения:
```bash
export YANDEX_API_KEY="<секрет>"
export YANDEX_FOLDER_ID="<folder-id>"
```

## Быстрый старт (CLI)

- Минимум (Google по умолчанию), файл → MP3, автоплей в VLC, прогресс сохраняется:
```bash
voice --input-file "/path/to/file.md" --play
```

- Сохранить на Desktop и автоплей:
```bash
voice -D --play "/path/to/file.md"
```

- Яндекс (скорость через env), Desktop, автоплей:
```bash
YANDEX_SPEED=0.9 voice --provider yandex --voice ermil -D --play "/path/to/file.md"
```

Поведение по умолчанию:
- если `--output-file` не задан, имя берётся из входного файла (`file.mp3`)
- без `-D` сохраняем в текущую директорию; с `-D` — на Desktop
- позиционный одиночный аргумент, указывающий на существующий файл, распознаётся как `--input-file`

Полный список опций смотрите в `voice --help` и `man voice`.

## Автовоспроизведение (VLC)
На macOS используется AppleScript для активации окна VLC. При желании можно создать системные алиасы для `vlc`/`VLC`:
```bash
P=$([ -d /opt/homebrew/bin ] && echo /opt/homebrew/bin || echo /usr/local/bin); \
[ -w "$P" ] && SUDO= || SUDO=sudo; \
$SUDO ln -sf "/Applications/VLC.app/Contents/MacOS/VLC" "$P/vlc"; \
$SUDO ln -sf "/Applications/VLC.app/Contents/MacOS/VLC" "$P/VLC"
```

## Веб‑сервис (локально)

```bash
python3 -m pip install -r requirements.txt
PORT=8080 python3 app.py
# затем откройте http://localhost:8080
```

Эндпоинты: `/` (форма), `/synthesize` (POST, выдаёт `{success, file_url}`), `/healthz`.

## Сообщения об ошибках
- Google 5000 bytes: автоматически избегаем за счёт SSML‑сплита по байтам
- Yandex UNAVAILABLE: сетевой сбой; повторите запрос (ретраи), проверьте VPN/фаервол
- Отладка: используйте `-v`/`-vv`/`-vvv` и смотрите stderr

## Документация
- Встроенная справка: `voice --help`
- Man‑страницы: `man voice`, `man tts-synthesize`
- Принципы CLI: `CLI_DESIGN_PRINCIPLES.md`
- Операционная модель: `OPERATIONAL_MODEL_PROPOSAL.md`
