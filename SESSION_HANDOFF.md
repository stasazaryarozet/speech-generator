# Handoff: текущая сессия и подробный план действий

Дата/время: 2025-08-10 (локальная сессия Cursor)

## Краткое резюме состояния

- Реализованы ключевые улучшения TTS для CLI и Web (частично):
  - Парсинг Markdown с удалением ссылок/картинок/инлайн-кода/маркеров списков, замена `[]` на `()`, паузы после заголовков/цитат/пустых строк.
  - Защищённые паузы: `[[PAUSExxx]]` → Google `<break time="xxxms"/>`, Yandex `sil<[xxx]>`.
  - Римские цифры: Google — обёртка в `<say-as interpret-as="cardinal">N</say-as>`, Yandex — конвертация в арабские числа.
  - «Естественные» границы чанков для обоих провайдеров (разрез по пунктуации/пробелам рядом с серединой).
  - Скорости по умолчанию: Google 0.80, Yandex 0.80.
  - Автовыбор провайдера: по умолчанию `auto` (сначала Yandex, при ошибках/недоступности — Google). Логи сообщают о провайдере/фоллбеке.
  - Встроены дефолтные креды Yandex (folder_id + api_key) как резерв, если нет ENV/файлов (маскируем в логах).
  - Чередование полов по абзацам:
    - Yandex: при отсутствии `--voice` абзацы читаются попеременно мужским/женским голосом. Добавлены списки предпочтений голосов по полу с ENV-переопределениями.
    - Google: ч/б логика добавлена, но нуждается в чистке отступов (см. TODO) и валидации.

- Выявленная проблема провайдера:
  - Yandex SpeechKit v3 «прикрепляет» голос к одному gRPC‑соединению, из-за чего попытки смены голоса в рамках одного долгоживущего соединения игнорируются. Решение: создавать отдельное соединение на каждый абзац при смене пола. Логика внедрена (см. `synthesize_yandex`), но есть незавершённые правки/отступы в файле (см. TODO) и требуется прогон.

- Web-приложение: часть улучшений уже в `app.py` (очистка MD, римские, паузы, скорость, Yandex hints без role), но чередование полов для Google и пер-абзацное подключение для Yandex пока не зеркалированы и не задеплоены. Последний Cloud Build: SUCCESS (до части последних правок).

## Файлы и ключевые изменения

- `synthesize_cli.py`:
  - Функции:
    - `insert_md_pauses`: теперь удаляет `![alt](url)`, `[txt](url)`, `(http...)`, очищает `*_```, маркеры списков, добавляет плейсхолдеры пауз.
    - `replace_roman_numerals_for_google` / `replace_roman_numerals_for_yandex`: строгие паттерны, регистронезависимы.
    - `split_text_google_ssml_safe` и `split_text_yandex_safe`: разрез по «естественным» границам.
    - `get_yandex_gender_voice_preferences`/`get_google_gender_voice_preferences`: списки предпочтений по полу с ENV‑override.
    - `synthesize_google`: подготовка текста, (план) чередование полов по абзацам с выбором из предпочтений (требуется доп. фикс отступов — см. TODO).
    - `synthesize_yandex`: подготовка текста, римские→числа, разбиение на абзацы; на каждый абзац создаётся отдельное gRPC‑соединение/`Stub` (функция `synthesize_block_with_yandex`), внутри чанкинг + синтез, закрытие канала, сборка `pcm`.
  - Провайдер `auto`: сначала Yandex (чередование полов при отсутствии `--voice`), при ошибке — Google.
  - Скорость: Yandex `YANDEX_SPEED` (по умолчанию 0.80), Google `GOOGLE_SPEAKING_RATE` (по умолчанию 0.80).
  - Логи: сообщают провайдера, количество чанков, по Yandex — выбранный голос на абзац.

- `app.py`:
  - Очистка Markdown, римские, паузы, Yandex hints без `role`, скорость по умолчанию 0.80.
  - Чередование полов (Google / Yandex пер‑абзац) ещё не перенесено полностью из CLI (см. TODO).

## ENV‑переменные и настройки

- Yandex:
  - `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` — при отсутствии используются зашитые дефолты (маскировать в логах).
  - `YANDEX_SPEED` — по умолчанию `0.80`.
  - Списки предпочтений голосов для чередования:
    - `YANDEX_VOICE_MALE_PREFS` (CSV), дефолт: `filipp,ermil,zahhar,madirus`.
    - `YANDEX_VOICE_FEMALE_PREFS` (CSV), дефолт: `jane,alyss,oksana,tatyana`.

- Google:
  - `GOOGLE_APPLICATION_CREDENTIALS` (если нет — ADC через `gcloud` либо локальный ключ).
  - `GOOGLE_SPEAKING_RATE` — по умолчанию `0.80`.
  - Списки предпочтений голосов для чередования:
    - `GOOGLE_VOICE_MALE_PREFS` (CSV), дефолт: `ru-RU-Wavenet-D,ru-RU-Wavenet-B,ru-RU-Standard-D,ru-RU-Standard-B`.
    - `GOOGLE_VOICE_FEMALE_PREFS` (CSV), дефолт: `ru-RU-Wavenet-E,ru-RU-Wavenet-A,ru-RU-Standard-E,ru-RU-Standard-A`.

## Известные проблемы

1) Yandex подмена голоса: при одном gRPC‑соединении голос прилипает. Исправляем разносом на «1 абзац = 1 соединение». Эта логика уже добавлена, но требует финальной правки отступов/тестов в `synthesize_cli.py`.

2) Ошибки отступов (CLI) — необходимо исправить:
   - В `synthesize_cli.py` ленты диагностики показывали проблемы в районе:
     - Блок `synthesize_google`: внутри `synthesize_block(...)` строки формирования `ssml_chunk`, `synthesis_input`, `response`, `parts.append(...)` должны быть внутри цикла `for idx, chunk in enumerate(...):`.
     - Блок `synthesize_yandex`: вложенные функции `synthesize_block_with_yandex` и `synthesize_chunk_safe` должны иметь корректные отступы; следить, чтобы «Разбиение и синтез» не был вложен в `except` и не выезжал из области функции.

3) Web (Cloud Run) — чередование полов (Google) и пер‑абзацные соединения (Yandex) ещё не синхронизированы с CLI. Требуется перенос и новая сборка.

4) «missing value» в логах — разово наблюдался вывод после синтеза; потребуется локализовать источник (возможно, печать пустой переменной после завершения или в обработчике `--play`).

## Подробный план дальнейших действий

1. Исправить отступы/блоки в `synthesize_cli.py` (CLI):
   - Функция `synthesize_google`:
     - Внутри `synthesize_block(block_text, dyn_voice)` убедиться, что:
       ```python
       for idx, chunk in enumerate(chunks_local, start=1):
           print(...)
           ssml_chunk = text_to_ssml_google(chunk)
           synthesis_input = google_tts.SynthesisInput(ssml=ssml_chunk)
           response = client.synthesize_speech(...)
           parts.append(response.audio_content)
       return b"".join(parts)
       ```
     - Снаружи: по абзацам выбирать голос по полу (если `--voice` не задан), вызывать `synthesize_block` и склеивать.

   - Функция `synthesize_yandex`:
     - Вызов `synthesize_block_with_yandex(para, dyn_voice)` для каждого абзаца; внутри `synthesize_block_with_yandex`:
       - создать `channel`/`stub`;
       - прогнать чанки `para_chunks`, для каждого `synthesize_chunk_safe(chunk)`;
       - закрыть канал `channel.close()`; вернуть склейку.
     - Проверить, что `def synthesize_chunk_safe(...)` и «Разбиение и синтез» имеют корректные уровни отступов.

   - Прогнать линтер/установку:
     - `python3 -m pip install .`
     - Быстрый тест: `voice <file> -D --play`.

2. Добавить «умный фоллбек» на абзац (опционально):
   - Если текущий абзац должен быть мужским, а Яндекс всё равно отдал женский (детекция сложна; можно конфигом принудить мужские абзацы синтезировать в Google):
     - ENV‑флаг, например: `AUTO_MALE_GOOGLE=1` — мужские абзацы всегда в Google, женские — в Yandex.
   - Это гарантирует стабильное чередование полов без детекции фактического голоса.

3. Перенести логику чередования полов в Google для Web (`app.py`):
   - Аналогично CLI: по абзацам выбирать `ru-RU-Wavenet-D/E` (или из ENV‑списков), склеивать.

4. Перенести пер‑абзацные соединения для Yandex в Web (`app.py`).

5. Сборка и деплой Web (Cloud Run):
   - `gcloud builds submit --region europe-west1 --tag europe-west1-docker.pkg.dev/tts-webapp-1f514b/tts-repo/tts-app:latest`
   - Проверить `/healthz`, затем форму `/` с тестом абзацев/цитат/заголовков.

6. Проверки качества синтеза:
   - Тесты Markdown: заголовки, цитаты (`>`), списки (-/*/1.), инлайн‑код, ссылки/картинки — теги не проговариваются, паузы на местах.
   - Римские: «XII», «IV», «MCMXCIX» — Google читает числами, Yandex — арабские.
   - Чанкинг: границы звучат естественно, нет «обрывов» слов.
   - Чередование полов:
     - CLI (Yandex): лог показывает `alternate_gender=true`, чередование муж/жен по абзацам, аудио действительно чередуется. При проблемах — включить `AUTO_MALE_GOOGLE=1` для мужских абзацев.
     - CLI (Google): при отсутствии `--voice` — чередование `male/female` по абзацам.
     - Web: аналогичное поведение.

## Команды для быстрого старта

Локальная переустановка CLI:

```bash
python3 -m pip install .
```

CLI‑проверка (Yandex auto):

```bash
voice /path/to/text.md -D --play
```

CLI‑проверка (Google):

```bash
tts-synthesize --provider google --input-file /path/to/text.md -D --play
```

Cloud Build (Web):

```bash
gcloud builds submit --region europe-west1 \
  --tag europe-west1-docker.pkg.dev/tts-webapp-1f514b/tts-repo/tts-app:latest
```

## Дополнительно

- Алиас `voice` полностью равнозначен `tts-synthesize` (текущая CLI точка входа).
- Логи CLI печатают выбранного провайдера, креды (маскированно), фоллбеки, число чанков, выбранный голос на абзац.

---

Контакты для следующего исполнителя: использовать данный файл как чек‑лист. Основной приоритет — довести чередование полов до предсказуемого состояния (пер‑абзацные соединения для Yandex/Google‑чередование), устранить ошибки отступов/синтаксиса в `synthesize_cli.py`, синхронизировать изменения в `app.py` и выполнить новый деплой.


