#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import traceback
import re
import wave
import io
import grpc
from google.cloud import texttospeech as google_tts
import yandex.cloud.ai.tts.v3.tts_pb2 as yandex_tts_pb2
import yandex.cloud.ai.tts.v3.tts_service_pb2_grpc as yandex_tts_service_pb2_grpc
from pydub import AudioSegment
from pydub.utils import which
import pymorphy2
import subprocess
import shutil

# --- Yandex credentials (defaults removed; use env or CLI args) ---
YANDEX_DEFAULT_FOLDER_ID = ''
YANDEX_DEFAULT_API_KEY = ''
def _ensure_google_adc_or_key_present(allow_auto_login: bool = False):
    """Ensure Google ADC is available. If not, try to invoke gcloud ADC login non-interactively
    (no browser). If gcloud not available or ADC still missing, rely on local key auto-discovery.
    """
    # If a key path is already set, nothing to do
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        return
    # Quick ADC check
    try:
        proc = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return
    except Exception:
        # gcloud may be missing; fallback handled by key auto-discovery later
        return
    if not allow_auto_login:
        # Не инициируем интерактивный логин автоматически, чтобы не "висеть" в неинтерактивных сценариях
        return
    try:
        print('[Google] Требуется первичная авторизация ADC. Откройте ссылку и подтвердите доступ...', flush=True)
        subprocess.run(
            ['gcloud', 'auth', 'application-default', 'login', '--no-launch-browser'],
            check=False,
            stdout=None,
            stderr=None,
        )
    except Exception:
        pass


# --- TEXT PROCESSING LOGIC (from app.py) ---
FFMPEG_BINARY = os.environ.get("FFMPEG_BINARY") or which("ffmpeg")
if FFMPEG_BINARY:
    os.environ["FFMPEG_BINARY"] = FFMPEG_BINARY
    AudioSegment.converter = FFMPEG_BINARY
    # Backward-compat attributes some environments rely on
    try:
        AudioSegment.ffmpeg = FFMPEG_BINARY
    except Exception:
        pass
    ffprobe_path = which("ffprobe")
    if ffprobe_path:
        os.environ["FFPROBE_BINARY"] = ffprobe_path
        try:
            AudioSegment.ffprobe = ffprobe_path
        except Exception:
            pass
RU_TO_IPA_ROUGH = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'je', 'ё': 'jo',
    'ж': 'ʒ', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'tʃ', 'ш': 'ʃ', 'щ': 'ʃʲ', 'ъ': '',
    'ы': 'ɨ', 'ь': 'ʲ', 'э': 'e', 'ю': 'ju', 'я': 'ja'
}

TERMS_AND_NAMES = {
    "энтропию": "энтропи+ю", "абиогенного": "абиоге+нного", "рибозимами": "рибози+мами",
    "гипотезы": "гипо+тезы", "фенотипом": "феноти+пом", "генотипом": "геноти+пом",
    "метаболизм": "метаболи+зм", "хемоосмотическая": "хемоосмоти+ческая",
    "протонный": "прото+нный", "градиент": "градие+нт", "АТФ": "а-тэ-эф",
    "аденозинтрифосфата": "аденозинтрифосфа+та", "автотрофным": "автотро+фным",
    "Шрёдингер": "Шрёдингер", "Опарин": "Опа+рин", "Миллер": "Ми+ллер",
    "Юри": "Ю+ри", "Альтман": "А+льтман", "Чек": "Чек", "Гилберт": "Ги+лберт",
    "Митчеллом": "Ми+тчеллом", "Лейном": "Ле+йном", "Вехтерсхойзером": "Вехтерсхо+йзером",
    "LUCA": '<say-as interpret-as="characters">LUCA</say-as>'
}
TERMS_AND_NAMES_YANDEX = {k: v.replace('+', '') if v.startswith('<') else v for k, v in TERMS_AND_NAMES.items()}
TERMS_AND_NAMES_YANDEX["LUCA"] = "Лу+ка"

def get_yandex_gender_voice_preferences() -> dict:
    """Return preferred voices for Yandex by gender with env overrides.
    Env overrides: YANDEX_VOICE_MALE_PREFS, YANDEX_VOICE_FEMALE_PREFS (comma-separated)
    """
    male_env = os.getenv('YANDEX_VOICE_MALE_PREFS')
    female_env = os.getenv('YANDEX_VOICE_FEMALE_PREFS')
    male_list = [v.strip() for v in male_env.split(',')] if male_env else [
        'filipp', 'ermil', 'zahhar', 'madirus'
    ]
    female_list = [v.strip() for v in female_env.split(',')] if female_env else [
        'jane', 'alyss', 'oksana', 'tatyana'
    ]
    return { 'male': male_list, 'female': female_list }

def get_google_gender_voice_preferences() -> dict:
    """Return preferred voices for Google by gender with env overrides.
    Env: GOOGLE_VOICE_MALE_PREFS, GOOGLE_VOICE_FEMALE_PREFS (comma-separated)
    Defaults target WaveNet voices for ru-RU.
    """
    male_env = os.getenv('GOOGLE_VOICE_MALE_PREFS')
    female_env = os.getenv('GOOGLE_VOICE_FEMALE_PREFS')
    male_list = [v.strip() for v in male_env.split(',')] if male_env else [
        'ru-RU-Wavenet-D', 'ru-RU-Wavenet-B', 'ru-RU-Standard-D', 'ru-RU-Standard-B'
    ]
    female_list = [v.strip() for v in female_env.split(',')] if female_env else [
        'ru-RU-Wavenet-E', 'ru-RU-Wavenet-A', 'ru-RU-Standard-E', 'ru-RU-Standard-A'
    ]
    return { 'male': male_list, 'female': female_list }

def clean_markdown(text):
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\(\s*[^)]+:\s*[^)]*\)', '', text)
    return text

def insert_md_pauses(text: str) -> str:
    """Вставляет маркеры пауз после заголовков, вокруг цитат и на пустых строках.
    Используются плейсхолдеры [[PAUSENNN]], позже конвертируются в провайдер-специфичные теги.
    Также переводит квадратные скобки в круглые, чтобы их не проговаривали.
    """
    # - Удалить мягкие переносы (soft hyphen)
    text = text.replace('\u00AD', '')
    # - Склеить переносы по дефису: "Орга-\nнизм" -> "Организм"
    text = re.sub(r'-\s*\n\s*', '', text)
    # 0) Удалить изображения ![alt](url)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # 1) Ссылки [text](url) -> text (без URL)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    # 2) Удалить явные (http...) остатки
    text = re.sub(r'\(https?://[^)]+\)', '', text)
    # 3) Заменим квадратные скобки, чтобы не проговаривались
    text = re.sub(r'\[(.*?)\]', r'(\1)', text)

    lines = text.splitlines()
    out: list[str] = []
    in_quote_block = False
    for line in lines:
        raw = line.rstrip('\n')
        if not raw.strip():
            # пустая строка
            out.append('[[PAUSE800]]')
            if in_quote_block:
                out.append('[[PAUSE600]]')
                in_quote_block = False
            continue
        if re.match(r'^\s*#+\s+', raw):
            # заголовок
            content = re.sub(r'^\s*#+\s+', '', raw).strip()
            content = re.sub(r'[\*_`]+', '', content)
            out.append(content)
            out.append('[[PAUSE1200]]')
            if in_quote_block:
                out.append('[[PAUSE600]]')
                in_quote_block = False
            continue
        if re.match(r'^\s*>\s*', raw):
            # цитата (blockquote)
            content = re.sub(r'^\s*>\s*', '', raw).strip()
            content = re.sub(r'[\*_`]+', '', content)
            if not in_quote_block:
                out.append('[[PAUSE600]]')
                in_quote_block = True
            out.append(content)
            continue
        # Маркеры списков
        raw = re.sub(r'^\s*[-*+]\s+', '', raw)
        raw = re.sub(r'^\s*\d+\.\s+', '', raw)
        # обычная строка
        if in_quote_block:
            out.append('[[PAUSE600]]')
            in_quote_block = False
        # уберем MD-выделения
        plain = re.sub(r'[\*_`]+', '', raw)
        out.append(plain)
    if in_quote_block:
        out.append('[[PAUSE600]]')
    result = '\n'.join(out)
    # страховка: убираем одиночные звёздочки, если остались
    result = result.replace('*', '')
    return result

# --- Roman numerals ---
_ROMAN_MAP = {
    'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
    'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
    'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1
}

def _roman_to_int(roman: str) -> int:
    roman = roman.upper()
    i = 0
    value = 0
    while i < len(roman):
        if i + 1 < len(roman) and roman[i:i+2] in _ROMAN_MAP:
            value += _ROMAN_MAP[roman[i:i+2]]
            i += 2
        else:
            value += _ROMAN_MAP.get(roman[i], 0)
            i += 1
    return value

def replace_roman_numerals_for_google(text: str) -> str:
    pattern = r'\b(?=[MDCLXVI])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b'
    def _repl(m):
        roman = m.group(0)
        if not roman:
            return roman
        num = _roman_to_int(roman)
        return f'<say-as interpret-as="cardinal">{num}</say-as>'
    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)

def replace_roman_numerals_for_yandex(text: str) -> str:
    pattern = r'\b(?=[MDCLXVI])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b'
    def _repl(m):
        roman = m.group(0)
        if not roman:
            return roman
        return str(_roman_to_int(roman))
    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)

def get_ipa_with_stress(word_with_plus):
    if '+' not in word_with_plus:
        return ''.join([RU_TO_IPA_ROUGH.get(c.lower(), c) for c in word_with_plus])
    clean_word = word_with_plus.replace('+', '')
    vowels = "аеёиоуыэюя"
    try:
        stressed_vowel_index = word_with_plus.find('+') - 1
        if clean_word[stressed_vowel_index].lower() not in vowels:
             stressed_vowel_index = word_with_plus.find('+')
             if clean_word[stressed_vowel_index].lower() not in vowels:
                  return ''.join([RU_TO_IPA_ROUGH.get(c.lower(), c) for c in clean_word])
    except IndexError:
        return ''.join([RU_TO_IPA_ROUGH.get(c.lower(), c) for c in clean_word])
    ipa_string = ""
    for i, char in enumerate(clean_word):
        ipa_char = RU_TO_IPA_ROUGH.get(char.lower(), char)
        if i == stressed_vowel_index:
            ipa_string += 'ˈ' + ipa_char
        else:
            ipa_string += ipa_char
    ipa_string = ipa_string.replace('ˈje', 'ˈe').replace('ˈjo', 'ˈo').replace('ˈju', 'ˈu').replace('ˈja', 'ˈa')
    return ipa_string

def text_to_ssml_google(text):
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    for word, stressed_form in TERMS_AND_NAMES.items():
        if stressed_form.startswith('<'):
            ssml_replacement = stressed_form
        else:
            ipa = get_ipa_with_stress(stressed_form)
            ssml_replacement = f'<phoneme alphabet="ipa" ph="{ipa}">{word}</phoneme>'
        final_replacement = f'<break time="300ms"/>{ssml_replacement}<break time="300ms"/>'
        text = re.sub(r'\b' + re.escape(word) + r'\b', final_replacement, text)
    text = re.sub(r'([.!?])(\s+|$)', r'\1<break time="800ms"/>\2', text)
    text = re.sub(r'([,;—])(\s+|$)', r'\1<break time="400ms"/>\2', text)
    # Плейсхолдеры пауз -> SSML
    def _br(m):
        return f'<break time="{m.group(1)}ms"/>'
    text = re.sub(r'\[\[PAUSE(\d+)\]\]', _br, text)
    return f'<speak>{text}</speak>'

def text_to_yandex_markup(text):
    for word, stressed_form in TERMS_AND_NAMES_YANDEX.items():
        replacement = f"sil<[250]>{stressed_form}sil<[250]>"
        text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text)
    # Смягчаем паузы, чтобы избежать ощущения "рваности"
    text = re.sub(r'([.!?])(\s+|$)', r'\1 sil<[600]>\2', text)
    text = re.sub(r'([,;—])(\s+|$)', r'\1 sil<[250]>\2', text)
    # Плейсхолдеры пауз -> Yandex sil
    def _sil(m):
        return f'sil<[{m.group(1)}]>'
    text = re.sub(r'\[\[PAUSE(\d+)\]\]', _sil, text)
    return text

def split_text_google_ssml_safe(text, target_max_bytes=4500):
    """Разбивает исходный текст так, чтобы после оборачивания в SSML
    (акценты/паузы) размер одного запроса оставался < target_max_bytes байт.
    """
    def _find_natural_split_index(s: str, approx: int) -> int:
        if approx >= len(s):
            return len(s)
        delimiters = set(" \t\n\r.,;:!?—-…)")
        # влево
        for i in range(approx, max(-1, approx - 400), -1):
            if s[i] in delimiters:
                j = i + 1
                while j < len(s) and s[j].isspace():
                    j += 1
                return j if j < len(s) else i
        # вправо
        for i in range(approx, min(len(s), approx + 400)):
            if s[i] in delimiters:
                j = i + 1
                while j < len(s) and s[j].isspace():
                    j += 1
                return j if j < len(s) else i
        return approx
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        ssml = text_to_ssml_google(candidate)
        if len(ssml.encode('utf-8')) <= target_max_bytes:
            current = candidate
            continue
        # Текущий + предложение не влезает: закрываем текущий, начинаем новое
        if current:
            chunks.append(current)
            current = ""
        # Если одно предложение всё ещё слишком длинное, резать по словам
        if len(text_to_ssml_google(sentence).encode('utf-8')) > target_max_bytes:
            # Режем по естественной границе около половины
            idx = _find_natural_split_index(sentence, max(1, len(sentence)//2))
            left = sentence[:idx].rstrip()
            right = sentence[idx:].lstrip()
            if left:
                chunks.append(left)
            if right:
                # возможно, правая часть все ещё длинная — вернётся в цикл
                sentences = [right] + sentences[sentences.index(sentence)+1:]
                continue
        else:
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def split_text(text, max_len=4500):
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_len:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def split_text_yandex_safe(text, target_max_chars=500):
    """Агрессивное разбиение для SpeechKit v3. Ставит жёсткий предел длины
    входного текста, режет по предложениям, при необходимости — по словам.
    """
    def _find_natural_split_index(s: str, approx: int) -> int:
        if approx >= len(s):
            return len(s)
        delimiters = set(" \t\n\r.,;:!?—-…)")
        for i in range(approx, max(-1, approx - 400), -1):
            if s[i] in delimiters:
                j = i + 1
                while j < len(s) and s[j].isspace():
                    j += 1
                return j if j < len(s) else i
        for i in range(approx, min(len(s), approx + 400)):
            if s[i] in delimiters:
                j = i + 1
                while j < len(s) and s[j].isspace():
                    j += 1
                return j if j < len(s) else i
        return approx

    # Сначала режем по абзацам
    paragraphs = [p for p in re.split(r'\n\s*\n+', text) if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(re.split(r'(?<=[.!?])\s+', para.replace('\n', ' ')))
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= target_max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sentence) > target_max_chars:
            idx = _find_natural_split_index(sentence, max(1, len(sentence)//2))
            left = sentence[:idx].rstrip()
            right = sentence[idx:].lstrip()
            if left:
                chunks.append(left)
            if right:
                sentences = [right] + sentences[sentences.index(sentence)+1:]
                continue
        else:
            current = sentence
    if current:
        chunks.append(current)
    return chunks

# --- SYNTHESIS FUNCTIONS (from app.py) ---
def _export_wav_to_mp3_with_ffmpeg(temp_wav_path: str, output_mp3_path: str) -> None:
    ffmpeg_bin = os.environ.get('FFMPEG_BINARY') or which('ffmpeg') or 'ffmpeg'
    try:
        subprocess.run(
            [ffmpeg_bin, '-y', '-i', temp_wav_path, '-b:a', '192k', output_mp3_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'ffmpeg conversion failed: {e.stderr[:4000]}')
    finally:
        try:
            os.remove(temp_wav_path)
        except OSError:
            pass
def synthesize_google(text, voice_name, output_file, speaking_rate: float = 0.80):
    # Try to auto-use local key file if ADC not configured
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        candidate_paths = [
            os.path.join(os.getcwd(), 'keys', 'tts-cli-sa.json'),
            os.path.join(os.path.dirname(__file__), 'keys', 'tts-cli-sa.json'),
            os.path.join(os.getcwd(), 'tts-cli-sa.json'),
            os.path.join(os.path.expanduser('~'), '.config', 'yandex-speech-cli', 'tts-cli-sa.json'),
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = p
                break
    # If still no key, try to ensure ADC through gcloud
    _ensure_google_adc_or_key_present()

    client = google_tts.TextToSpeechClient()
    audio_config = google_tts.AudioConfig(
        audio_encoding=google_tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        speaking_rate=float(speaking_rate)
    )
    cleaned_text = insert_md_pauses(text)
    cleaned_text = replace_roman_numerals_for_google(cleaned_text)
    paragraphs = [p for p in re.split(r'\n\s*\n+', cleaned_text) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned_text]
    # Чередование полов по абзацам, если голос явно не задан
    alternate = not (voice_name and voice_name.strip())
    prefs = get_google_gender_voice_preferences()
    def synthesize_block(block_text: str, dyn_voice: str) -> bytes:
        voice_params = google_tts.VoiceSelectionParams(language_code="ru-RU", name=dyn_voice)
        chunks_local = split_text_google_ssml_safe(block_text, target_max_bytes=4500)
        print(f"[Google] Чанков к синтезу: {len(chunks_local)}", flush=True)
        parts = []
        for idx, chunk in enumerate(chunks_local, start=1):
            print(f"[Google] Чанк {idx}/{len(chunks_local)}...", flush=True)
            ssml_chunk = text_to_ssml_google(chunk)
            synthesis_input = google_tts.SynthesisInput(ssml=ssml_chunk)
            response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
            parts.append(response.audio_content)
        return b"".join(parts)
    audio_segments = []
    for i, para in enumerate(paragraphs):
        if alternate:
            gender = 'male' if i % 2 == 0 else 'female'
            cand = prefs.get(gender, [])
            g_voice = cand[0] if cand else (voice_name or 'ru-RU-Wavenet-D')
        else:
            g_voice = voice_name or 'ru-RU-Wavenet-D'
        audio_segments.append(synthesize_block(para, g_voice))
        
    wav_data = b"".join(audio_segments)
    
    # Save to file
    temp_wav = output_file
    if output_file.lower().endswith('.mp3'):
        # We'll save as wav first, then convert
        temp_wav = output_file + ".wav"

    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # LINEAR16 is 2 bytes
        wf.setframerate(48000)
        wf.writeframes(wav_data)

    if output_file.lower().endswith('.mp3'):
        _export_wav_to_mp3_with_ffmpeg(temp_wav, output_file)


def synthesize_yandex(text, voice_name, api_key, folder_id, output_file, speed: float = 0.80, role: str = 'neutral'):
    """Стабильный синтез через SpeechKit v3 с RAW PCM и конвертацией через ffmpeg.
    Избегаем pydub для промежуточной обработки, чтобы не зависать на некорректных чанках.
    """
    cleaned_text = insert_md_pauses(text)
    cleaned_text = replace_roman_numerals_for_yandex(cleaned_text)
    # Разбиваем на абзацы для возможного чередования (если голос не задан)
    paragraphs = [p for p in re.split(r'\n\s*\n+', cleaned_text) if p.strip()]

    # Получаем сырые PCM-данные 16-bit, 48kHz, mono. На каждый абзац — отдельный канал/stub.
    def synthesize_block_with_yandex(block_text: str, dyn_voice: str) -> bytes:
        cred = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel('tts.api.cloud.yandex.net:443', cred)
        stub = yandex_tts_service_pb2_grpc.SynthesizerStub(channel)

        def synthesize_chunk_safe(text_chunk: str) -> bytes:
            try:
                request = yandex_tts_pb2.UtteranceSynthesisRequest(
                    text=text_chunk,
                    output_audio_spec=yandex_tts_pb2.AudioFormatOptions(
                        raw_audio=yandex_tts_pb2.RawAudio(
                            audio_encoding=yandex_tts_pb2.RawAudio.LINEAR16_PCM,
                            sample_rate_hertz=48000
                        )
                    ),
                    hints=[yandex_tts_pb2.Hints(voice=dyn_voice, speed=float(speed))],
                    loudness_normalization_type=yandex_tts_pb2.UtteranceSynthesisRequest.LUFS,
                )
                meta = (
                    ('authorization', f'Api-Key {api_key}'),
                    ('x-folder-id', folder_id)
                )
                it = stub.UtteranceSynthesis(request, metadata=meta)
                return b"".join([resp.audio_chunk.data for resp in it if getattr(resp, 'audio_chunk', None)])
            except grpc.RpcError as e:
                detail = getattr(e, 'details', lambda: '')() or str(e)
                if e.code() == grpc.StatusCode.INVALID_ARGUMENT and 'Too long text' in detail and len(text_chunk) > 60:
                    mid = len(text_chunk) // 2
                    left = text_chunk[:mid].rstrip()
                    right = text_chunk[mid:].lstrip()
                    return synthesize_chunk_safe(left) + synthesize_chunk_safe(right)
                raise

        # Разбиение и синтез: сначала пробуем целиком абзац, при ошибке Too long дробим рекурсивно
        yandex_ready_para = text_to_yandex_markup(block_text)
        parts = []
        parts.append(synthesize_chunk_safe(yandex_ready_para))
        try:
            channel.close()
        except Exception:
            pass
        return b"".join(parts)

    # Если голос не задан — чередуем пол: male/female; подбираем голоса из предпочтений (с env-override)
    alternate = not (voice_name and voice_name.strip())
    if not paragraphs:
        paragraphs = [cleaned_text]
    combined_segments: list[bytes] = []
    prefs = get_yandex_gender_voice_preferences()
    genders = ['male', 'female']
    total_paragraphs = len(paragraphs)
    print(f"[Yandex] Абзацев: {total_paragraphs} (alternate_gender={str(alternate).lower()})", flush=True)
    force_male_google = bool(int(os.getenv('AUTO_MALE_GOOGLE', '0') or '0'))
    for i, para in enumerate(paragraphs):
        gender = genders[i % 2] if alternate else None
        if alternate and force_male_google and gender == 'male':
            g_prefs = get_google_gender_voice_preferences()
            g_voice = (g_prefs.get('male') or ['ru-RU-Wavenet-D'])[0]
            print(f"[Google] Параграф {i+1}/{total_paragraphs} (forced male via Google): голос={g_voice}", flush=True)
            client = google_tts.TextToSpeechClient()
            voice_params = google_tts.VoiceSelectionParams(language_code="ru-RU", name=g_voice)
            audio_config_g = google_tts.AudioConfig(audio_encoding=google_tts.AudioEncoding.LINEAR16, sample_rate_hertz=48000, speaking_rate=float(os.getenv('GOOGLE_SPEAKING_RATE', '0.80')))
            chunks_local = split_text_google_ssml_safe(replace_roman_numerals_for_google(insert_md_pauses(para)), target_max_bytes=4500)
            bytes_parts = []
            for chunk in chunks_local:
                ssml_chunk = text_to_ssml_google(chunk)
                synthesis_input = google_tts.SynthesisInput(ssml=ssml_chunk)
                resp = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config_g)
                bytes_parts.append(resp.audio_content)
            combined_segments.append(b''.join(bytes_parts))
            continue

        if alternate:
            candidate_voices = prefs.get(gender, [])
            dyn_voice = candidate_voices[0] if candidate_voices else 'ermil'
        else:
            dyn_voice = voice_name or 'ermil'
        print(f"[Yandex] Параграф {i+1}/{total_paragraphs}: голос={dyn_voice}", flush=True)
        try:
            combined_segments.append(synthesize_block_with_yandex(para, dyn_voice))
        except grpc.RpcError as e:
            print(f"[Yandex->Google] Fallback paragraph {i+1}: {e}", file=sys.stderr)
            fallback_voice = 'ru-RU-Wavenet-D' if (gender == 'male') else 'ru-RU-Wavenet-E'
            client = google_tts.TextToSpeechClient()
            voice_params = google_tts.VoiceSelectionParams(language_code="ru-RU", name=fallback_voice)
            audio_config_g = google_tts.AudioConfig(audio_encoding=google_tts.AudioEncoding.LINEAR16, sample_rate_hertz=48000, speaking_rate=float(os.getenv('GOOGLE_SPEAKING_RATE', '0.80')))
            chunks_local = split_text_google_ssml_safe(replace_roman_numerals_for_google(insert_md_pauses(para)), target_max_bytes=4500)
            bytes_parts = []
            for chunk in chunks_local:
                ssml_chunk = text_to_ssml_google(chunk)
                synthesis_input = google_tts.SynthesisInput(ssml=ssml_chunk)
                resp = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config_g)
                bytes_parts.append(resp.audio_content)
            combined_segments.append(b''.join(bytes_parts))
    pcm_data = b"".join(combined_segments)

    # Экспорт в MP3 через ffmpeg напрямую из RAW PCM
    if output_file.lower().endswith('.mp3'):
        ffmpeg_bin = os.environ.get('FFMPEG_BINARY') or which('ffmpeg') or 'ffmpeg'
        try:
            proc = subprocess.Popen([
                ffmpeg_bin,
                '-hide_banner', '-loglevel', 'error',
                '-y',
                '-f', 's16le', '-ar', '48000', '-ac', '1', '-i', 'pipe:0',
                '-codec:a', 'libmp3lame', '-b:a', '192k',
                output_file
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_data, stderr_data = proc.communicate(input=pcm_data)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr_data.decode('utf-8', errors='ignore')[:2000]}")
        except FileNotFoundError:
            # Нет ffmpeg — fallback: сохраняем WAV
            print('[WARN] ffmpeg не найден. Сохраняю WAV.', file=sys.stderr)
            wav_path = os.path.splitext(output_file)[0] + '.wav'
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm_data)
    else:
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm_data)



# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech CLI using Yandex or Google.")
    parser.add_argument("--version", "-V", action='version', version='yandex-speech-cli 0.1.2')
    parser.add_argument("--verbose", "-v", action='count', default=0, help='Increase verbosity (-v, -vv, -vvv)')
    parser.add_argument("--quiet", "-q", action='store_true', help='Suppress non-error output')
    parser.add_argument("--provider", type=str, choices=['auto', 'google', 'yandex'], help="TTS provider. Default: auto (Yandex→Google fallback)")
    # Positional text (optional). Supports multiple words without quotes.
    parser.add_argument("text_args", nargs='*', help="Text to synthesize (positional, supports multiple words without quotes)")
    parser.add_argument("--text", type=str, help="Text to synthesize. If omitted, stdin is read.")
    parser.add_argument("--input-file", type=str, help="Path to a text file.")
    parser.add_argument("--output-file", type=str, help="Output filename or path. If omitted: derive from input file name or use UUID.")
    parser.add_argument("-D", "--desktop", action='store_true', help="Save result on Desktop instead of current directory (applies to default naming and bare filenames)")
    parser.add_argument("--voice", type=str, help="Voice name. Default: google->ru-RU-Wavenet-D, yandex->ermil")
    parser.add_argument("-p", "--play", action='store_true', help="Open the resulting audio in VLC (or default player) after synthesis")
    
    # Provider-specific arguments
    parser.add_argument("--yandex-api-key", type=str, help="Yandex API Key. Can also be set via YANDEX_API_KEY env var.")
    parser.add_argument("--yandex-folder-id", type=str, help="Yandex Folder ID. Can also be set via YANDEX_FOLDER_ID env var.")
    
    args = parser.parse_args()

    try:
        # Logging level
        log_level = logging.WARNING if args.quiet else (logging.INFO if args.verbose == 0 else logging.DEBUG)
        logging.basicConfig(level=log_level)
        # Infer input source
        # If a single positional argument points to an existing file, treat it as --input-file
        if not args.input_file and not args.text and args.text_args and len(args.text_args) == 1:
            possible_path = args.text_args[0]
            if os.path.isfile(possible_path):
                args.input_file = possible_path
                args.text_args = []

        # Get text
        if args.text:
            text = args.text
        elif args.input_file:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif args.text_args:
            text = " ".join(args.text_args)
        else:
            # Read from stdin if available
            if not sys.stdin.isatty():
                text = sys.stdin.read()
            else:
                print("Error: Provide --input-file or pipe text to stdin.")
                parser.print_help()
                return

        # Defaults
        provider = (args.provider or 'auto').lower()
        if provider not in ('auto', 'google', 'yandex'):
            provider = 'auto'

        # Voice will be resolved per-provider at call time to avoid mismatches in auto fallback
        user_voice = args.voice

        # Resolve output path
        def _ensure_mp3_extension(path_str: str) -> str:
            lower = path_str.lower()
            if lower.endswith('.mp3') or lower.endswith('.wav'):
                return path_str
            return path_str + '.mp3'

        base_dir_desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        base_dir_cwd = os.getcwd()
        prefer_desktop = bool(args.desktop) and os.path.isdir(base_dir_desktop)
        base_dir_default = base_dir_desktop if prefer_desktop else base_dir_cwd

        output_file = args.output_file
        if output_file:
            # If user provided only a filename (no directory separators), honor -D by placing on Desktop
            if not os.path.isabs(output_file) and os.path.dirname(output_file) == '':
                output_file = os.path.join(base_dir_default, output_file)
            # Ensure extension if missing
            output_file = _ensure_mp3_extension(output_file)
        else:
            # Derive name from input file, else use UUID; place according to -D flag
            if args.input_file:
                base_name = os.path.splitext(os.path.basename(args.input_file))[0]
                output_file = os.path.join(base_dir_default, f"{base_name}.mp3")
            else:
                import uuid
                output_file = os.path.join(base_dir_default, f"{uuid.uuid4()}.mp3")

        # Synthesize with provider selection and explicit fallback
        def _print_google_cred_source():
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                print(f"[Google] credentials: keyfile={os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}", file=sys.stderr)
            else:
                try:
                    proc = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if proc.returncode == 0 and proc.stdout.strip():
                        print("[Google] credentials: ADC token (gcloud)", file=sys.stderr)
                    else:
                        print("[Google] credentials: auto-discovery (may use env or metadata)", file=sys.stderr)
                except Exception:
                    print("[Google] credentials: unknown (gcloud not available)", file=sys.stderr)

        def _synthesize_google_report():
            if not args.quiet:
                print("Synthesizing with Google...", file=sys.stderr)
                _print_google_cred_source()
            g_voice = user_voice or 'ru-RU-Wavenet-D'
            synthesize_google(text, g_voice, output_file, speaking_rate=float(os.getenv('GOOGLE_SPEAKING_RATE', '0.80')))
            if not args.quiet:
                print(f"Synthesis complete. Output saved to {output_file}", file=sys.stderr)

        if provider == 'google':
            _synthesize_google_report()

        elif provider == 'yandex':
            # Fallback chain for Yandex keys: args -> env -> defaults
            api_key = args.yandex_api_key or os.getenv('YANDEX_API_KEY') or YANDEX_DEFAULT_API_KEY
            folder_id = args.yandex_folder_id or os.getenv('YANDEX_FOLDER_ID') or YANDEX_DEFAULT_FOLDER_ID
            if not api_key or not folder_id:
                print("Error: Yandex API Key and Folder ID are required (args/env/default).", file=sys.stderr)
                parser.print_help()
                return
            if not args.quiet:
                masked = (api_key[:4] + '...' + api_key[-4:]) if len(api_key) > 8 else 'set'
                print(f"Synthesizing with Yandex... (folder_id={folder_id}, api_key={masked})", file=sys.stderr)
            # Не подставляем дефолтный голос, чтобы при отсутствии --voice включалось чередование полов
            y_voice = user_voice
            synthesize_yandex(text, y_voice, api_key, folder_id, output_file, speed=float(os.getenv('YANDEX_SPEED', '0.80')))
            if not args.quiet:
                print(f"Synthesis complete. Output saved to {output_file}", file=sys.stderr)

        else:  # auto
            # Try Yandex first
            api_key = args.yandex_api_key or os.getenv('YANDEX_API_KEY') or YANDEX_DEFAULT_API_KEY
            folder_id = args.yandex_folder_id or os.getenv('YANDEX_FOLDER_ID') or YANDEX_DEFAULT_FOLDER_ID
            fallback_reason = None
            try:
                if not args.quiet:
                    masked = (api_key[:4] + '...' + api_key[-4:]) if api_key else 'missing'
                    print(f"[auto] Trying Yandex first... (folder_id={folder_id or 'missing'}, api_key={masked})", file=sys.stderr)
                if not api_key or not folder_id:
                    raise RuntimeError('Missing Yandex credentials')
                # Если голос не задан, передаём None — внутри включится чередование полов
                y_voice = user_voice
                synthesize_yandex(text, y_voice, api_key, folder_id, output_file, speed=float(os.getenv('YANDEX_SPEED', '0.80')))
                if not args.quiet:
                    print(f"[auto] Provider: Yandex. Output saved to {output_file}", file=sys.stderr)
            except grpc.RpcError as e:
                fallback_reason = f"Yandex RPC error: {e.code().name} {getattr(e, 'details', lambda: '')()}"
            except Exception as e:
                fallback_reason = f"Yandex error: {e}"

            if fallback_reason:
                print(f"[auto] Fallback to Google: {fallback_reason}", file=sys.stderr)
                _synthesize_google_report()

        # Auto-open player
        if args.play:
            # На macOS всегда открываем полноценное GUI-приложение
            if sys.platform == 'darwin':
                # Сначала пытаемся через AppleScript, чтобы гарантировать поднятие окна
                try:
                    osa = f'tell application "VLC" to activate\n' \
                          f'tell application "VLC" to open POSIX file "{output_file}"'
                    subprocess.run(['osascript', '-e', osa], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    # Fallback: обычный open -a VLC file
                    try:
                        subprocess.Popen(['open', '-a', 'VLC', output_file])
                    except Exception:
                        pass
            else:
                player = shutil.which('vlc') or shutil.which('VLC')
                if player:
                    try:
                        subprocess.Popen([player, output_file])
                    except Exception:
                        pass
                else:
                    # generic fallback
                    opener = shutil.which('xdg-open')
                    if opener:
                        subprocess.Popen([opener, output_file])
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
        traceback.print_exc()



if __name__ == '__main__':
    main()

