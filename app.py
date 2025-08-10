import os
import re
import wave
import uuid
import logging

# Yandex credentials defaults removed to avoid committing secrets. Use env.
YANDEX_DEFAULT_FOLDER_ID = ''
YANDEX_DEFAULT_API_KEY = ''
import grpc
import subprocess
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from google.cloud import texttospeech as google_tts
import yandex.cloud.ai.tts.v3.tts_pb2 as yandex_tts_pb2
import yandex.cloud.ai.tts.v3.tts_service_pb2_grpc as yandex_tts_service_pb2_grpc
from pydub import AudioSegment
import pymorphy2

# --- CONFIG & INITIALIZATION ---
app = Flask(__name__)
# Structured logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
app.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit for uploads

# --- TEXT PROCESSING LOGIC (OUR RULES) ---
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
    """Списки предпочтений голосов для Yandex по полу с env-override."""
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
    """Списки предпочтений голосов для Google по полу с env-override."""
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
    # Удалить изображения ![alt](url)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # Ссылки [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\(https?://[^)]+\)', '', text)
    # квадратные скобки -> круглые
    text = re.sub(r'\[(.*?)\]', r'(\1)', text)
    lines = text.splitlines()
    out = []
    in_quote_block = False
    for line in lines:
        raw = line.rstrip('\n')
        if not raw.strip():
            out.append('[[PAUSE800]]')
            if in_quote_block:
                out.append('[[PAUSE600]]')
                in_quote_block = False
            continue
        if re.match(r'^\s*#+\s+', raw):
            content = re.sub(r'^\s*#+\s+', '', raw).strip()
            content = re.sub(r'[\*_`]+', '', content)
            out.append(content)
            out.append('[[PAUSE1200]]')
            if in_quote_block:
                out.append('[[PAUSE600]]')
                in_quote_block = False
            continue
        if re.match(r'^\s*>\s*', raw):
            content = re.sub(r'^\s*>\s*', '', raw).strip()
            content = re.sub(r'[\*_`]+', '', content)
            if not in_quote_block:
                out.append('[[PAUSE600]]')
                in_quote_block = True
            out.append(content)
            continue
        # Списки
        raw = re.sub(r'^\s*[-*+]\s+', '', raw)
        raw = re.sub(r'^\s*\d+\.\s+', '', raw)
        if in_quote_block:
            out.append('[[PAUSE600]]')
            in_quote_block = False
        plain = re.sub(r'[\*_`]+', '', raw)
        out.append(plain)
    if in_quote_block:
        out.append('[[PAUSE600]]')
    result = '\n'.join(out)
    result = result.replace('*', '')
    return result

# Roman numerals
_ROMAN_MAP = {
    'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
    'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
    'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1
}

def _roman_to_int(roman: str) -> int:
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
    def _repl(m):
        roman = m.group(0)
        num = _roman_to_int(roman)
        return f'<say-as interpret-as="cardinal">{num}</say-as>'
    return re.sub(r'\bM{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})\b', _repl, text)

def replace_roman_numerals_for_yandex(text: str) -> str:
    def _repl(m):
        roman = m.group(0)
        return str(_roman_to_int(roman))
    return re.sub(r'\bM{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})\b', _repl, text)

def get_ipa_with_stress(word_with_plus):
    # ... (Implementation from synthesize_google_long_v8.py)
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
    # ... (Implementation from synthesize_google_long_v8.py)
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
    text = re.sub(r'\[\[PAUSE(\d+)\]\]', lambda m: f'<break time="{m.group(1)}ms"/>', text)
    return f'<speak>{text}</speak>'

def text_to_yandex_markup(text):
    # ... (Implementation from synthesize_yandex_long_v4.py)
    for word, stressed_form in TERMS_AND_NAMES_YANDEX.items():
        replacement = f"sil<[300]>{stressed_form}sil<[300]>"
        text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text)
    text = re.sub(r'([.!?])(\s+|$)', r'\1 sil<[800]>\2', text)
    text = re.sub(r'([,;—])(\s+|$)', r'\1 sil<[400]>\2', text)
    text = re.sub(r'\[\[PAUSE(\d+)\]\]', lambda m: f'sil<[{m.group(1)}]>', text)
    return text

def split_text(text, max_len=4500): # Increased limit for Google
    # ... (Robust splitting logic)
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

def split_text_google_ssml_safe(text, target_max_bytes=4500):
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
            words = sentence.split()
            wbuf = []
            for w in words:
                cand = (" ".join(wbuf + [w])).strip()
                if len(text_to_ssml_google(cand).encode('utf-8')) <= target_max_bytes:
                    wbuf.append(w)
                else:
                    if wbuf:
                        chunks.append(" ".join(wbuf))
                        wbuf = [w]
                    else:
                        # Слово само по себе слишком длинное из-за разметки — добавляем как есть
                        chunks.append(w)
                        wbuf = []
            if wbuf:
                chunks.append(" ".join(wbuf))
        else:
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def split_text_yandex_safe(text, target_max_chars=500):
    # Агрессивно безопасное разбиение для SpeechKit v3: короткие фрагменты
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
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
        # Если само предложение длиннее лимита — режем по словам
        if len(sentence) > target_max_chars:
            words = sentence.split()
            buf = []
            for w in words:
                cand = (" ".join(buf + [w])).strip()
                if len(cand) <= target_max_chars:
                    buf.append(w)
                else:
                    if buf:
                        chunks.append(" ".join(buf))
                        buf = [w]
                    else:
                        # слово слишком длинное — добавляем как есть
                        chunks.append(w)
                        buf = []
            if buf:
                chunks.append(" ".join(buf))
        else:
            current = sentence
    if current:
        chunks.append(current)
    return chunks

# --- SYNTHESIS FUNCTIONS ---
def synthesize_google(text, voice_name, speaking_rate: float = 0.80, pitch_semitones: int = 0):
    """Синтез Google с поддержкой чередования полов по абзацам при отсутствующем voice_name."""
    client = google_tts.TextToSpeechClient()
    audio_config = google_tts.AudioConfig(
        audio_encoding=google_tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        speaking_rate=float(speaking_rate),
        pitch=pitch_semitones
    )

    cleaned_text = insert_md_pauses(text)
    cleaned_text = replace_roman_numerals_for_google(cleaned_text)
    paragraphs = [p for p in re.split(r'\n\s*\n+', cleaned_text) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned_text]

    alternate = not (voice_name and str(voice_name).strip())
    prefs = get_google_gender_voice_preferences()

    def synthesize_block(block_text: str, dyn_voice: str) -> bytes:
        voice_params = google_tts.VoiceSelectionParams(language_code="ru-RU", name=dyn_voice)
        chunks_local = split_text_google_ssml_safe(block_text, target_max_bytes=4500)
        app.logger.info(f"[google] chunks={len(chunks_local)} voice={dyn_voice}")
        parts: list[bytes] = []
        for idx, chunk in enumerate(chunks_local, start=1):
            app.logger.info(f"[google] chunk {idx}/{len(chunks_local)} size={len(chunk)}")
            ssml_chunk = text_to_ssml_google(chunk)
            synthesis_input = google_tts.SynthesisInput(ssml=ssml_chunk)
            response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
            parts.append(response.audio_content)
        return b"".join(parts)

    audio_segments: list[bytes] = []
    for i, para in enumerate(paragraphs):
        if alternate:
            gender = 'male' if i % 2 == 0 else 'female'
            cand = prefs.get(gender, [])
            g_voice = cand[0] if cand else (voice_name or 'ru-RU-Wavenet-D')
        else:
            g_voice = voice_name or 'ru-RU-Wavenet-D'
        audio_segments.append(synthesize_block(para, g_voice))
    return b"".join(audio_segments)

def synthesize_yandex(text, voice_name, api_key, folder_id, speed: float = 0.80, role: str = "neutral"):
    """Синтез Yandex с отдельным соединением на абзац и опциональным чередованием полов."""
    cleaned_text = insert_md_pauses(text)
    cleaned_text = replace_roman_numerals_for_yandex(cleaned_text)

    paragraphs = [p for p in re.split(r'\n\s*\n+', cleaned_text) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned_text]

    alternate = not (voice_name and str(voice_name).strip())
    prefs = get_yandex_gender_voice_preferences()
    genders = ['male', 'female']
    app.logger.info(f"[yandex] paragraphs={len(paragraphs)} alternate_gender={str(alternate).lower()}")

    def _find_split_index_natural(text_chunk: str) -> int:
        n = len(text_chunk)
        if n < 2:
            return n // 2
        mid = n // 2
        delimiters = set(" \t\n\r.,;:!?—-…")
        for i in range(mid, max(-1, mid - 2000), -1):
            if text_chunk[i] in delimiters:
                j = i + 1
                while j < n and text_chunk[j].isspace():
                    j += 1
                return j if j < n else i
        for i in range(mid, min(n, mid + 2000)):
            if text_chunk[i] in delimiters:
                j = i + 1
                while j < n and text_chunk[j].isspace():
                    j += 1
                return j if j < n else i
        return mid

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
                iam_token = os.getenv('YANDEX_IAM_TOKEN')
                if iam_token:
                    meta = (
                        ('authorization', f'Bearer {iam_token}'),
                        ('x-folder-id', folder_id),
                    )
                else:
                    meta = (
                        ('authorization', f'Api-Key {api_key}'),
                        ('x-folder-id', folder_id),
                    )
                it = stub.UtteranceSynthesis(request, metadata=meta)
                return b"".join([resp.audio_chunk.data for resp in it if getattr(resp, 'audio_chunk', None)])
            except grpc.RpcError as e:
                detail = getattr(e, 'details', lambda: '')() or str(e)
                if e.code() == grpc.StatusCode.INVALID_ARGUMENT and 'Too long text' in detail and len(text_chunk) > 60:
                    split_idx = _find_split_index_natural(text_chunk)
                    left = text_chunk[:split_idx].rstrip()
                    right = text_chunk[split_idx:].lstrip()
                    if not left or not right:
                        mid = len(text_chunk) // 2
                        left = text_chunk[:mid].rstrip()
                        right = text_chunk[mid:].lstrip()
                    return synthesize_chunk_safe(left) + synthesize_chunk_safe(right)
                raise

        yandex_ready_para = text_to_yandex_markup(block_text)
        para_chunks = split_text_yandex_safe(yandex_ready_para, target_max_chars=700)
        parts: list[bytes] = []
        for chunk in para_chunks:
            parts.append(synthesize_chunk_safe(chunk))
        try:
            channel.close()
        except Exception:
            pass
        return b"".join(parts)

    combined_segments: list[bytes] = []
    for i, para in enumerate(paragraphs):
        if alternate:
            gender = genders[i % 2]
            candidate_voices = prefs.get(gender, [])
            dyn_voice = candidate_voices[0] if candidate_voices else 'ermil'
        else:
            dyn_voice = voice_name or 'ermil'
        app.logger.info(f"[yandex] paragraph {i+1}/{len(paragraphs)} voice={dyn_voice}")
        combined_segments.append(synthesize_block_with_yandex(para, dyn_voice))
    return b"".join(combined_segments)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/healthz')
def healthz():
    try:
        return jsonify({
            'status': 'ok',
            'ffmpeg': os.getenv('FFMPEG_BINARY', 'auto'),
        }), 200
    except Exception as e:
        app.logger.error(f"healthz error: {e}")
        return jsonify({'status': 'error', 'detail': str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        provider = (request.form.get('provider') or 'auto').strip().lower()
        voice = request.form.get('voice')
        text = request.form.get('text_input')
        file = request.files.get('file_input')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            os.remove(filepath)

        if not text or not text.strip():
            return jsonify({'success': False, 'error': 'No text provided for synthesis.'}), 400

        report = { 'provider': None, 'notice': None, 'fallback_reason': None }

        if provider == 'google':
            try:
                g_speed = float(request.form.get('speed') or 0.8)
            except Exception:
                g_speed = 0.8
            try:
                g_pitch = int(request.form.get('pitch') or 0)
            except Exception:
                g_pitch = 0
            wav_data = synthesize_google(text, voice, speaking_rate=g_speed, pitch_semitones=g_pitch)
            report['provider'] = 'google'
        elif provider == 'yandex':
            api_key = os.getenv('YANDEX_API_KEY') or YANDEX_DEFAULT_API_KEY
            folder_id = os.getenv('YANDEX_FOLDER_ID') or YANDEX_DEFAULT_FOLDER_ID
            if not api_key or not folder_id:
                return jsonify({'success': False, 'error': 'Yandex credentials not configured on server.'}), 500
            try:
                speed = float(request.form.get('speed') or 0.85)
            except Exception:
                speed = 0.85
            role = (request.form.get('emotion') or 'neutral').strip().lower()
            if role not in ('neutral', 'good', 'evil'):
                role = 'neutral'
            pcm_data = synthesize_yandex(text, voice, api_key, folder_id, speed=speed, role=role)
            unique_id = uuid.uuid4()
            mp3_filename = f"{unique_id}.mp3"
            mp3_filepath = os.path.join(app.config['STATIC_FOLDER'], mp3_filename)
            ffmpeg_bin = os.getenv('FFMPEG_BINARY', 'ffmpeg')
            cmd = [ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y', '-f', 's16le', '-ar', '48000', '-ac', '1', '-i', 'pipe:0', '-codec:a', 'libmp3lame', '-b:a', '192k', mp3_filepath]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_data, stderr_data = proc.communicate(input=pcm_data)
            if proc.returncode != 0:
                return jsonify({'success': False, 'error': f"ffmpeg conversion failed: {stderr_data.decode('utf-8', errors='ignore')}"}), 500
            report['provider'] = 'yandex'
            return jsonify({'success': True, 'file_url': f'/static/{mp3_filename}', **report})
        else:  # auto
            api_key = os.getenv('YANDEX_API_KEY') or YANDEX_DEFAULT_API_KEY
            folder_id = os.getenv('YANDEX_FOLDER_ID') or YANDEX_DEFAULT_FOLDER_ID
            try:
                speed = float(request.form.get('speed') or 0.85)
            except Exception:
                speed = 0.85
            role = (request.form.get('emotion') or 'neutral').strip().lower()
            if role not in ('neutral', 'good', 'evil'):
                role = 'neutral'
            try:
                if not api_key or not folder_id:
                    raise RuntimeError('Missing Yandex credentials')
                pcm_data = synthesize_yandex(text, voice, api_key, folder_id, speed=speed, role=role)
                unique_id = uuid.uuid4()
                mp3_filename = f"{unique_id}.mp3"
                mp3_filepath = os.path.join(app.config['STATIC_FOLDER'], mp3_filename)
                ffmpeg_bin = os.getenv('FFMPEG_BINARY', 'ffmpeg')
                cmd = [ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y', '-f', 's16le', '-ar', '48000', '-ac', '1', '-i', 'pipe:0', '-codec:a', 'libmp3lame', '-b:a', '192k', mp3_filepath]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout_data, stderr_data = proc.communicate(input=pcm_data)
                if proc.returncode != 0:
                    raise RuntimeError(stderr_data.decode('utf-8', errors='ignore'))
                report['provider'] = 'yandex'
                return jsonify({'success': True, 'file_url': f'/static/{mp3_filename}', **report})
            except Exception as e:
                report['fallback_reason'] = f"yandex_error: {e}"
                try:
                    g_speed = float(request.form.get('speed') or 0.8)
                except Exception:
                    g_speed = 0.8
                try:
                    g_pitch = int(request.form.get('pitch') or 0)
                except Exception:
                    g_pitch = 0
                wav_data = synthesize_google(text, voice, speaking_rate=g_speed, pitch_semitones=g_pitch)
                report['provider'] = 'google'

        if not wav_data:
            return jsonify({'success': False, 'error': 'Synthesis failed and returned no audio data.'}), 500

        unique_id = uuid.uuid4()
        wav_filepath = os.path.join(app.config['STATIC_FOLDER'], f"{unique_id}.wav")
        mp3_filename = f"{unique_id}.mp3"
        mp3_filepath = os.path.join(app.config['STATIC_FOLDER'], mp3_filename)
        with open(wav_filepath, 'wb') as f:
            f.write(wav_data)
        sound = AudioSegment.from_wav(wav_filepath)
        sound.export(mp3_filepath, format="mp3", bitrate="192k")
        os.remove(wav_filepath)
        return jsonify({'success': True, 'file_url': f'/static/{mp3_filename}', **report})

    except Exception as e:
        app.logger.error(f"Synthesis error: {e}")
        return jsonify({'success': False, 'error': f"An internal error occurred: {e}"}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
