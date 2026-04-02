#!/usr/bin/env python3
"""
Offline multilingual voice assistant for Raspberry Pi.

Features
--------
- Microphone capture via sounddevice
- WebRTC VAD utterance segmentation
- Speech-to-text via faster-whisper with automatic language detection
- Simple rule-based dialog manager
- Text-to-speech via Piper
- Playback via ALSA/aplay in MONO mode

Why aplay for TTS?
------------------
On the target Raspberry Pi setup, microphone capture via sounddevice worked
reliably, but TTS playback via sounddevice caused issues with the USB audio
output chain. Playback via ALSA/aplay in mono mode was stable, so this script
uses:
    - sounddevice for microphone input
    - aplay for TTS output

Supported languages
-------------------
- English (en)
- German (de)
- Hindi (hi)

Environment variables
---------------------
Required:
    PIPER_MODEL_PATH
        Directory containing Piper ONNX voices.

Optional:
    INPUT_DEVICE
        sounddevice input device index (default: 0)
    OUTPUT_DEVICE
        sounddevice output device index (default: 1)
    APLAY_DEVICE
        ALSA playback target for aplay (default: "plughw:3,0")
    TTS_VOLUME
        Software gain for TTS audio (default: 0.02)

Example
-------
export PIPER_MODEL_PATH=/home/pi/voices/
export INPUT_DEVICE=0
export OUTPUT_DEVICE=1
export APLAY_DEVICE=plughw:3,0
python assistant.py
"""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from piper import PiperVoice
from scipy.io.wavfile import write as wav_write
from scipy.signal import resample_poly

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

INPUT_DEVICE = int(os.getenv("INPUT_DEVICE", "0"))
OUTPUT_DEVICE = int(os.getenv("OUTPUT_DEVICE", "1"))
APLAY_DEVICE = os.getenv("APLAY_DEVICE", "plughw:3,0")

MIC_SAMPLE_RATE = 48000
ASR_SAMPLE_RATE = 16000
TTS_OUTPUT_RATE = 48000

FRAME_MS = 30
MIC_FRAME_SAMPLES = int(MIC_SAMPLE_RATE * FRAME_MS / 1000)
MIC_FRAME_BYTES = MIC_FRAME_SAMPLES * 2  # int16 mono

CHANNELS = 1
DTYPE = "int16"

VAD_AGGRESSIVENESS = 2
END_SILENCE_MS = 700
MIN_UTTERANCE_MS = 250
MAX_UTTERANCE_SEC = 10.0

WHISPER_MODEL_SIZE = "tiny"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

SUPPORTED_LANGS = {"en", "de", "hi"}
LANG_PROB_THRESHOLD = 0.60
FALLBACK_LANG = "en"

TTS_VOLUME = float(os.getenv("TTS_VOLUME", "0.02"))
TTS_COOLDOWN_SEC = 0.20

PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH")
if not PIPER_MODEL_PATH:
    raise RuntimeError("PIPER_MODEL_PATH is not set.")

VOICE_PATHS = {
    "en": os.path.join(PIPER_MODEL_PATH, "en_US-lessac-medium.onnx"),
    "de": os.path.join(PIPER_MODEL_PATH, "de_DE-ramona-low.onnx"),
    "hi": os.path.join(PIPER_MODEL_PATH, "hi_IN-priyamvada-medium.onnx"),
}

# sounddevice is only used for microphone input
sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)
sd.check_input_settings(
    device=INPUT_DEVICE,
    channels=1,
    dtype=DTYPE,
    samplerate=MIC_SAMPLE_RATE,
)

# ============================================================================
# Dialog manager
# ============================================================================

RESPONSES = {
    "en": {
        "hello": "Hello! Nice to meet you.",
        "how_are_you": "I am doing well, thank you. How can I help?",
        "purpose": "My purpose is to listen and respond with speech.",
        "fallback": "Sorry, I can respond to hello, how are you, and what is your purpose for now.",
    },
    "de": {
        "hello": "Hallo! Schön, dich zu treffen.",
        "how_are_you": "Mir geht es gut, danke! Wie kann ich helfen?",
        "purpose": "Mein Zweck ist es, zuzuhören und mit Sprache zu antworten.",
        "fallback": "Entschuldigung, ich kann im Moment nur auf Hallo, wie geht's und was ist dein Zweck antworten.",
    },
    "hi": {
        "hello": "नमस्ते! आपसे मिलकर खुशी हुई।",
        "how_are_you": "मैं ठीक हूँ, धन्यवाद। मैं आपकी कैसे मदद कर सकता हूँ?",
        "purpose": "मेरा उद्देश्य सुनना और आवाज़ में जवाब देना है।",
        "fallback": "माफ़ कीजिए, अभी मैं सिर्फ नमस्ते, आप कैसे हैं, और आपका उद्देश्य क्या है पर जवाब दे सकता हूँ।",
    },
}


def detect_intent(text: str) -> str:
    t = text.lower().strip()

    if any(x in t for x in [
        "hello", "hi", "hey",
        "hallo", "guten tag", "servus",
        "नमस्ते", "नमस्कार", "हैलो",
        "namaste", "namaskar",
    ]):
        return "hello"

    if any(x in t for x in [
        "how are you", "how're you", "how are u",
        "wie geht", "wie geht's", "wie geht es",
        "कैसे हो", "कैसी हो", "आप कैसे हैं",
        "kaise ho", "kaisi ho", "aap kaise hain",
    ]):
        return "how_are_you"

    if any(x in t for x in [
        "what is your purpose", "what's your purpose", "purpose", "what do you do",
        "was ist dein zweck", "was ist deine aufgabe", "zweck", "aufgabe", "was machst du",
        "उद्देश्य", "मकसद", "तुम क्या करते हो", "आप क्या करते हैं",
        "uddेश्य", "maksud", "tum kya karte ho", "aap kya karte hain",
    ]):
        return "purpose"

    return "fallback"


def respond(text: str, lang: str) -> str:
    language = lang if lang in RESPONSES else FALLBACK_LANG
    intent = detect_intent(text)
    return RESPONSES[language].get(intent, RESPONSES[language]["fallback"])

# ============================================================================
# Utilities
# ============================================================================

def resample_i16(audio_i16: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return audio_i16

    gcd = math.gcd(in_sr, out_sr)
    up = out_sr // gcd
    down = in_sr // gcd

    audio_f32 = audio_i16.astype(np.float32)
    resampled = resample_poly(audio_f32, up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def downsample_i16(audio_i16: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    return resample_i16(audio_i16, in_sr, out_sr)

# ============================================================================
# State
# ============================================================================

class State(Enum):
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


@dataclass
class Utterance:
    audio_f32: np.ndarray

# ============================================================================
# TTS: Piper + aplay
# ============================================================================

class MultiPiperSpeaker:
    """Multilingual Piper TTS speaker using mono WAV playback via aplay."""

    def __init__(self, voice_paths: Dict[str, str], volume: float = 0.02, default_lang: str = "en"):
        self.voices: Dict[str, PiperVoice] = {}
        for lang, path in voice_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Piper voice not found for '{lang}': {path}")
            self.voices[lang] = PiperVoice.load(path)

        self.volume = float(volume)
        self.default_lang = default_lang
        self.output_rate = TTS_OUTPUT_RATE
        self.aplay_device = APLAY_DEVICE

    def _get_voice_sample_rate(self, voice: PiperVoice) -> int:
        cfg = getattr(voice, "config", None)
        rate = getattr(cfg, "sample_rate", None) if cfg is not None else None
        if isinstance(rate, int) and rate > 0:
            return rate

        first = next(voice.synthesize("test"))
        inferred = getattr(first, "sample_rate", None)
        if not isinstance(inferred, int) or inferred <= 0:
            raise RuntimeError("Could not determine Piper voice sample rate.")
        return inferred

    def speak(self, text: str, lang: str, stop_event: Optional[threading.Event] = None) -> None:
        if not text:
            return

        language = lang if lang in self.voices else self.default_lang
        voice = self.voices[language]
        voice_rate = self._get_voice_sample_rate(voice)

        pcm_parts = []
        for chunk in voice.synthesize(text):
            if stop_event is not None and stop_event.is_set():
                return
            pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            if pcm.size > 0:
                pcm_parts.append(pcm)

        if not pcm_parts:
            return

        pcm = np.concatenate(pcm_parts)
        pcm = resample_i16(pcm, voice_rate, self.output_rate)

        if self.volume != 1.0:
            pcm = np.clip(
                pcm.astype(np.float32) * self.volume,
                -32768,
                32767,
            ).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            # IMPORTANT: keep mono, because this was the stable working mode
            wav_write(f.name, self.output_rate, pcm)

            result = subprocess.run(
                ["aplay", "-D", self.aplay_device, f.name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"aplay failed with code {result.returncode}: {result.stderr.strip()}"
                )

    def close(self) -> None:
        pass

# ============================================================================
# Audio segmentation
# ============================================================================

class AudioSegmenter(threading.Thread):
    """Captures microphone audio and segments utterances using WebRTC VAD."""

    def __init__(self, utterance_queue: queue.Queue, mute_event: threading.Event):
        super().__init__(daemon=True)
        self.utterance_queue = utterance_queue
        self.mute_event = mute_event
        self._audio_q: queue.Queue = queue.Queue(maxsize=32)
        self._stop = threading.Event()

        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.in_speech = False
        self.buf = bytearray()
        self.last_voice_ts: Optional[float] = None

        self.overflow_count = 0
        self.dropped_chunks = 0
        self.last_diag_print = time.time()

    def stop(self) -> None:
        self._stop.set()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            if getattr(status, "input_overflow", False) or "overflow" in str(status).lower():
                self.overflow_count += 1

        if self.mute_event.is_set():
            return

        try:
            self._audio_q.put_nowait(bytes(indata))
        except queue.Full:
            self.dropped_chunks += 1

    @staticmethod
    def _frames_from_bytes(data: bytes):
        for i in range(0, len(data), MIC_FRAME_BYTES):
            chunk = data[i:i + MIC_FRAME_BYTES]
            if len(chunk) == MIC_FRAME_BYTES:
                yield chunk

    def _maybe_print_diagnostics(self) -> None:
        now = time.time()
        if now - self.last_diag_print >= 5.0:
            if self.overflow_count > 0 or self.dropped_chunks > 0:
                print(
                    f"[Audio] overflows={self.overflow_count}, "
                    f"dropped_chunks={self.dropped_chunks}, "
                    f"queue_size={self._audio_q.qsize()}"
                )
            self.last_diag_print = now

    def run(self) -> None:
        min_samples = int((MIN_UTTERANCE_MS / 1000) * MIC_SAMPLE_RATE)
        max_samples = int(MAX_UTTERANCE_SEC * MIC_SAMPLE_RATE)

        with sd.RawInputStream(
            samplerate=MIC_SAMPLE_RATE,
            blocksize=MIC_FRAME_SAMPLES * 24,
            dtype=DTYPE,
            channels=1,
            device=INPUT_DEVICE,
            latency="high",
            callback=self._callback,
        ):
            while not self._stop.is_set():
                self._maybe_print_diagnostics()

                try:
                    data = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                now = time.time()

                for frame in self._frames_from_bytes(data):
                    is_voice = self.vad.is_speech(frame, MIC_SAMPLE_RATE)

                    if is_voice:
                        if not self.in_speech:
                            self.in_speech = True
                            self.buf = bytearray()
                        self.buf.extend(frame)
                        self.last_voice_ts = now
                    elif self.in_speech and self.last_voice_ts is not None:
                        if (now - self.last_voice_ts) * 1000 > END_SILENCE_MS:
                            raw = bytes(self.buf)
                            self.in_speech = False
                            self.buf = bytearray()
                            self.last_voice_ts = None

                            audio_i16 = np.frombuffer(raw, dtype=np.int16)

                            if len(audio_i16) < min_samples:
                                continue
                            if len(audio_i16) > max_samples:
                                audio_i16 = audio_i16[:max_samples]

                            audio_i16_16k = downsample_i16(
                                audio_i16,
                                MIC_SAMPLE_RATE,
                                ASR_SAMPLE_RATE,
                            )
                            audio_f32 = audio_i16_16k.astype(np.float32) / 32768.0
                            self.utterance_queue.put(Utterance(audio_f32=audio_f32))

# ============================================================================
# ASR
# ============================================================================

class ASRWorker(threading.Thread):
    """Transcribes utterances with faster-whisper and auto language detection."""

    def __init__(
        self,
        utterance_queue: queue.Queue,
        text_queue: queue.Queue,
        state_lock: threading.Lock,
        state_ref: dict,
    ):
        super().__init__(daemon=True)
        self.utterance_queue = utterance_queue
        self.text_queue = text_queue
        self.state_lock = state_lock
        self.state_ref = state_ref
        self._stop = threading.Event()

        self.model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                utt: Utterance = self.utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.state_lock:
                self.state_ref["state"] = State.PROCESSING

            text, lang, prob = self.transcribe_auto(utt.audio_f32)
            if text:
                self.text_queue.put((text, lang, prob))

            with self.state_lock:
                if self.state_ref["state"] == State.PROCESSING:
                    self.state_ref["state"] = State.LISTENING

    def transcribe_auto(self, audio_f32: np.ndarray) -> Tuple[str, str, float]:
        segments, info = self.model.transcribe(
            audio_f32,
            vad_filter=False,
            beam_size=1,
            temperature=0.0,
        )

        parts = []
        for seg in segments:
            if seg.text:
                parts.append(seg.text.strip())
        text = " ".join(parts).strip()

        lang = getattr(info, "language", None) or FALLBACK_LANG
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)

        if lang not in SUPPORTED_LANGS or prob < LANG_PROB_THRESHOLD:
            lang = FALLBACK_LANG

        return text, lang, prob

# ============================================================================
# TTS worker
# ============================================================================

class TTSWorker(threading.Thread):
    """Speaks responses while muting the mic to avoid feedback."""

    def __init__(
        self,
        response_queue: queue.Queue,
        mute_event: threading.Event,
        state_lock: threading.Lock,
        state_ref: dict,
    ):
        super().__init__(daemon=True)
        self.response_queue = response_queue
        self.mute_event = mute_event
        self.state_lock = state_lock
        self.state_ref = state_ref
        self._stop = threading.Event()
        self._interrupt = threading.Event()

        self.speaker = MultiPiperSpeaker(
            VOICE_PATHS,
            volume=TTS_VOLUME,
            default_lang=FALLBACK_LANG,
        )

    def stop(self) -> None:
        self._stop.set()
        self._interrupt.set()
        self.speaker.close()

    def interrupt(self) -> None:
        self._interrupt.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                text, lang = self.response_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._interrupt.clear()

            with self.state_lock:
                self.state_ref["state"] = State.SPEAKING

            self.mute_event.set()
            try:
                self.speaker.speak(text, lang, stop_event=self._interrupt)
            except Exception as e:
                print(f"[TTS] ERROR: {e}")
            finally:
                time.sleep(TTS_COOLDOWN_SEC)
                self.mute_event.clear()

            with self.state_lock:
                self.state_ref["state"] = State.LISTENING

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Running:", sys.executable)
    print("Python :", sys.version.split()[0])
    print(f"Input device : {INPUT_DEVICE}")
    print(f"Output device: {OUTPUT_DEVICE}")
    print(f"aplay device : {APLAY_DEVICE}")

    state_lock = threading.Lock()
    state_ref = {"state": State.LISTENING}
    mute_event = threading.Event()

    utterance_queue: queue.Queue = queue.Queue(maxsize=8)
    text_queue: queue.Queue = queue.Queue(maxsize=8)
    response_queue: queue.Queue = queue.Queue(maxsize=8)

    segmenter = AudioSegmenter(utterance_queue, mute_event)
    asr = ASRWorker(utterance_queue, text_queue, state_lock, state_ref)
    tts = TTSWorker(response_queue, mute_event, state_lock, state_ref)

    print("\nStarting assistant. Speak English, German, or Hindi.\n")

    segmenter.start()
    asr.start()
    tts.start()

    try:
        while True:
            try:
                user_text, lang, prob = text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            print(f"\nYou ({lang}, p={prob:.2f}): {user_text}")
            bot_text = respond(user_text, lang)
            print(f"Bot ({lang}): {bot_text}")

            response_queue.put((bot_text, lang))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        segmenter.stop()
        asr.stop()
        tts.stop()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
