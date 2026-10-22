"""VoiceBankLookup: free-text voice description -> best match in the TTS-AGI moss-character-reference-voices
dataset (1336 synthetic voices, CC-BY-4.0) -> its reference clip as AUDIO. No pip deps: BM25 is inline and the
voice index (voices.json) is bundled; only the chosen ~60 KB mp3 is fetched at run time."""
import json
import math
import os
import re
import urllib.request
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_URL = "https://huggingface.co/datasets/TTS-AGI/moss-character-reference-voices/resolve/main/audio/{cid}.mp3"
SEPARATOR = "::"

MALE = {"man", "men", "male", "boy", "gentleman", "he", "his", "father", "grandfather", "king", "guy", "dude", "masculine"}
FEMALE = {"woman", "women", "female", "girl", "lady", "she", "her", "mother", "grandmother", "queen", "feminine"}
STOP = {"a", "an", "the", "and", "of", "with", "in", "to", "is", "it", "very", "like", "sounds", "voice", "who", "that"}

_index = None


def tokens(s):
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if t not in STOP]


def load_index():
    global _index
    if _index is None:
        with open(os.path.join(HERE, "voices.json"), encoding="utf8") as f:
            voices = json.load(f)
        docs = [Counter(tokens(v["search_text"] + " " + " ".join(v.get("tags") or []) + " " + (v.get("age") or ""))) for v in voices]
        df = Counter(t for d in docs for t in d)
        lengths = [sum(d.values()) for d in docs]
        _index = {"voices": voices, "docs": docs, "df": df, "lengths": lengths, "avg": sum(lengths) / len(lengths)}
    return _index


def rank(query, k1=1.5, b=0.75):
    """[(score, voice)] best first. One detected gender hard-filters; none detected searches everything."""
    ix = load_index()
    q = tokens(query)
    qset = set(q)
    want = "Male" if qset & MALE and not qset & FEMALE else "Female" if qset & FEMALE and not qset & MALE else None
    n = len(ix["voices"])
    out = []
    for voice, doc, length in zip(ix["voices"], ix["docs"], ix["lengths"]):
        if want and (voice.get("gender") or "").lower().replace("masculine", "male") != want.lower():
            continue
        score = 0.0
        for t in q:
            f = doc.get(t, 0)
            if f:
                idf = math.log(1 + (n - ix["df"][t] + 0.5) / (ix["df"][t] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / ix["avg"]))
        out.append((score, voice))
    out.sort(key=lambda p: -p[0])
    return out


def decode_mp3(path):
    import av
    with av.open(path) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        chunks = [frame.to_ndarray().astype(np.float32) for frame in container.decode(stream)]
    wave = np.concatenate(chunks, axis=-1)
    if np.issubdtype(chunks[0].dtype, np.integer):
        wave = wave / 32768.0
    return torch.from_numpy(wave).float(), rate


def decode_frames(path):
    # planar float (fltp) from PyAV: [channels, samples]; fall back to a packed layout if PyAV hands that back
    import av
    with av.open(path) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        parts = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray())
        for out in resampler.resample(None) or []:
            parts.append(out.to_ndarray())
    return torch.from_numpy(np.concatenate(parts, axis=-1)).float(), rate


class VoiceBankLookup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "a warm gravelly old man :: Well, here we are."}),
            "pick": ("INT", {"default": 0, "min": 0, "max": 50, "tooltip": "0 = best match, 1 = runner-up ..."}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("reference_audio", "text", "voice_name", "voice_id")
    FUNCTION = "lookup"
    CATEGORY = "audio/voicebank"

    def lookup(self, prompt, pick=0):
        if SEPARATOR not in prompt:
            raise ValueError(f"VoiceBank: write '<voice description> {SEPARATOR} <text to speak>' (no '{SEPARATOR}' found)")
        description, text = (p.strip() for p in prompt.split(SEPARATOR, 1))
        if not description or not text:
            raise ValueError(f"VoiceBank: both the voice description and the text are required around '{SEPARATOR}'")
        ranked = rank(description)
        if not ranked:
            raise ValueError("VoiceBank: no voices matched the description")
        score, voice = ranked[min(pick, len(ranked) - 1)]
        import folder_paths
        cache = os.path.join(folder_paths.get_temp_directory(), "voicebank")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, voice["cid"] + ".mp3")
        if not os.path.isfile(path):
            urllib.request.urlretrieve(CLIP_URL.format(cid=voice["cid"]), path)
        wave, rate = decode_frames(path)
        print(f"VoiceBank: '{description}' -> {voice['cid']} '{voice['name']}' ({voice['gender']}, {voice['age']}) score {score:.2f}")
        return ({"waveform": wave[None], "sample_rate": rate}, text, voice["name"], voice["cid"])


NODE_CLASS_MAPPINGS = {"VoiceBankLookup": VoiceBankLookup}
NODE_DISPLAY_NAME_MAPPINGS = {"VoiceBankLookup": "Voice Bank Lookup"}
