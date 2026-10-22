"""VoiceBankLookup: free-text voice description -> best match in the TTS-AGI moss-character-reference-voices
dataset (1336 synthetic voices, CC-BY-4.0) -> its reference clip as AUDIO. The voice index (voices.json) and the
dataset's own embeddings (emb.npy) are bundled; only the chosen ~60 KB mp3 is fetched at run time.

Matching = BM25 over the dataset's search_text, plus a timbre match (the dataset scores each voice 0-5 for roughness,
brightness, chest voice, ...) and an age match; the optional semantic mode adds embedding similarity."""
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

# query word -> timbre dimensions it asks for (name, +1 wants high / -1 wants low); dataset scores are 0-5
TIMBRE_WORDS = {
    "gravel": [("roughness", 1), ("throat_guttural", 1)], "gravelly": [("roughness", 1), ("throat_guttural", 1)],
    "raspy": [("roughness", 1)], "rough": [("roughness", 1)], "gritty": [("roughness", 1)], "hoarse": [("roughness", 1)],
    "husky": [("roughness", 1)], "growl": [("throat_guttural", 1), ("roughness", 1)],
    "growly": [("throat_guttural", 1), ("roughness", 1)], "guttural": [("throat_guttural", 1)],
    "smooth": [("roughness", -1)], "silky": [("roughness", -1)], "soft": [("roughness", -1)],
    "deep": [("chest_voice", 1)], "low": [("chest_voice", 1)], "booming": [("chest_voice", 1)], "resonant": [("chest_voice", 1)],
    "bright": [("brightness", 1)], "crisp": [("brightness", 1)], "clear": [("brightness", 1)], "sparkling": [("brightness", 1)],
    "dark": [("brightness", -1)], "dull": [("brightness", -1)], "muffled": [("brightness", -1)],
    "metallic": [("metallic", 1)], "robotic": [("metallic", 1)], "synthetic": [("metallic", 1)], "android": [("metallic", 1)],
    "mechanical": [("metallic", 1)], "airy": [("falsetto", 1)], "falsetto": [("falsetto", 1)], "squeaky": [("falsetto", 1)],
    "shrill": [("falsetto", 1)], "high": [("falsetto", 1)],
}
# query word -> target age in years
AGE_WORDS = {"child": 8, "kid": 8, "little": 8, "boy": 10, "girl": 10, "teen": 16, "teenager": 16, "young": 22, "youthful": 22,
             "adult": 32, "mature": 50, "middleaged": 45, "old": 70, "older": 65, "elderly": 72, "aged": 72, "ancient": 80,
             "senior": 68, "grandfather": 70, "grandmother": 70, "elder": 72, "veteran": 58}
WEAK_AGE = {"boy", "girl", "little"}  # only mean a child when no other word gave an age
W_BM25, W_TIMBRE, W_AGE, W_SEM = 1.0, 0.8, 0.8, 1.5
# the dataset's emb.npy was built with this model (its index_info.json), not GTE as its README says
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

_index = None
_embedder = None


def tokens(s):
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("-", "")).split() if t not in STOP]


def age_of(text):
    """Midpoint age in years from strings like 'Late 20s to Mid 30s', '65 to 80 years old', 'Child or Sprite (6-10 ...)'."""
    text = text or ""
    nums = [int(n) for n in re.findall(r"\d+", text) if 3 <= int(n) <= 110]
    if nums:
        return (min(nums) + max(nums)) / 2 + (5 if re.search(r"\d0s", text) and len(nums) == 1 else 0)
    low = text.lower()
    for word, years in (("child", 8), ("teen", 16), ("young", 22), ("elder", 72), ("old", 70)):
        if word in low:
            return years
    return None


def load_index():
    global _index
    if _index is None:
        with open(os.path.join(HERE, "voices.json"), encoding="utf8") as f:
            voices = json.load(f)
        docs = [Counter(tokens(v["search_text"] + " " + " ".join(v.get("tags") or []) + " " + (v.get("age") or ""))) for v in voices]
        df = Counter(t for d in docs for t in d)
        lengths = [sum(d.values()) for d in docs]
        _index = {"voices": voices, "docs": docs, "df": df, "lengths": lengths, "avg": sum(lengths) / len(lengths),
                  "ages": [age_of(v.get("age")) for v in voices]}
    return _index


def embed_query(query):
    """Unit vector for the query with the dataset's embedding model (CLS pooling, as sentence-transformers does for bge).
    Loads models/voicebank/bge-large-en-v1.5 when staged, else downloads EMBED_MODEL from the hub."""
    global _embedder
    from transformers import AutoModel, AutoTokenizer
    if _embedder is None:
        path = EMBED_MODEL
        try:
            import folder_paths
            local = os.path.join(folder_paths.models_dir, "voicebank", "bge-large-en-v1.5")
            if os.path.isfile(os.path.join(local, "config.json")):
                path = local
        except ImportError:
            path = os.environ.get("VOICEBANK_BGE", EMBED_MODEL)
        _embedder = (AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path).eval())
    tok, model = _embedder
    with torch.no_grad():
        out = model(**tok([query], padding=True, truncation=True, max_length=512, return_tensors="pt"))
    return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0].numpy()


def unit(values):
    """Scale a list to 0..1 (all equal -> zeros)."""
    lo, hi = min(values), max(values)
    return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in values]


def rank(query, semantic=False, k1=1.5, b=0.75):
    """[(score, voice)] best first. One detected gender hard-filters; none detected searches everything.
    Score = BM25 + timbre match + age match (+ embedding similarity when `semantic`), each scaled 0..1 and weighted."""
    ix = load_index()
    q = tokens(query)
    qset = set(q)
    want = "male" if qset & MALE and not qset & FEMALE else "female" if qset & FEMALE and not qset & MALE else None
    target_age = next((AGE_WORDS[t] for t in q if t in AGE_WORDS and t not in WEAK_AGE), None)
    if target_age is None:
        target_age = next((AGE_WORDS[t] for t in q if t in AGE_WORDS), None)
    wants = [d for t in q for d in TIMBRE_WORDS.get(t, [])]
    n = len(ix["voices"])
    keep = [i for i, v in enumerate(ix["voices"])
            if not want or (v.get("gender") or "").lower().replace("masculine", "male") == want]
    bm = []
    for i in keep:
        doc, length, score = ix["docs"][i], ix["lengths"][i], 0.0
        for t in q:
            f = doc.get(t, 0)
            if f:
                idf = math.log(1 + (n - ix["df"][t] + 0.5) / (ix["df"][t] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / ix["avg"]))
        bm.append(score)
    timbre = []
    for i in keep:
        profile = ix["voices"][i].get("timbre_profile") or {}
        timbre.append(sum((profile.get(d, 0) if sign > 0 else 5 - profile.get(d, 0)) / 5 for d, sign in wants) / len(wants)
                      if wants else 0.0)
    age = [max(0.0, 1 - abs(ix["ages"][i] - target_age) / 25) if target_age and ix["ages"][i] else 0.0 for i in keep]
    total = [W_BM25 * a + W_TIMBRE * t + W_AGE * g for a, t, g in zip(unit(bm), timbre, age)]
    if semantic:
        sims = (np.load(os.path.join(HERE, "emb.npy"))[keep] @ embed_query(query)).tolist()
        total = [t + W_SEM * x for t, x in zip(total, unit(sims))]
    return [(score, ix["voices"][i]) for score, i in sorted(zip(total, keep), key=lambda p: -p[0])]


def decode_clip(path):
    """mp3 -> (mono float32 waveform [1, samples], sample rate) through PyAV, which ComfyUI already ships."""
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
        }, "optional": {
            "semantic": ("INT", {"default": 0, "min": 0, "max": 1,
                                 "tooltip": "1 adds embedding similarity (bge-large, the model the dataset was embedded with)"}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("reference_audio", "text", "voice_name", "voice_id")
    FUNCTION = "lookup"
    CATEGORY = "audio/voicebank"

    def lookup(self, prompt, pick=0, semantic=0):
        if SEPARATOR not in prompt:
            raise ValueError(f"VoiceBank: write '<voice description> {SEPARATOR} <text to speak>' (no '{SEPARATOR}' found)")
        description, text = (p.strip() for p in prompt.split(SEPARATOR, 1))
        if not description or not text:
            raise ValueError(f"VoiceBank: both the voice description and the text are required around '{SEPARATOR}'")
        ranked = rank(description, semantic=bool(semantic))
        if not ranked:
            raise ValueError("VoiceBank: no voices matched the description")
        score, voice = ranked[min(pick, len(ranked) - 1)]
        import folder_paths
        cache = os.path.join(folder_paths.get_temp_directory(), "voicebank")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, voice["cid"] + ".mp3")
        if not os.path.isfile(path):
            urllib.request.urlretrieve(CLIP_URL.format(cid=voice["cid"]), path)
        wave, rate = decode_clip(path)
        print(f"VoiceBank: '{description}' -> {voice['cid']} '{voice['name']}' ({voice['gender']}, {voice['age']}) "
              f"score {score:.2f}, semantic={bool(semantic)}")
        return ({"waveform": wave[None], "sample_rate": rate}, text, voice["name"], voice["cid"])


NODE_CLASS_MAPPINGS = {"VoiceBankLookup": VoiceBankLookup}
NODE_DISPLAY_NAME_MAPPINGS = {"VoiceBankLookup": "Voice Bank Lookup"}
