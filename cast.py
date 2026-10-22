"""VoiceCast: one node that turns a script with speaker tags into Drama 3 reference voices.

  {1}Hello there.                     speaker 1, no reference voice (the model invents one)
  {1:a warm gravelly old man}Hello    speaker 1 voiced by the best match in the voice bank
  audio1 / audio2 ... audio8          clone speaker 1 / 2 ... 8 from the user's own audio file (wins over a description)
  desc :: text                        shorthand for {0:desc} text

It converts the script to Drama 3's syntax ({N} -> <|speaker:N|>, (sighs) -> [sighs]), picks/encodes the voices and
hands them to Drama3Generate as DRAMA3_VOICE outputs (None for unused slots; Generate skips those).

Drama 3 and its Qwen3-ASR are used through the ComfyUI-Drama3 pack's own python functions, NOT through its node
classes: Graydient's base image ships Comfyui_SynVow_Qwen3ASR, which registers the same Qwen3ASRLoader /
Qwen3ASRTranscribe class names and can shadow the pack's nodes depending on the runner."""
import os
import re
import sys
import urllib.request
from collections import OrderedDict

import numpy as np
import torch

from .nodes import CLIP_URL, SEPARATOR, decode_clip, rank

SPEAKER = re.compile(r"\{(\d{1,2})(?::([^}]*))?\}")
MAX_VOICES = 8
ASR_FILE = "qwen3_asr_0.6b_bf16.safetensors"


def parse_script(prompt):
    """-> (text for Drama 3, speakers in order of first appearance, {speaker: description})."""
    if not SPEAKER.search(prompt) and SEPARATOR in prompt:
        description, text = (p.strip() for p in prompt.split(SEPARATOR, 1))
        prompt = "{0:%s}%s" % (description, text)
    order, descriptions = [], {}
    for m in SPEAKER.finditer(prompt):
        n = int(m.group(1))
        if n not in order:
            order.append(n)
        if m.group(2) and m.group(2).strip():
            descriptions.setdefault(n, m.group(2).strip())
    first = SPEAKER.search(prompt)
    if (prompt[:first.start()] if first else prompt).strip() and 0 not in order:
        order.insert(0, 0)  # untagged lead-in is speaker 0
    text = SPEAKER.sub(lambda m: "<|speaker:%d|>" % int(m.group(1)), prompt)
    text = text.replace("(", "[").replace(")", "]").strip()
    return text, order, descriptions


def pick_bank_voices(descriptions, pick=0, semantic=False):
    """{speaker: voice record}; every speaker gets a different voice, `pick` skips that many better matches."""
    chosen, used = {}, set()
    for speaker, description in descriptions.items():
        ranked = [v for _, v in rank(description, semantic=semantic) if v["cid"] not in used]
        if not ranked:
            raise ValueError(f"VoiceCast: no voice left for speaker {speaker} ('{description}')")
        chosen[speaker] = ranked[min(pick, len(ranked) - 1)]
        used.add(chosen[speaker]["cid"])
    return chosen


def drama3_pack():
    """The loaded ComfyUI-Drama3 nodes module (its python functions, not its node classes)."""
    for module in list(sys.modules.values()):
        ns = getattr(module, "__dict__", {})
        if "Drama3EncodeVoice" in ns and "load_qwen3_asr" in ns and "transcribe" in ns:
            return module
    raise RuntimeError("VoiceCast: ComfyUI-Drama3 is not loaded (add https://github.com/PedroMarinhoDev/ComfyUI-Drama3)")


def load_user_audio(name):
    """Audio file by name in ComfyUI's input folder, or by http(s) URL."""
    import folder_paths
    if name.startswith(("http://", "https://")):
        cache = os.path.join(folder_paths.get_temp_directory(), "voicecast")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, re.sub(r"[^A-Za-z0-9_.-]", "_", name.split("?")[0])[-80:] or "audio")
        urllib.request.urlretrieve(name, path)
    else:
        path = folder_paths.get_annotated_filepath(name)
    if not os.path.isfile(path):
        raise ValueError(f"VoiceCast: audio file '{name}' not found")
    import av
    with av.open(path) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        parts = [o.to_ndarray() for frame in container.decode(stream) for o in resampler.resample(frame)]
    return torch.from_numpy(np.concatenate(parts, axis=-1)).float(), rate


def bank_clip(voice):
    import folder_paths
    cache = os.path.join(folder_paths.get_temp_directory(), "voicebank")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, voice["cid"] + ".mp3")
    if not os.path.isfile(path):
        urllib.request.urlretrieve(CLIP_URL.format(cid=voice["cid"]), path)
    return decode_clip(path)


class VoiceCast:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"audio%d" % i: ("STRING", {"default": "", "tooltip": "Reference audio file for speaker %d" % i})
                    for i in range(1, MAX_VOICES + 1)}
        optional["init_audio"] = ("STRING", {"default": "", "tooltip": "Telegram: audio you replied to (any form); speaker 1 when audio1 is empty"})
        optional["init_audio_filename"] = ("STRING", {"default": "", "tooltip": "Telegram: audio you replied to; used as speaker 1 when audio1 is empty"})
        optional["init_audio_url"] = ("STRING", {"default": "", "tooltip": "Same, URL form; used when audio1 and init_audio_filename are empty"})
        optional["asr_model"] = ("STRING", {"default": ASR_FILE, "tooltip": "models/asr file used to transcribe your audio"})
        return {"required": {
            "codec": ("DRAMA3_CODEC",),
            "prompt": ("STRING", {"multiline": True, "default": "{1:a warm gravelly old man}Well, here we are."}),
            "pick": ("INT", {"default": 0, "min": 0, "max": 50, "tooltip": "0 = best bank match, 1 = runner-up ..."}),
            "semantic": ("INT", {"default": 0, "min": 0, "max": 1}),
        }, "optional": optional}

    RETURN_TYPES = ("STRING",) + ("DRAMA3_VOICE",) * MAX_VOICES
    RETURN_NAMES = ("text",) + tuple("voice%d" % i for i in range(1, MAX_VOICES + 1))
    FUNCTION = "cast"
    CATEGORY = "audio/voicebank"

    def cast(self, codec, prompt, pick=0, semantic=0, asr_model=ASR_FILE, init_audio_filename="", init_audio_url="",
             init_audio="", **audio_files):
        text, speakers, descriptions = parse_script(prompt)
        if not text:
            raise ValueError("VoiceCast: the prompt has no text to speak")
        clones = {i: audio_files.get("audio%d" % i, "").strip() for i in range(1, MAX_VOICES + 1)}
        clones = {i: name for i, name in clones.items() if name}
        replied = next((x.strip() for x in (init_audio_filename, init_audio_url, init_audio) if x and x.strip()), "")
        if 1 not in clones and replied:
            clones[1] = replied  # replied-to audio is speaker 1
        # a speaker with an audio file keeps it; a description only matters for speakers without one
        wanted = {s: d for s, d in descriptions.items() if s not in clones}
        bank = pick_bank_voices(wanted, pick, bool(semantic))
        jobs = [(s, "audio", clones[s]) for s in speakers if s in clones]
        jobs += [(s, "bank", bank[s]) for s in speakers if s in bank]
        jobs += [(i, "audio", name) for i, name in clones.items() if i not in speakers]  # audioN given, never tagged
        if len(jobs) > MAX_VOICES:
            raise ValueError(f"VoiceCast: at most {MAX_VOICES} voices, got {len(jobs)}")
        pack, asr, voices = drama3_pack(), None, []
        for speaker, kind, source in jobs:
            if kind == "audio":
                wave, rate = load_user_audio(source)
                transcript, label = None, f"audio{speaker} ({source})"
            else:
                wave, rate = bank_clip(source)
                transcript, label = source.get("transcript"), f"{source['name']} ({source['cid']})"
            if not transcript:
                if asr is None:
                    import folder_paths
                    asr = pack.load_qwen3_asr(folder_paths.get_full_path_or_raise("asr", asr_model))
                transcript, _ = pack.transcribe(asr, wave, rate, None, "", 4096)
            audio = {"waveform": wave[None], "sample_rate": rate}
            out = pack.Drama3EncodeVoice.execute(codec, audio, transcript)
            encoded = (getattr(out, "result", None) or out.args)[0]
            voices.append({"transcript": "<|speaker:%d|> %s" % (speaker, transcript), "codes": encoded["codes"]})
            print(f"VoiceCast: speaker {speaker} <- {label}: {transcript[:80]!r}")
        voices += [None] * (MAX_VOICES - len(voices))
        return (text, *voices)


NODE_CLASS_MAPPINGS = {"VoiceCast": VoiceCast}
NODE_DISPLAY_NAME_MAPPINGS = {"VoiceCast": "Voice Cast (Drama 3)"}
