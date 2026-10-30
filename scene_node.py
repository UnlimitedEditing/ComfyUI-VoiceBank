"""SceneAudioMix: Drama 3 dialogue + script (+ optional ambience bed) -> mix, dry dialogue, cue sheet.
The logic lives in scene.py (pure numpy, tested without ComfyUI). See D:\\tripostl\\SCENE-AUDIO-SPEC.md."""
import json

import numpy as np
import torch

from . import scene


def _wave(audio):
    w = audio["waveform"][0].detach().cpu().float().numpy()
    return w, int(audio["sample_rate"])


def _resample(w, src, dst):
    if src == dst or w.shape[1] == 0:
        return w
    n = int(round(w.shape[1] * dst / src))
    x = np.linspace(0, w.shape[1] - 1, n)
    return np.stack([np.interp(x, np.arange(w.shape[1]), ch) for ch in w]).astype(np.float32)


class SceneAudioMix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dialogue": ("AUDIO",),
            "script": ("STRING", {"multiline": True, "default": "{1}Well, here we are.{2}Yes. At last."}),
            "bed_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": 0.0, "step": 0.5}),
            "duck_db": ("FLOAT", {"default": -6.0, "min": -30.0, "max": 0.0, "step": 0.5,
                                  "tooltip": "extra ambience reduction while someone speaks"}),
            "min_pause": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 2.0, "step": 0.05,
                                    "tooltip": "pauses shorter than this are not offered as video cut points"}),
        }, "optional": {
            "ambience": ("AUDIO",),
            "scene_prompt": ("STRING", {"default": "", "multiline": True}),
            "ambience_label": ("STRING", {"default": ""}),
            "voices_json": ("STRING", {"default": "", "tooltip": 'optional {"1": "voice id"} recorded in the cue sheet'}),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "IMAGE")
    RETURN_NAMES = ("mix", "dialogue", "cues", "cues_image")
    FUNCTION = "run"
    CATEGORY = "audio/voicebank"

    def run(self, dialogue, script, bed_db=-18.0, duck_db=-6.0, min_pause=0.25, ambience=None,
            scene_prompt="", ambience_label="", voices_json=""):
        wave, sr = _wave(dialogue)
        turns = scene.script_turns(script)
        spans = scene.find_turns(wave, sr, len(turns))
        voices = {int(k): v for k, v in json.loads(voices_json).items()} if voices_json.strip() else None
        cues = scene.build_cues(turns, spans, wave.shape[1] / sr, scene_prompt, ambience_label, voices, min_pause)
        bed = None
        if ambience is not None:
            bed, bsr = _wave(ambience)
            bed = _resample(bed, bsr, sr)
        mixed = scene.mix(wave, bed, spans, sr, bed_db, duck_db)
        img = torch.from_numpy(scene.cues_to_pixels(cues).astype(np.float32) / 255.0)[None]
        pack = lambda a: {"waveform": torch.from_numpy(np.ascontiguousarray(a))[None], "sample_rate": sr}
        print(f"SceneAudioMix: {len(turns)} turns, {len(cues['cuts'])} cuts, {cues['total']:.1f} s")
        return (pack(mixed), {"waveform": dialogue["waveform"], "sample_rate": sr},
                json.dumps(cues, separators=(",", ":")), img)


NODE_CLASS_MAPPINGS = {"SceneAudioMix": SceneAudioMix}
NODE_DISPLAY_NAME_MAPPINGS = {"SceneAudioMix": "Scene Audio Mix"}
