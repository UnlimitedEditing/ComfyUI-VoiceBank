import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scene

SR = 24000
rng = np.random.default_rng(0)


def speech(seconds, words=3, word_gap=0.12):
    """noise bursts with short inside-turn pauses (speech-like)."""
    out, per = [], (seconds - word_gap * (words - 1)) / words
    for i in range(words):
        out.append(rng.standard_normal(int(per * SR)).astype(np.float32) * 0.2)
        if i < words - 1:
            out.append(np.zeros(int(word_gap * SR), np.float32))
    return np.concatenate(out)


def build(turn_lens, gap):
    parts, truth, t = [], [], 0.0
    for L in turn_lens:
        parts.append(speech(L)); truth.append((t, t + L)); t += L
        parts.append(np.zeros(int(gap * SR), np.float32)); t += gap
    return np.concatenate(parts)[None], truth


# script parsing
assert scene.script_turns("{1}Hi there.{2:old man}Hello.{1}Bye") == [(1, "Hi there."), (2, "Hello."), (1, "Bye")]
assert scene.script_turns("lead{1}x") == [(0, "lead"), (1, "x")]
assert scene.script_turns("a warm man :: Well") == [(0, "Well")]

# turn recovery: 4 turns, each with inside pauses of 0.12 s, turn gaps 0.6 s
audio, truth = build([2.0, 1.5, 3.0, 1.0], 0.6)
spans = scene.find_turns(audio, SR, 4)
assert len(spans) == 4
for (a, b), (ta, tb) in zip(spans, truth):
    assert abs(a - ta) < 0.1 and abs(b - tb) < 0.1, (spans, truth)

# short turn gaps (0.3 s) still resolve
audio2, truth2 = build([2.0, 1.5, 3.0], 0.3)
spans2 = scene.find_turns(audio2, SR, 3)
for (a, b), (ta, tb) in zip(spans2, truth2):
    assert abs(a - ta) < 0.1 and abs(b - tb) < 0.1, (spans2, truth2)

# too few regions -> AlignmentError
try:
    scene.find_turns(audio, SR, 20); raise SystemExit("expected AlignmentError")
except scene.AlignmentError as e:
    print("ok error:", e)

# cues
turns = scene.script_turns("{1}a{2}b{1}c{2}d")
cues = scene.build_cues(turns, spans, audio.shape[1] / SR, "kitchen", "rain")
assert cues["speakers"]["1"]["angle"] == 1 and cues["speakers"]["2"]["angle"] == 2
assert len(cues["cuts"]) == 3 and all(spans[i][1] < c < spans[i + 1][0] for i, c in enumerate(cues["cuts"]))
assert scene.speaker_at(cues, 0.5) == 1 and scene.speaker_at(cues, truth[1][0] + 0.3) == 2
assert scene.next_cut(cues, 0.0, 5.17) == max(c for c in cues["cuts"] if c <= 5.17) == cues["cuts"][1]  # latest cut in window
assert scene.next_cut(cues, 0.0, 0.5) is None
short = scene.build_cues(turns, [(0, 1), (1.1, 2), (2.9, 3.5), (3.6, 4)], 4.0)
assert len(short["cuts"]) == 1 and abs(short["cuts"][0] - 2.45) < 1e-6   # 0.1 s pauses are not cuts

# pixel packing roundtrip
img = scene.cues_to_pixels(cues)
assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[1] == 512
assert scene.pixels_to_cues(img) == cues
# survives the float /255 round trip the IMAGE tensor does
assert scene.pixels_to_cues(np.round((img / 255.0).astype(np.float32) * 255).astype(np.uint8)) == cues

# mix: looped bed is ducked under speech, quiet outside, no clipping
bed = (rng.standard_normal(int(2.0 * SR)).astype(np.float32) * 0.1)[None]
mixed = scene.mix(audio, bed, spans, SR, bed_db=-12, duck_db=-12)
assert mixed.shape == audio.shape and np.abs(mixed).max() <= 10 ** (-1 / 20) + 1e-6
gap_a, gap_b = int((spans[0][1] + 0.1) * SR), int((spans[1][0] - 0.1) * SR)
gap_rms = np.sqrt((mixed[0, gap_a:gap_b] ** 2).mean())
assert gap_rms > 0.005, "bed missing in the pause"
only_bed = scene.mix(np.zeros_like(audio), bed, [], SR, bed_db=-12)
assert np.abs(only_bed).max() > 0.01 and only_bed.shape == audio.shape
# no bed -> dialogue unchanged except peak limit
assert np.allclose(scene.mix(audio, None, spans, SR), audio * min(1, 10 ** (-1 / 20) / np.abs(audio).max()))
print("ALL OK")
