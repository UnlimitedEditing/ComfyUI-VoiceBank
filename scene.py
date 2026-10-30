"""SceneAudio core: pure numpy (no torch / ComfyUI imports) so it can be tested on its own.

Takes the ONE mixed dialogue track Drama 3 produces plus the script it was made from, and returns:
  turns   [{t0, t1, speaker}]  recovered by voice-activity detection, forced to the script's turn count
  cuts    [seconds]            midpoints of the pauses between turns, where a video chunk may end
  mix     dialogue + looped ambience bed ducked under speech
  cue sheet JSON (tag sceneaudio/1), and a pixel-data packing of it (4-byte BE length + UTF-8, 3 bytes per pixel)
"""
import json
import re

import numpy as np

TAG = "sceneaudio/1"
SPEAKER = re.compile(r"\{(\d{1,2})(?::([^}]*))?\}")
SEPARATOR = "::"


class AlignmentError(RuntimeError):
    """The audio does not split into the script's number of turns; fall back to one Drama 3 call per turn."""


def script_turns(prompt):
    """-> [(speaker, text)] in order. `{1}text`, `{1:desc}text`; `desc :: text` is one turn of speaker 0.
    Text before the first tag is speaker 0."""
    matches = list(SPEAKER.finditer(prompt))
    if not matches:
        text = prompt.split(SEPARATOR, 1)[1] if SEPARATOR in prompt else prompt
        return [(0, text.strip())] if text.strip() else []
    turns = []
    lead = prompt[:matches[0].start()].strip()
    if lead:
        turns.append((0, lead))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        text = prompt[m.end():end].strip()
        if text:
            turns.append((int(m.group(1)), text))
    return turns


def _voiced(audio, sr, frame=0.02, rel_thresh=0.04):
    """Boolean per `frame`: RMS above rel_thresh of the 95th-percentile RMS (adapts to loudness)."""
    mono = np.abs(audio).mean(axis=0) if audio.ndim == 2 else np.abs(audio)
    n = int(sr * frame)
    cnt = len(mono) // n
    if cnt == 0:
        return np.zeros(0, bool), frame
    rms = np.sqrt((mono[:cnt * n].reshape(cnt, n) ** 2).mean(axis=1))
    ref = np.percentile(rms, 95)
    return (rms > max(ref * rel_thresh, 1e-5)), frame


def _regions(flags, frame, min_gap):
    """Runs of True, merging runs closer than min_gap seconds. -> [[t0, t1]]"""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            t0, t1 = start * frame, i * frame
            if out and t0 - out[-1][1] < min_gap:
                out[-1][1] = t1
            else:
                out.append([t0, t1])
            start = None
    return out


def find_turns(audio, sr, n_turns):
    """Split `audio` [C, N] into exactly n_turns spans. Tries finer pause thresholds until there are at least
    n_turns regions, then merges across the smallest gaps down to n_turns. -> [(t0, t1)]"""
    if n_turns < 1:
        raise AlignmentError("the script has no turns")
    flags, frame = _voiced(audio, sr)
    regions = []
    for min_gap in (0.30, 0.20, 0.12, 0.08, 0.05):
        regions = _regions(flags, frame, min_gap)
        if len(regions) >= n_turns:
            break
    if len(regions) < n_turns:
        raise AlignmentError(f"found {len(regions)} speech regions for {n_turns} script turns")
    while len(regions) > n_turns:
        gaps = [regions[i + 1][0] - regions[i][1] for i in range(len(regions) - 1)]
        i = int(np.argmin(gaps))
        regions[i][1] = regions[i + 1][1]
        del regions[i + 1]
    return [(round(a, 3), round(b, 3)) for a, b in regions]


def build_cues(turns, spans, total, scene_prompt="", ambience="", voices=None, min_pause=0.25):
    """Cue sheet dict. turns = script_turns(); spans = find_turns(). `voices` {speaker: voice id} is informational."""
    order = []
    for s, _ in turns:
        if s not in order:
            order.append(s)
    speakers = {str(s): {"angle": min(i + 1, 8), "voice": (voices or {}).get(s, "")} for i, s in enumerate(order)}
    tl = [{"t0": a, "t1": b, "speaker": s} for (s, _), (a, b) in zip(turns, spans)]
    cuts = [round((tl[i]["t1"] + tl[i + 1]["t0"]) / 2, 3) for i in range(len(tl) - 1)
            if tl[i + 1]["t0"] - tl[i]["t1"] >= min_pause]
    return {"tag": TAG, "total": round(float(total), 3), "speakers": speakers, "turns": tl,
            "scenes": [{"t0": 0.0, "t1": round(float(total), 3), "prompt": scene_prompt, "ambience": ambience}],
            "cuts": cuts}


def speaker_at(cues, t):
    """Speaker of the turn covering t, else the nearest preceding turn, else the first turn. None if no turns."""
    turns = cues.get("turns") or []
    if not turns:
        return None
    best = turns[0]
    for tr in turns:
        if tr["t0"] <= t:
            best = tr
    return best["speaker"]


def next_cut(cues, cursor, max_seconds, min_seconds=0.9):
    """Latest allowed cut in (cursor+min_seconds, cursor+max_seconds], or None (caller falls back to quietest-of-3)."""
    ok = [c for c in cues.get("cuts", []) if cursor + min_seconds < c <= cursor + max_seconds]
    return max(ok) if ok else None


def loop_bed(bed, n, sr, xfade=1.0):
    """Tile bed [C, M] to n samples with equal-power crossfades at the joins."""
    c, m = bed.shape
    if m >= n:
        return bed[:, :n].copy()
    x = min(int(xfade * sr), m // 2)
    out = bed.copy()
    t = np.linspace(0, np.pi / 2, x, dtype=np.float32)
    fin, fout = np.sin(t), np.cos(t)
    while out.shape[1] < n:
        head = bed[:, :x] * fin
        out[:, -x:] = out[:, -x:] * fout + head
        out = np.concatenate([out, bed[:, x:]], axis=1)
    return out[:, :n]


def mix(dialogue, bed, spans, sr, bed_db=-18.0, duck_db=-6.0, ramp=0.15):
    """dialogue [C, N] + bed (any [C, M] or None). Ambience is looped to length and ducked by duck_db while a turn
    speaks (smoothed by `ramp` seconds). Peak-limited to -1 dBFS. -> [C, N] float32"""
    c, n = dialogue.shape
    out = dialogue.astype(np.float32).copy()
    if bed is not None and bed.shape[1] > 0:
        if bed.shape[0] != c:
            bed = np.repeat(bed[:1], c, axis=0) if bed.shape[0] == 1 else bed[:c]
        wet = loop_bed(bed.astype(np.float32), n, sr)
        env = np.ones(n, np.float32)
        duck = 10 ** (duck_db / 20)
        for a, b in spans:
            env[int(a * sr):int(b * sr)] = duck
        k = max(1, int(ramp * sr))
        env = np.convolve(env, np.ones(k, np.float32) / k, mode="same")
        out += wet * env * 10 ** (bed_db / 20)
    peak = float(np.abs(out).max())
    limit = 10 ** (-1 / 20)
    if peak > limit:
        out *= limit / peak
    return out


def pack_pixels(data, width=512):
    """bytes -> uint8 [H, W, 3] (4-byte big-endian length, payload, zero pad)."""
    buf = len(data).to_bytes(4, "big") + data
    rows = -(-len(buf) // (width * 3))
    arr = np.zeros(rows * width * 3, np.uint8)
    arr[:len(buf)] = np.frombuffer(buf, np.uint8)
    return arr.reshape(rows, width, 3)


def unpack_pixels(img):
    raw = np.ascontiguousarray(img, np.uint8).reshape(-1).tobytes()
    n = int.from_bytes(raw[:4], "big")
    return raw[4:4 + n]


def cues_to_pixels(cues):
    return pack_pixels(json.dumps(cues, separators=(",", ":")).encode("utf-8"))


def pixels_to_cues(img):
    return json.loads(unpack_pixels(img).decode("utf-8"))
