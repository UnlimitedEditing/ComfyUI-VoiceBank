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


def _atoms(audio, sr, flags, frame):
    """Cut the voiced stretches at every pause >= 0.06 s. -> (atom [t0, t1] list, gap-before-each-atom list)."""
    regions = _regions(flags, frame, 0.05)
    gaps = [0.0] + [regions[i + 1][0] - regions[i][1] for i in range(len(regions) - 1)]
    return regions, gaps


def _voice_features(audio, sr, regions, bands=24):
    """One feature row per region: log2 median F0 (autocorrelation, 65-500 Hz) + long-term log spectrum (24 bands to 8 kHz).
    Rows are z-scored across regions weighted by nothing; F0 gets extra weight (strongest cue between voices)."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    win, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 500), int(sr / 65)
    edges = np.geomspace(80, min(8000, sr / 2 - 1), bands + 1)
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    feats = []
    for a, b in regions:
        i0, i1 = int(a * sr), int(b * sr)
        f0s, spec, n = [], np.zeros(bands), 0
        for st in range(i0, max(i0 + 1, i1 - win), hop):
            seg = mono[st:st + win]
            if len(seg) < win:
                break
            seg = seg - seg.mean()
            e = float(seg @ seg)
            if e < 1e-6:
                continue
            ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 2048)) ** 2)[:hi + 1]
            if ac[0] > 0 and hi > lo:
                lag = lo + int(np.argmax(ac[lo:hi + 1]))
                if ac[lag] / ac[0] > 0.35:
                    f0s.append(sr / lag)
            ps = np.abs(np.fft.rfft(seg * np.hanning(win), 2048)) ** 2
            for k in range(bands):
                m = (freqs >= edges[k]) & (freqs < edges[k + 1])
                spec[k] += ps[m].sum() if m.any() else 0.0
            n += 1
        spec = np.log10(spec / max(n, 1) + 1e-10)
        spec -= spec.mean()
        f0 = np.log2(np.median(f0s)) if len(f0s) >= 2 else np.nan
        feats.append(np.concatenate([[f0], spec]))
    feats = np.array(feats, np.float64)
    f0col = feats[:, 0]
    if np.isnan(f0col).all():
        f0col[:] = 0.0
    f0col[np.isnan(f0col)] = np.nanmedian(f0col)
    feats[:, 0] = f0col
    sd = feats.std(axis=0)
    sd[sd < 1e-6] = 1.0
    feats = (feats - feats.mean(axis=0)) / sd
    feats[:, 0] *= 4.0     # pitch dominates
    return feats


def find_turns(audio, sr, n_turns, speakers=None, weights=None):
    """Split `audio` [C, N] into exactly n_turns spans. -> [(t0, t1)]

    Pauses alone are unreliable (Drama 3 puts 0.2 s between turns and 0.7 s inside one), so when `speakers` (the speaker id of
    each script turn, in order) is given the voice is used too: speech is cut into atoms at every pause, each atom gets a
    voice feature, speaker prototypes are re-estimated from the alignment (EM, starting from the character-length prior),
    and a monotone DP assigns atoms to turns minimising voice distance + a bonus for starting turns at long pauses +
    a weak penalty for lengths far from `weights` (e.g. characters per turn)."""
    if n_turns < 1:
        raise AlignmentError("the script has no turns")
    flags, frame = _voiced(audio, sr)
    regions, gaps = _atoms(audio, sr, flags, frame)
    m = len(regions)
    if m < n_turns:
        raise AlignmentError(f"found {m} speech regions for {n_turns} script turns")
    dur = np.array([b - a for a, b in regions])
    gaps = np.array(gaps)
    total = regions[-1][1] - regions[0][0]
    w = np.array(weights if weights is not None else [1.0] * n_turns, np.float64)
    w = w / w.sum()
    exp_len = w * total
    use_voice = speakers is not None and len(set(speakers)) >= 2
    feats = _voice_features(audio, sr, regions) if use_voice else None
    mid = np.array([(a + b) / 2 for a, b in regions]) - regions[0][0]
    # initial assignment: character-length prior
    bounds = np.cumsum(exp_len)
    assign = np.minimum(np.searchsorted(bounds, mid), n_turns - 1)
    BONUS = 1.2     # cost saved per second of pause when a turn starts there (capped at 0.8 s)
    for _ in range(4 if use_voice else 1):
        if use_voice:
            protos = {}
            for sp in set(speakers):
                rows = [i for i in range(m) if speakers[assign[i]] == sp]
                protos[sp] = (np.average(feats[rows], axis=0, weights=dur[rows]) if rows
                              else feats.mean(axis=0))
            cost = np.zeros((n_turns, m))
            for k in range(n_turns):
                cost[k] = dur * np.linalg.norm(feats - protos[speakers[k]], axis=1)
        else:
            cost = np.zeros((n_turns, m))
        start_cost = -BONUS * np.minimum(gaps, 0.8)          # cost of starting a new turn at atom i
        inf = 1e18
        dp = np.full((n_turns, m), inf)
        back = np.zeros((n_turns, m), np.int8)               # 0 = continue turn, 1 = new turn here
        dp[0, 0] = cost[0, 0]
        for i in range(1, m):
            for k in range(n_turns):
                stay = dp[k, i - 1]
                adv = dp[k - 1, i - 1] + start_cost[i] if k else inf
                if adv < stay:
                    dp[k, i], back[k, i] = adv + cost[k, i], 1
                else:
                    dp[k, i] = stay + cost[k, i]
        # length prior is applied on the final path only through the EM re-estimate; enforce end state
        k, path = n_turns - 1, np.zeros(m, int)
        if dp[k, m - 1] >= inf:
            raise AlignmentError("could not place all turns")
        for i in range(m - 1, -1, -1):
            path[i] = k
            if back[k, i] == 1:
                k -= 1
        assign = path
    spans = []
    for k in range(n_turns):
        idx = [i for i in range(m) if assign[i] == k]
        spans.append((round(regions[idx[0]][0], 3), round(regions[idx[-1]][1], 3)))
    return spans


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
