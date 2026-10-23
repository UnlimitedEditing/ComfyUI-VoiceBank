"""AudioEcho: returns the audio Graydient hands over, unchanged, so the render has a durable link and a render hash
(reply to a Telegram voice note with /run:archive-submit-audio, then use /audio2:R<hash> in other workflows).
Takes the whole init_audio triplet plus audio1; the first non-empty one wins. Staged filenames and http(s) URLs work."""
import os
import re
import urllib.request

import numpy as np
import torch


def audio_from(source):
    import av
    import folder_paths
    source = source.strip()
    if source.startswith(("http://", "https://")):
        cache = os.path.join(folder_paths.get_temp_directory(), "audioecho")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, re.sub(r"[^A-Za-z0-9_.-]", "_", source.split("?")[0])[-80:] or "audio")
        urllib.request.urlretrieve(source, path)
    else:
        path = folder_paths.get_annotated_filepath(source)
    if not os.path.isfile(path):
        raise ValueError(f"AudioEcho: audio '{source}' not found")
    with av.open(path) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        resampler = av.AudioResampler(format="fltp", rate=rate)
        parts = [o.to_ndarray() for frame in container.decode(stream) for o in resampler.resample(frame)]
    return {"waveform": torch.from_numpy(np.concatenate(parts, axis=-1)).float()[None], "sample_rate": rate}


class VoiceBankAudioEcho:
    @classmethod
    def INPUT_TYPES(cls):
        text = ("STRING", {"default": ""})
        return {"required": {}, "optional": {"init_audio": text, "init_audio_url": text, "init_audio_filename": text, "audio1": text}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "echo"
    CATEGORY = "audio/voicebank"

    def echo(self, init_audio="", init_audio_url="", init_audio_filename="", audio1=""):
        for source in (audio1, init_audio_filename, init_audio_url, init_audio):
            if source and source.strip():
                return (audio_from(source),)
        raise ValueError("AudioEcho: no audio received -- reply to a voice note or attach an audio file")


def video_path(source):
    """Local path for a staged filename or an http(s) URL."""
    import folder_paths
    source = source.strip()
    if source.startswith(("http://", "https://")):
        cache = os.path.join(folder_paths.get_temp_directory(), "archiveecho")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, re.sub(r"[^A-Za-z0-9_.-]", "_", source.split("?")[0])[-80:] or "video")
        urllib.request.urlretrieve(source, path)
        return path
    path = folder_paths.get_annotated_filepath(source)
    if not os.path.isfile(path):
        raise ValueError(f"ArchiveEcho: video '{source}' not found")
    return path


class VoiceBankArchiveEcho:
    """One workflow for video AND audio: whichever kind arrives is returned unchanged on its own output; the other
    output is a silent ExecutionBlocker, so the Save node wired to it simply does not run. Video wins if both arrive."""

    @classmethod
    def INPUT_TYPES(cls):
        text = ("STRING", {"default": ""})
        names = ["init_video", "init_video_url", "init_video_filename", "video1",
                 "init_audio", "init_audio_url", "init_audio_filename", "audio1"]
        return {"required": {}, "optional": {name: text for name in names}}

    RETURN_TYPES = ("VIDEO", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "echo"
    CATEGORY = "audio/voicebank"

    def echo(self, init_video="", init_video_url="", init_video_filename="", video1="",
             init_audio="", init_audio_url="", init_audio_filename="", audio1=""):
        from comfy_execution.graph_utils import ExecutionBlocker
        video = next((v.strip() for v in (video1, init_video_filename, init_video_url, init_video) if v and v.strip()), "")
        if video:
            from comfy_api.latest import InputImpl
            return (InputImpl.VideoFromFile(video_path(video)), ExecutionBlocker(None))
        audio = next((a.strip() for a in (audio1, init_audio_filename, init_audio_url, init_audio) if a and a.strip()), "")
        if audio:
            return (ExecutionBlocker(None), audio_from(audio))
        raise ValueError("ArchiveEcho: nothing received -- reply to a video, voice note or audio file, or attach one")


NODE_CLASS_MAPPINGS = {"VoiceBankAudioEcho": VoiceBankAudioEcho, "VoiceBankArchiveEcho": VoiceBankArchiveEcho}
NODE_DISPLAY_NAME_MAPPINGS = {"VoiceBankAudioEcho": "Audio Echo (voice note -> link)",
                              "VoiceBankArchiveEcho": "Archive Echo (video or audio -> link)"}
