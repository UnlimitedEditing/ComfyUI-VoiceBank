# ComfyUI-VoiceBank

`Voice Bank Lookup`: a free-text voice description picks a reference clip from the
[TTS-AGI/moss-character-reference-voices](https://huggingface.co/datasets/TTS-AGI/moss-character-reference-voices)
dataset (1336 synthetic voices, CC-BY-4.0, annotations by Gemini-3-Flash). Input `"<voice description> :: <text to speak>"`;
outputs the clip (AUDIO), the text, the voice name and id. `pick` selects runner-ups (0 = best).

Matching is BM25 over the dataset's `search_text` with a hard gender filter when the description names one. No pip
dependencies; the index (`voices.json`) is bundled and only the chosen clip is downloaded at run time.
Built to feed a reference-voice cloner (e.g. ComfyUI-Drama3) so users never need a clip URL. The dataset has no
transcripts: pair it with an ASR node when the cloner needs the clip's words.

## Matching
BM25 over `search_text`, plus a timbre match (the dataset's 0-5 scores for roughness, brightness, chest voice, metallic,
falsetto, throat) and an age match, with a hard gender filter when the description names one. `semantic=1` adds cosine
similarity against the bundled `emb.npy`; those were built with `BAAI/bge-large-en-v1.5` (see the dataset's
`index_info.json`; its README says GTE, which is wrong). The model is loaded from `models/voicebank/bge-large-en-v1.5`
when present, else downloaded from the hub (about 1.3 GB, slow on a cold machine), so keep it off unless staged.

## Voice Cast (Drama 3)
`Voice Cast` parses a script with speaker tags and builds reference voices for `ComfyUI-Drama3`'s Generate node:
`{1:a warm gravelly old man}Hello. {2:a bright cheerful girl}Hi! {1}Hm.`  Each described speaker gets a different
bank voice (`pick` skips better matches). `audio1`...`audio8` clone speakers 1...8 from your own files instead
(transcribed with the pack's Qwen3-ASR); `init_audio_filename`/`init_audio_url` (a replied-to Telegram audio) is speaker 1.
`(sighs)` becomes `[sighs]` and `{N}` becomes `<|speaker:N|>` because Graydient's prompt parser eats `[ ] < > |`.
It calls ComfyUI-Drama3's python functions, not its Qwen3-ASR node classes, which Graydient's own Qwen3-ASR pack can shadow.
Bank transcripts (`voices.json`) are Whisper large-v3 output; the dataset itself has none.
