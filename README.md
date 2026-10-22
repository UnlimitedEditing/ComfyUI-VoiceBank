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
