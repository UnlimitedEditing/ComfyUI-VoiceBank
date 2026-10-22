# ComfyUI-VoiceBank

`Voice Bank Lookup`: a free-text voice description picks a reference clip from the
[TTS-AGI/moss-character-reference-voices](https://huggingface.co/datasets/TTS-AGI/moss-character-reference-voices)
dataset (1336 synthetic voices, CC-BY-4.0, annotations by Gemini-3-Flash). Input `"<voice description> :: <text to speak>"`;
outputs the clip (AUDIO), the text, the voice name and id. `pick` selects runner-ups (0 = best).

Matching is BM25 over the dataset's `search_text` with a hard gender filter when the description names one. No pip
dependencies; the index (`voices.json`) is bundled and only the chosen clip is downloaded at run time.
Built to feed a reference-voice cloner (e.g. ComfyUI-Drama3) so users never need a clip URL. The dataset has no
transcripts: pair it with an ASR node when the cloner needs the clip's words.
