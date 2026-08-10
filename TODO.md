# TODO

## Xiaomi ASR integration (parked — pending testing)

- [ ] Integrate Xiaomi ASR as an alternative transcription backend
  - Status: parked here for testing next; API details still to be figured out
  - Docs: https://mimo.mi.com/models/en-US/mimo-v2.5-asr (MiMo v2.5 ASR)
  - API key env var: `XIAOMI_ASR_API_KEY`
  - Next steps:
    - [x] Figure out the API access (OpenAI-compatible chat completions at
          https://api.xiaomimimo.com/v1, base64 `input_audio`, key in
          `XIAOMI_ASR_API_KEY`/`MIMO_API_KEY`) — smoke-tested OK 2026-07-07
    - [x] Shadow testing: every recording also produces `<base>_by_MiMoASR.txt`
          next to the normal transcript for side-by-side diffing
    - [ ] Compare bilingual (EN/中文) quality vs. gpt-4o-transcribe over a batch
          of real recordings, then decide whether to promote to a selectable backend
    - [ ] If promoted: add as a dropdown model in the frontend
