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
    - [x] Compare bilingual (EN/中文) quality vs. gpt-4o-transcribe over a batch
          of real recordings — done 2026-08-12: MiMo won 12/15 recordings judged
          by subagent panels (best code-switch fidelity + completeness; caught two
          gpt-4o truncations). Full findings: AudioWrite/_evals/
    - [x] Lift MiMo's length ceiling: uploads now transcoded to MP3
          (~3.5 min -> ~26 min; 2K output tokens bind past ~10-15 min)
    - [ ] Promote MiMo to a selectable backend in the frontend dropdown
          (quality case is made; needs UI option + server mode)
    - [x] Chunking: >10 min clips split into ~600s MP3 chunks, transcribed
          concurrently, merged into one txt (done 2026-08-12)
    - [ ] Also evaluated: Aliyun paraformer-mtl-v1 (second shadow) — last in all
          15 on quality; keep for speed/long-file lane + third-witness only

## ASR model evaluation exercise — zh/en code-switching (2026-08-23, IN PROGRESS)

Goal: find the most accurate model for the user's Mandarin–English code-switched
dictation. Standing record, registry, and status: `AudioWrite/_evals/README.md`.

- [x] Third shadow: `qwen3-asr-flash-realtime` (batch-at-Stop over WebSocket; returns
      the instant `session.finished` arrives — 1.5 s → 0.5–0.7 s). Cheapest API lane.
- [x] Frontend: shadow tabs (one per model, spinner until its result lands), stacked
      below the main transcript; shadows pushed the moment each model finishes, not
      after gpt-4o. Every pane editable; Save writes ALL edited panes back to their
      .txt with the `_Transcribed by … in N.Ns_` footer preserved + an edit stamp;
      autosave ~2 s after typing stops, status shown next to Save.
- [x] Reverted the gpt-4o chunked-recovery experiment (shadows arrive too late to
      cross-check against; heavy post-processing defeats the point of the fast lane).
- [x] Corpus hygiene: 8 half-speed WAVs repaired (lossless header rewrite), 61 clips
      moved to `AudioWrite/drop/`, benchmark pool = 553 good recordings.
      Records: `AudioWrite/_CLEANUP-2026-08-23.md`, `_AUDIO-ENCODING-ISSUES.md`.
- [x] Full-corpus transcript sets in `AudioWrite/_evals/raw/`: `gpt-4o-transcribe`
      (553) and `qwen3-asr-1.7b-local` (553, RTX 3090, 19 min for 8.9 h of audio).
- [x] Findings so far (details in `_evals/`): gpt-4o-transcribe is the WEAKEST model
      on this corpus — 10% content drops, hallucinates on silence, translates EN→中文
      on ~24% of mixed clips, 22% unpunctuated, odd-one-out in 30/43 three-way clips;
      also the most expensive viable option (0.6¢/min vs MiMo 0.12, local 0).
      MiMo and local Qwen agree with each other at 0.95. Qwen3 has its own defect:
      translates 中文→EN on ~10% of mixed clips. Code-switch preservation is
      CONTESTED; only hand-labeled mixed clips can rank it.
- [x] Literature/API survey (4 agents): no public zh-en code-switch leaderboard;
      FireRedASR2-LLM (public sets) and MiMo-V2.5-ASR (vendor set) beat Qwen3.
      Local harness + install playbook ready in `~/Tooling/qwen3-asr-local/`.
- [ ] DECIDE: promote MiMo-V2.5-ASR (API) to primary; keep gpt-4o as tie-breaker shadow.
- [ ] DECIDE: which local models to run on the 553 pool (small-first: Fun-ASR-Nano
      ± hotwords → FireRedASR2-AED → one 8B: MiMo local or FireRedASR2-LLM).
- [ ] Hand-label ~30 genuinely mixed clips → real code-switch ranking.
- [ ] Hotword/context A/B on the jargon list (Qwen3 `context=`, Fun-ASR `hotwords`).
- [ ] API shadows to consider (paid, needs explicit yes): `fun-asr` (independent #1 on
      Chinese domains, same DashScope key), `gpt-transcribe` (OpenAI's successor to
      gpt-4o-transcribe, which is being retired).
- [ ] Fix the transcript naming bug: upload paths (and some recording-time paths)
      generate a second descriptive name for the .txt, leaving 31 .txt/.wav pairs
      with mismatched suffixes; re-pair them + move the 28 WAV-less .txt to drop/.
- [ ] Rule (learned the hard way): no bulk paid API runs without explicit approval.
