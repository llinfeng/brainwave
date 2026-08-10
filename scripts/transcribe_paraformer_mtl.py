# /// script
# dependencies = ["dashscope", "requests"]
# ///
"""Transcribe a local audio file with Aliyun DashScope paraformer-mtl-v1.

The file-transcription API only accepts URLs, so the local file is first
uploaded to DashScope's temporary OSS bucket (readable only by this API key)
and submitted as an oss:// URL with resource resolution enabled.

Usage:
    uv run scripts/transcribe_paraformer_mtl.py <audio.wav> [more files...]

Output: <stem>_by_ParaformerMTL.txt next to each input, plus the raw JSON
(with sentence/word timestamps) as <stem>_by_ParaformerMTL.json.

API key: QWEN_Prepaid_10RMB_per_Month or DASHSCOPE_API_KEY env var.
First test (2026-08-04): 117s bilingual WAV transcribed in ~2s; quality on
code-switched audio below MiMo but supports files up to 2 GB — candidate for
the long-recording role (see AudioWrite/dev/mimo_improvement_plan.md).
"""
import json
import os
import sys
from http import HTTPStatus

import requests
import dashscope
from dashscope.audio.asr import Transcription
from dashscope.utils.oss_utils import check_and_upload_local

MODEL = 'paraformer-mtl-v1'


def transcribe(path: str, api_key: str) -> str:
    is_upload, file_url, _cert = check_and_upload_local(MODEL, f'file://{path}', api_key)
    print(f'uploaded={is_upload}')

    task = Transcription.async_call(
        model=MODEL,
        file_urls=[file_url],
        headers={'X-DashScope-OssResourceResolve': 'enable'},
    )
    print(f'task_id={task.output.task_id}')

    result = Transcription.wait(task=task.output.task_id)
    if result.status_code != HTTPStatus.OK:
        raise RuntimeError(f'{result.status_code}: {result.output}')

    texts = []
    stem = os.path.splitext(path)[0]
    for r in result.output.get('results', []):
        if r.get('subtask_status') != 'SUCCEEDED':
            raise RuntimeError(f'subtask failed: {json.dumps(r, ensure_ascii=False)}')
        detail = requests.get(r['transcription_url']).json()
        with open(f'{stem}_by_ParaformerMTL.json', 'w', encoding='utf-8') as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
        texts += [t.get('text', '') for t in detail.get('transcripts', [])]

    text = '\n'.join(texts)
    out = f'{stem}_by_ParaformerMTL.txt'
    with open(out, 'w', encoding='utf-8-sig') as f:
        f.write(text + f'\n\n---\n_Transcribed by: {MODEL} (aliyun)_\n')
    print(f'saved {out} ({len(text)} chars)')
    return text


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    key = os.environ.get('QWEN_Prepaid_10RMB_per_Month') or os.environ.get('DASHSCOPE_API_KEY')
    if not key:
        sys.exit('Set QWEN_Prepaid_10RMB_per_Month or DASHSCOPE_API_KEY')
    dashscope.api_key = key
    for p in sys.argv[1:]:
        print(f'=== {p}')
        print(transcribe(p, key))
