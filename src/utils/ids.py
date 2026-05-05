"""Deterministic call_id generation for batch processing.

The orchestrator's default call_id format (`call_<basename>_<uuid8>`)
produces a different id every run, which prevents
`ResultsStore.is_already_processed` from skipping completed files on
resume. The batch runner uses `make_call_id` instead so the id is a
function of the audio content; re-running the same file produces the
same id and the runner can skip it.
"""
import hashlib
import os


def make_call_id(audio_path: str) -> str:
    """Build a deterministic call_id from the audio file's basename and content.

    Format:
        call_<basename>_<sha256_hex_first_12>

    Streaming hash (~50 ms for a typical call WAV). Two paths to the same
    file content produce the same id; the same basename with different
    content produces different ids.
    """
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    return f"call_{basename}_{h.hexdigest()[:12]}"
