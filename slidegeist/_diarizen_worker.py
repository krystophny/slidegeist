"""Standalone DiariZen worker, executed by a foreign Python interpreter.

This file ships inside the slidegeist package but is **never imported by it**.
It runs under a separate interpreter (see ``SLIDEGEIST_DIARIZEN_PYTHON``) which
has DiariZen and torch installed but not slidegeist, so it must stay on the
standard library plus ``diarizen``.

Contract: exactly one JSON document on stdout. Everything else - progress,
Hugging Face download chatter, torch warnings - goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys

SCHEMA = "slidegeist-diarization-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diarize an audio file with DiariZen")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Imported here so --help works without torch present.
    from diarizen.pipelines.inference import DiariZenPipeline

    kwargs = {}
    if args.device and args.device != "auto":
        kwargs["device"] = args.device

    print(f"loading {args.model} (device={args.device})", file=sys.stderr, flush=True)
    pipeline = DiariZenPipeline.from_pretrained(args.model, **kwargs)

    print(f"diarizing {args.audio}", file=sys.stderr, flush=True)
    results = pipeline(args.audio)

    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in results.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda item: (item["start"], item["end"]))

    audio_duration = max((turn["end"] for turn in turns), default=0.0)
    json.dump(
        {
            "schema": SCHEMA,
            "model": args.model,
            "audio_duration": audio_duration,
            "turns": turns,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
