#!/usr/bin/env bash
# Keeps transcribing newly-accepted videos on GPU0 only, alongside the
# scrape running on GPUs 1-3. First pass uses --overwrite once to fix the
# broken empty transcripts from the earlier wrong-env run; after that it
# runs plain (no --overwrite) so it only ever picks up videos that don't
# have a transcript.json yet — i.e. freshly accepted ones.
set -u
cd /data3/vandan.raval/indic-duplex-pipeline
set -a; source .env; set +a
export PATH="/data3/vandan.raval/indic-duplex-pipeline/venv/bin:/home/vandan.raval/miniconda3/bin:$PATH"

echo "[transcribe_loop] initial pass with --overwrite (fixing broken empties)"
python3 duplex_corpus.py transcribe --root ./corpus --gpus 0 --overwrite

while pgrep -f "duplex_corpus.py run --urls urls_scale.txt" > /dev/null; do
    sleep 180
    echo "[transcribe_loop] catching up on newly-accepted videos"
    python3 duplex_corpus.py transcribe --root ./corpus --gpus 0
done

echo "[transcribe_loop] scrape process ended — final catch-up pass"
python3 duplex_corpus.py transcribe --root ./corpus --gpus 0
echo "[transcribe_loop] done"
