#!/usr/bin/env bash
# Successor to transcribe_loop.sh, retargeted at the urls_batch2.txt scrape
# (run3) now that urls_scale.txt's run has finished. Waits for the still-
# running transcribe_loop.sh instance to vacate GPU0 before starting, so we
# never run two `transcribe` processes on the same GPU at once.
set -u
cd /data3/vandan.raval/indic-duplex-pipeline
set -a; source .env; set +a
export PATH="/data3/vandan.raval/indic-duplex-pipeline/venv/bin:/home/vandan.raval/miniconda3/bin:$PATH"

echo "[transcribe_loop2] waiting for prior transcribe pass to finish"
while pgrep -f "duplex_corpus.py transcribe" > /dev/null; do
    sleep 30
done

echo "[transcribe_loop2] starting, tracking urls_batch2.txt scrape"
while pgrep -f "duplex_corpus.py run --urls urls_batch2.txt" > /dev/null; do
    python3 duplex_corpus.py transcribe --root ./corpus --gpus 0
    sleep 180
done

echo "[transcribe_loop2] scrape process ended — final catch-up pass"
python3 duplex_corpus.py transcribe --root ./corpus --gpus 0
echo "[transcribe_loop2] done"
