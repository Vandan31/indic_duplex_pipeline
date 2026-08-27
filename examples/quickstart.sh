#!/usr/bin/env bash
# Smallest end-to-end run.
set -e
: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY — see .env.example}"

python duplex_corpus.py discover \
    --query "hindi podcast two people conversation" --limit 20 --out urls.txt

python duplex_corpus.py run --urls urls.txt --root ./corpus --limit 5

python duplex_corpus.py stats --root ./corpus
python duplex_corpus.py check --root ./corpus
