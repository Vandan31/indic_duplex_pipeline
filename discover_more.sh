#!/usr/bin/env bash
# Gathers a second batch of candidate URLs (beyond the original 751) across
# the same 10 target languages, so there's a ready-to-go pool once the
# current run exhausts urls_scale.txt. Doesn't touch corpus/state.db or GPU
# resources — pure yt-dlp search, safe to run alongside the live scrape.
set -u
cd /data3/vandan.raval/indic-duplex-pipeline
export PATH="/home/vandan.raval/miniconda3/envs/duplex/bin:/home/vandan.raval/miniconda3/bin:$PATH"

LANGS=(hindi tamil telugu bengali marathi kannada malayalam gujarati punjabi urdu)
QUERIES=("podcast interview conversation" "two friends talk show chat")

mkdir -p data/discover2
for lang in "${LANGS[@]}"; do
    for q in "${QUERIES[@]}"; do
        out="data/discover2/${lang// /_}_${q// /_}.txt"
        echo "[discover_more] $lang: $q"
        python3 duplex_corpus.py discover --query "$lang $q" --limit 150 \
            --cookies cookies.txt --out "$out"
    done
done

# merge, dedupe against the batch already in flight
cat data/discover2/*.txt urls_scale.txt 2>/dev/null | sort -u > /tmp/all_seen.txt
cat data/discover2/*.txt | sort -u > /tmp/new_only.txt
comm -23 /tmp/new_only.txt <(sort -u urls_scale.txt) > urls_batch2.txt
echo "[discover_more] wrote $(wc -l < urls_batch2.txt) new candidate url(s) -> urls_batch2.txt"
