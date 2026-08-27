import json
tot = 0
n = 0
for line in open('corpus/manifest.jsonl'):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    dur = d.get('duration') or d.get('duration_s') or d.get('seconds')
    if dur:
        tot += dur
        n += 1
print('videos_with_duration', n, 'total_seconds', tot, 'total_hours', tot/3600)
