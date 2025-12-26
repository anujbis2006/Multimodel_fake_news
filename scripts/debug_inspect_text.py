#!/usr/bin/env python3
from pathlib import Path
import json
import re
from text_preprocessing import extract_text, clean_text

p = Path('data/fakeddit_subset/validation_data_fakeddit.jsonl')
cnt = 0
with p.open('r', encoding='utf-8') as f:
    for i, line in enumerate(f, start=1):
        if i > 30:
            break
        rec = json.loads(line)
        raw = json.dumps(rec)[:500]
        cleaned = extract_text(rec)
        # find common answer tokens
        has_yes = bool(re.search(r"\byes\b", json.dumps(rec), flags=re.I))
        has_no = bool(re.search(r"\bno\b", json.dumps(rec), flags=re.I))
        print('--- sample', i)
        print('cleaned:', cleaned)
        print('contains yes in raw JSON:', has_yes, 'contains no in raw JSON:', has_no)
        print()
