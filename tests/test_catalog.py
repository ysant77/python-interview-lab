import csv,json
from pathlib import Path

def test_curriculum_coverage():
    root=Path(__file__).resolve().parents[1]
    coverage=json.loads((root/'catalog/coverage.json').read_text())
    assert coverage['original_16_source_chapters']==16
    assert coverage['workbook_unique_leetcode_ids']==178
    with (root/'catalog/problems.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==coverage['total_index_entries']
    assert all(r['leetcode_url'].startswith('https://leetcode.com/problems/') for r in rows)
    assert all(r['implementation'] for r in rows if r['status']=='implemented')
    assert all(not r['implementation'] for r in rows if r['status']=='catalogued')
    patterns=json.loads((root/'catalog/patterns.json').read_text())
    assert len(patterns)==27
    assert sum(len(p['leetcode_ids']) for p in patterns)==162


def test_complete_public_index_and_ml_track():
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'catalog/coverage.json').read_text())
    assert data['workbook_ids_implemented']==178
    assert data['total_index_entries']==261
    assert data['total_index_implemented']==261
    with (root/'catalog/problems.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==261 and all(row['status']=='implemented' for row in rows)
    topics=json.loads((root/'catalog/ml_topics.json').read_text())
    assert len(topics)==9
    assert all(topic['status']=='implemented' for topic in topics)
