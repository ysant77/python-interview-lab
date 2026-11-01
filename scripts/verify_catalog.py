"""Check the public CSV catalog against Python code symbols (no private workbooks needed).

Used in CI to prevent claiming implementations that don't exist. The catalog
is a fixed, attributed index derived from user-supplied reference lists.
"""
from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with (ROOT/'catalog/problems.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    symbols={}
    for source in (ROOT/'src/interview_patterns').glob('*.py'):
        tree=ast.parse(source.read_text(encoding='utf-8'))
        defined={node.name:ast.get_docstring(node) or '' for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        symbols[source.relative_to(ROOT).as_posix()]=defined
    errors=[]
    for record in rows:
        if not record['leetcode_url'].startswith('https://leetcode.com/problems/'):
            errors.append(f"invalid URL {record['problem']}")
        if record['status']=='implemented':
            entry=record['implementation']
            if '::' not in entry:errors.append(f"missing symbol ref {record['problem']}");continue
            file_name,symbol=entry.split('::',1)
            if symbol not in symbols.get(file_name,{}):errors.append(f"missing symbol {entry}")
            elif record['leetcode_id'] and f"LC {record['leetcode_id']}" not in symbols[file_name][symbol]:
                errors.append(f"LC ID absent in docstring: {entry}")
        elif record['status']=='catalogued' and record['implementation']:
            errors.append(f"catalogued row contains symbol: {record['problem']}")
    metadata=json.loads((ROOT/'catalog/coverage.json').read_text())
    if sum(r['status']=='implemented' for r in rows)!=metadata['total_index_implemented']:
        errors.append('coverage count inconsistent')
    if len(rows)!=metadata['total_index_entries']:
        errors.append('catalog row count inconsistent')
    for line in errors:print('ERROR',line,file=sys.stderr)
    print(f"Catalog: {len(rows)} indexed; {metadata['total_index_implemented']} implemented; {len(errors)} inconsistencies")
    return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
