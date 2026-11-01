"""Build a navigable Markdown index from public CSV and Python docstrings."""
from __future__ import annotations
import ast
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main()->None:
    with (ROOT/'catalog/problems.csv').open(newline='',encoding='utf-8') as file:
        problems=list(csv.DictReader(file))
    documents={}
    for file in (ROOT/'src/interview_patterns').glob('*.py'):
        tree=ast.parse(file.read_text())
        documents[file.relative_to(ROOT).as_posix()]={node.name:(ast.get_docstring(node) or '') for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
    lines=['# Python Solution Index', '',
           'All entries link to official LeetCode problem pages and to Python reference implementations.',
           'The descriptions are summaries, not full correctness proofs. Before using an implementation, read its assumptions and tests.', '',
           '| Priority | LC | Pattern/problem | Approach and complexity | Code |', '|:--|--:|:--|:--|:--|']
    for record in problems:
        filename,symbol=record['implementation'].split('::',1) if record['implementation'] else ('','')
        desc=documents.get(filename,{}).get(symbol,'').split('\n',1)[0].replace('|','\\|')
        code_link=f"[`{symbol}`](../{filename})" if filename else 'Not implemented'
        problem=f"[{record['problem']}]({record['leetcode_url']})"
        lines.append(f"| {record['priority']} | {record['leetcode_id']} | {problem} | {desc} | {code_link} |")
    (ROOT/'docs/SOLUTION_INDEX.md').write_text('\n'.join(lines)+'\n')
    print(f"Index generated: {len(problems)} entries")


if __name__=='__main__':main()
