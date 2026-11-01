"""Browse indexed practice tasks without an external service.

Examples:
    python scripts/practice.py --tier P0 --limit 10
    python scripts/practice.py --id 200
    python scripts/practice.py --status catalogued --pattern Heap
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Browse the Python Interview Lab task catalog')
    parser.add_argument('--tier', choices=['P0', 'P1', 'P2'])
    parser.add_argument('--status', choices=['implemented', 'catalogued'])
    parser.add_argument('--pattern', help='Case-insensitive substring search')
    parser.add_argument('--id', type=int, help='LeetCode problem number')
    parser.add_argument('--limit', type=int, default=25)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'catalog' / 'problems.csv'
    with source.open(newline='', encoding='utf-8') as fh:
        records = list(csv.DictReader(fh))
    for row in records:
        if args.tier and row['priority'] != args.tier:continue
        if args.status and row['status'] != args.status:continue
        if args.pattern and args.pattern.casefold() not in row['patterns'].casefold():continue
        if args.id is not None and row['leetcode_id'] != str(args.id):continue
        print(f"{row['priority']:2} | LC {row['leetcode_id']:>4} | {row['status']:11} | {row['problem']}")
        print(f"    {row['leetcode_url']}")
        if row['implementation']:print(f"    code: {row['implementation']}")
        args.limit -= 1
        if args.limit <= 0:break


if __name__=='__main__':main()
