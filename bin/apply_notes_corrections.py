#!/usr/bin/env python3
"""Render hash-bound private edit decisions offline; never grant publication."""
import argparse
import json
from pathlib import Path

from meeting_postprocess import notes_correction as nc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path,help='Original held notes directory')
    parser.add_argument('correction',type=Path,help='Directory containing saved response and packets')
    parser.add_argument('decisions',type=Path,help='Independent private decisions bound to response/evidence SHA-256')
    parser.add_argument('output',type=Path,help='Fresh private output directory')
    args=parser.parse_args()
    packets=json.loads((args.correction/nc.PACKETS).read_bytes())
    if isinstance(packets,dict):packets=packets['packets']
    report=nc.render_version(args.original,args.output,(args.correction/nc.RESPONSE).read_bytes(),packets,
                             decisions=json.loads(args.decisions.read_bytes()))
    print(json.dumps({k:report[k] for k in ('status','mode','candidate_selected','proposed_count','applied_count','word_count')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
