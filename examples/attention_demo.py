#!/usr/bin/env python3
"""Attention fusion demo.

    python examples/attention_demo.py
    python examples/attention_demo.py --dtype f32 --seq 64 --head-dim 16
    python -m fusionpass demo attention
"""
from __future__ import annotations

import sys

from fusionpass.__main__ import main

raise SystemExit(main(["demo", "attention", *sys.argv[1:]]))
