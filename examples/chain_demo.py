#!/usr/bin/env python3
"""Elementwise / softmax-chain fusion demo.

    python examples/chain_demo.py
    python examples/chain_demo.py --dtype f32 --rows 64 --cols 128
    python -m fusionpass demo chain
"""
from __future__ import annotations

import sys

from fusionpass.__main__ import main

raise SystemExit(main(["demo", "chain", *sys.argv[1:]]))
