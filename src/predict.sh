#!/usr/bin/env bash
set -e
set -v
# Use Wikitext test split for predictions
python3 src/NGram_Model.py test --work_dir work --split test --test_output $2
