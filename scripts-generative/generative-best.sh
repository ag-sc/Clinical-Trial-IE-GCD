#!/bin/bash
cd "$(dirname "$0")"
cd ..
source venv/bin/activate
export PYTHONPATH=./src/
python -u src/generative_approach/training.py $1 $2
