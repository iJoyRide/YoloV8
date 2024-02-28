#!/bin/bash
pip install -U ultralytics
pip install -U ultralytics "ray[tune]"
python /app/train2.py
