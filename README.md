# ThermoShield — Thermal Anomaly Detection Simulation

A software simulation of a thermal-sensing pipeline that classifies low-resolution thermal frames as CLEAR, PERSON, or THERMAL ANOMALY. Threshold-based, no machine learning.

## Overview

ThermoShield uses a rule-based, threshold-driven classifier that counts warm and cold pixels in 32 x 24 thermal frames. There is no trained model: the algorithm lives in scripts/detect_threat.py and every decision is inspectable.

## Project Structure

- `scripts/generate_data.py` — generates 180 synthetic thermal images
- `scripts/detect_threat.py` — threshold detection logic
- `scripts/demo_display.py` — visual demo with heatmap and result banner
- `scripts/run_tests.py` — runs 8 test cases

## Requirements

- Python 3.x
- numpy
- matplotlib

## Setup

pip3 install numpy matplotlib
python3 scripts/generate_data.py
python3 scripts/run_tests.py

## Demo

- [Demo Video](demo/demovideo.mp4)
- [UI Screenshot](demo/demoscreen.JPG)
## Author

Michelle Sayi
China Jiliang University
September 2026