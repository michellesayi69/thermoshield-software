# ThermoShield — Software Simulation

AI-powered thermal detection prototype that classifies thermal images as weapon, person, or clear.

## Overview

ThermoShield uses a threshold-based detection algorithm and a trained AI classifier to detect possible concealed metal objects on the human body.

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

## Author

Person C — AI & Detection Logic Lead
China Jiliang University
September 2026