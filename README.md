# ThermoShield — Thermal Anomaly Detection Simulation

A software simulation of a thermal-sensing pipeline that detects unusual
surface temperature patterns in low-resolution thermal frames. Written
in Python, with no machine learning — the detector is a rule-based,
threshold-driven classifier, deliberately kept simple so that every
decision can be inspected and tested.

---

## Problem statement

Low-resolution thermal cameras (such as the MLX90640, which produces
32 × 24 = 768 surface-temperature readings per frame) measure thermal
patterns rather than imaging objects. They cannot identify a person, a
weapon, or a face. They can, however, reveal whether a scene contains
regions that are significantly warmer or colder than the surrounding
background.

This project investigates a narrowly scoped question:

> How can a small number of temperature readings be processed to
> distinguish between (a) an empty scene, (b) a scene containing a
> warm object consistent with a person, and (c) a scene containing a
> warm region plus an unusual cold region — a *thermal anomaly* —
> using only lightweight, explainable rules?

The answer is a threshold-based classifier that counts pixels above a
body-temperature threshold and pixels within a "cold-spot" temperature
band, and applies two minimum-count rules. There is no neural network,
no trained model, and no hidden state. The whole algorithm is visible
in `src/thermoshield/detection.py`.

---

## Original hardware concept (background only — not implemented)

ThermoShield began as a concept for a wearable smart-glasses system
that would use an MLX90640 thermal camera, an ESP32-S3 microcontroller
and a small OLED display to surface thermal anomalies to the wearer.

That concept motivated the software in this repository but **has not
been built**. There is no ESP32 firmware in this repository, no code
that communicates with a physical MLX90640 sensor, and no OLED driver.
The current repository is a Python simulation only.

The hardware concept is documented here because it explains *why* the
software was designed the way it was — low resolution, limited compute,
fixed thresholds tuned to indoor ambient conditions — but it must not
be read as a claim of implemented hardware.

---

## What this repository contains

A reproducible Python simulation of the detection pipeline:

1. **Synthetic thermal data generation** (`scripts/generate_data.py`) —
   180 synthetic 32 × 24 thermal frames, split evenly across three
   scenario types (clear, person, thermal anomaly), generated
   deterministically using random seed 42.
2. **Threshold-based detection** (`src/thermoshield/detection.py`) — two
   functions: `detect_threat(frame)` returns a class label;
   `detect_threat_with_confidence(frame)` also returns a confidence
   score. Five thresholds, no machine learning.
3. **A deterministic test suite** (`scripts/run_tests.py`) — eight
   hand-constructed test frames covering the main scenarios, with
   regression assertions that guard the diagnostic against drift.
4. **A visual demonstration** (`scripts/demo_display.py`) — produces
   heatmap + result images for one frame of each scenario.

---

## Detection methodology

The classifier assigns one of three labels to each frame:

| Label | Rule |
|---|---|
| `CLEAR` | Fewer than `MIN_BODY_PIXELS` pixels are at or above `BODY_TEMP_MIN`. |
| `PERSON` | Enough warm pixels to satisfy the body rule, but fewer than `MIN_COLD_PIXELS` pixels fall inside the cold band. |
| `THERMAL ANOMALY` | Both the body rule and the cold-spot rule are satisfied — a warm region and a distinct cold region are both present. |

The five thresholds, defined in `src/thermoshield/configuration.py`:

```python
BODY_TEMP_MIN   = 28.0    # °C — minimum for a pixel to count as "body"
COLD_SPOT_MIN   = 15.0    # °C — lower bound of the cold band
COLD_SPOT_MAX   = 21.0    # °C — upper bound of the cold band
MIN_BODY_PIXELS = 50      # count threshold for the body rule
MIN_COLD_PIXELS = 8       # count threshold for the cold-spot rule
```

These values were chosen for indoor ambient scenes around 22 °C with no
calibration. They are **not** claimed to be optimal. Any real deployment
would require empirical tuning against representative data. Making the
thresholds configurable and comparing fixed-threshold detection against
filtering and adaptive-threshold variants is listed as future work.

An important limitation: a **cold** region is not a *weapon*. The
classifier cannot distinguish "cold drink can", "cold metal object in a
pocket", and "cold surface in an otherwise warm scene". The label
`THERMAL ANOMALY` is therefore literally descriptive and makes no claim
about what object produced the pattern.

---

## Installation

Requirements: Python 3.9 or later (tested on Python 3.9.6, macOS).
Dependencies: `numpy`, `matplotlib`.

```bash
pip3 install numpy matplotlib
```

---

## How to run

From the repository root:

```bash
# Generate 180 synthetic thermal frames into data/
python3 scripts/generate_data.py

# Run the eight deterministic tests
python3 scripts/run_tests.py

# Produce the demo heatmaps (writes PNGs to results/)
python3 scripts/demo_display.py
```

`run_tests.py` is standalone — it does not require `generate_data.py`
to have run first, because it constructs its test frames in memory.

---

## How to run the tests

```bash
cd scripts
python3 run_tests.py              # baseline detection tests (8 tests)
python3 run_config_tests.py       # DetectionConfig tests (7 tests)
python3 run_filtering_tests.py    # median filter tests (8 tests)
python3 run_background_tests.py   # background-aware tests (8 tests)
```

Expected output:

```
Passed: 8 / 8
exit code: 0
```

The test suite covers empty scenes, single-person scenes, cold objects
inside a person region, multiple people, distant subjects, and the
threshold boundary conditions implied by the eight scenario frames.
Each test case uses an exactly constructed frame rather than random
data, so the results are fully deterministic.

---

## Demonstration

`demo/` contains:

- `demo/demovideo.mp4` — a short walkthrough of the detection pipeline.
- `demo/demoscreen.JPG` — a screenshot of the demo output.

The three demo PNGs (`demo_clear_room.png`, `demo_person.png`,
`demo_thermal_anomaly.png`) are generated on demand by
`demo_display.py` into `results/` and are gitignored, not committed.

---

## Limitations

This project is deliberately narrow, and being explicit about what it
does **not** do is part of the engineering.

- **No real sensor data.** Every frame is synthetic (Gaussian noise
  around nominal temperatures). Real MLX90640 data would include
  fixed-pattern noise, per-pixel calibration drift and emissivity
  variation across surfaces — none of which the synthetic generator
  models.
- **No hardware integration.** The ESP32-S3 + MLX90640 + OLED concept
  is documented as motivation only. No firmware, no sensor driver, no
  display driver is in this repository.
- **No machine learning.** The classifier is threshold-based. There
  is no trained model, no dataset, and no accuracy figure based on
  real data. The eight test cases establish that the *rule logic*
  behaves as specified — they do not establish detection performance
  on realistic thermal scenes.
- **Low thermal resolution.** At 32 × 24, each pixel covers roughly
  1/768th of the frame. Small objects and thin structures are
  invisible at this resolution.
- **Environmental sensitivity.** The cold band [15 °C, 21 °C] is
  tuned to indoor ambient conditions around 22 °C. In a colder room
  (e.g. 18 °C ambient) the background itself would fall inside the
  cold band and the "person + cold region" rule would misfire.
  Adaptive thresholds would help; they are listed as future work.
- **No accuracy claims on real data.** Any numbers in this repository
  describe behaviour on synthetic test frames only.

---

## Implemented since initial release

The following were listed as future work in the initial release and
have since been implemented, tested, and pushed:

1. **Package structure** — code lives in `src/thermoshield/` with thin
   entry points in `scripts/`. `pip install --user -e .` works from any
   directory.
2. **Configurable thresholds** — `DetectionConfig` in
   `src/thermoshield/configuration.py` parameterises all five
   thresholds. Custom configurations can be compared side-by-side
   without editing the detection module.
3. **Median filtering** — a 3 × 3 median filter in
   `src/thermoshield/filtering.py`, opt-in via
   `DetectionConfig(median_filter=True)`.
   `scripts/run_filtering_tests.py` demonstrates that enabling the
   filter eliminates a noise-induced false positive that the baseline
   produces (10 × 10 body region + 20 scattered cold pixels: baseline
   calls it THERMAL ANOMALY, filtered classifier correctly calls it
   PERSON).
4. **Background-aware detection** —
   `DetectionConfig(background_aware=True)` applies body and cold-spot
   rules to (pixel − background) deltas rather than absolute
   temperatures, where the background is the median of all 768 pixel
   values. Directly addresses the 'Environmental sensitivity'
   limitation: at 18 °C ambient the baseline misclassifies a
   person-only scene as THERMAL ANOMALY (the 18 °C background falls
   inside the [15, 21] °C cold band), while background-aware detection
   correctly returns PERSON.
   `scripts/run_background_tests.py` proves this with 8 tests.
5. **Input validation** — `validate_frame` in
   `src/thermoshield/validation.py` rejects `None`, non-numeric input,
   frames of the wrong length or shape, and non-finite values (NaN,
   ±inf) before any detection logic runs. Validation runs *before*
   the `median_filter` and `background_aware` branches, so callers
   get the validation error rather than a lower-level downstream
   error. `scripts/run_validation_tests.py` covers 15 cases.

## Still planned

1. **Evaluation harness** — precision, recall, F1 and processing time
   over controlled synthetic datasets with known ground-truth labels.
2. **Documentation** — `docs/architecture.md`,
   `docs/detection_methodology.md`, `docs/testing_and_evaluation.md`,
   `docs/limitations.md`.

## Design decisions and deferred features

**Adaptive thresholds — analysis and decision.** The initial
future-work list proposed moving-average or percentile-based adaptive
thresholds. Background-aware detection (implemented in commit
`929744c`) already addresses the motivating problem — sensitivity to
ambient temperature — by computing a per-frame background estimate
from the median of all 768 pixel values.

The median of 768 samples has a standard error of approximately
1.25 × σ / √768. At typical sensor noise (σ ≈ 0.5 °C) this is
below 0.03 °C — far smaller than any physically meaningful ambient
fluctuation over the timescales a thermal camera operates on.
Cross-frame smoothing would therefore add no measurable detection
improvement over per-frame estimation.

A stateful moving-average variant was designed and rejected on three
grounds:

1. **No measurable benefit** — as above, the per-frame median is
   already stable to within 0.03 °C.
2. **API cost** — a moving-average classifier must carry state across
   frames. The current API (`classify_frame(frame, config) -> label`)
   is deliberately stateless: identical inputs always produce identical
   outputs, which makes the 46-test suite reproducible without setup
   or mocking.
3. **Mismatch to the synthetic data model** — the generator produces
   independent frames with no temporal sequence. There is no 'previous
   frame' for a moving average to average over.

If the project moves to real MLX90640 recordings with temporal
sequences, this decision should be revisited: a moving-average or
Kalman filter over the background estimate would then be justified by
a real signal-to-noise argument.
