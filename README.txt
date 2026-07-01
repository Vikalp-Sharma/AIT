AIT — AI Trainer & Tester
==========================

INSTALL
-------
pip install numpy scikit-learn joblib sounddevice matplotlib

RUN
---
python AIT.py

HOW TO USE
----------
TRAIN tab:
  1. Type a word name and ID number, click + ADD
  2. Click the class row to select it
  3. Hold R (or type Quick ID + hold R) to record 1 second
  4. Repeat 5-10 times per word for good accuracy
  5. Click TRAIN MODEL
  6. BGN (background noise) is added automatically - no need to record it

TEST tab:
  1. Make sure a model is trained first
  2. Hold R and say a word
  3. Release R to see the prediction + confidence bars
  4. BGN appears in the chart showing how much background noise is detected

FILES CREATED
-------------
  recordings/       - your .npy audio files
  model.pkl         - trained sklearn pipeline
  weights.json      - model metadata
  meta.json         - class names and IDs

NOTES
-----
- BGN is class ID 0, always reserved for background noise
- BGN samples are generated automatically (white noise + low-freq rumble)
- The chart shows all classes including BGN
- If BGN wins in the test, it means the model heard silence/noise, not a word
