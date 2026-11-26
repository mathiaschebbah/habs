"""Minimal, reproducible codebase for HABS Neuro-AI Hackathon 2026 (Team 12).

Modules
-------
signals
    Multimodal BDF reading, GSR conductance calibration, automatic ECG polarity
    alignment and P-QRS-T delineation, and strictly causal t = 5.000 s windowing.
protocol
    Zero-leakage person x theme splits, nested cross-validation, Cold-Start vs.
    causal Empirical-Bayes normalization, and paired person-level inference.
physiology
    Acute stress definitions, within-person effect sizes, RR-adjusted ECG tests,
    and fixed-video appraisal contrasts.
forecasting
    Causal early-EEG to future autonomic forecasting (M0 vs. M1 vs. M2),
    artefact residualization, and idiosyncratic same-video controls.
generalization
    Unsupervised CORAL, chronological k-shot Empirical-Bayes MAP adaptation,
    Varimax subjective-autonomic coherence, and 10 s to 100 s monitoring curves.
figures
    Deterministic reproduction of the four publication figures from results/*.json.
"""

from habs_stress import (
    figures,
    forecasting,
    generalization,
    physiology,
    protocol,
    signals,
)

__all__ = [
    "figures",
    "forecasting",
    "generalization",
    "physiology",
    "protocol",
    "signals",
]
