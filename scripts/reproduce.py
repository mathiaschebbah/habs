"""One-command reproduction script for Team 12 (HABS Neuro-AI Hackathon 2026).

Validates the five publication-safe aggregate JSON files in ``results/``, prints
the headline quantitative tables cited in ``paper/submission_paper.pdf``, and
regenerates all four 300-dpi publication figures in ``paper/figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from habs_stress.figures import generate_all


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load and validate the five verified aggregate JSON files."""
    expected = (
        "timecourse.json",
        "physiology.json",
        "benchmarks_ssl.json",
        "causal_forecasting.json",
        "generalization_clinical.json",
    )
    loaded: dict[str, dict] = {}
    for name in expected:
        path = results_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing verified aggregate file: {path}")
        loaded[name] = json.loads(path.read_text())
    return loaded


def print_summary_tables(data: dict[str, dict]) -> None:
    """Print the key quantitative summaries reported in the paper."""
    ph = data["physiology.json"]
    cf = data["causal_forecasting.json"]
    gc = data["generalization_clinical.json"]

    ecg_aud = ph["ecg_morphology"]["ecg_audit"]
    ecg_eff = ph["ecg_morphology"]["ecg_effects"]
    rr_ctrl = ph["ecg_morphology"]["ecg_rr_controlled_and_coupling"]
    print("=== 1. Acute Stress Physiology & ECG Morphology (N = 314, 22,974 trials) ===")
    print(f"  Lead I inverted fraction : {ecg_aud['inverted_lead_fraction']:.4f}")
    print(f"  RR interval d            : {ecg_eff['rr']['vid_change']['stress'][0]:+.4f}")
    print(f"  PR segment d (AV node)   : {ecg_eff['pr_seg']['vid_change']['stress'][0]:+.4f}")
    print(f"  QTc Bazett plain -> RR   : {rr_ctrl['qtc_baz']['plain'][0]:+.4f} -> {rr_ctrl['qtc_baz']['rr_controlled'][0]:+.4f}")

    pri = cf["audit_forecast"]["module2_primary_forecasting"]["primary_cold_start_3x3"]
    loco_hr = gc["module10c_coral_kshot_loco"]["loco_cross_cohort_dev"]["cardiac_forecast_loco"]
    print("\n=== 2. Strictly Causal Early-EEG -> Future Physiology (t = 5.000 s wall) ===")
    for label_name, key in (
        ("Heart rate [5, 10) s", "tgt_ecg_hr_chg_5_10"),
        ("Skin cond. [5, 10) s", "tgt_eda_sc_chg_5_10"),
        ("Respiration [5, 10) s", "tgt_resp_amp_chg_5_10"),
    ):
        m0 = pri[key]["models"]["M0_linear"]["r"]["mean"]
        m2r = pri[key]["paired_vs_m0_linear"]["M2_reve_resid"]["delta_r"]["mean"]
        m2j = pri[key]["paired_vs_m0_linear"]["M2_reve_joint"]["delta_r"]["mean"]
        print(f"  {label_name:22s}: M0={m0:.4f} | d_resid={m2r:+.4f} | d_joint={m2j:+.4f}")
    d_loco = loco_hr["paired_M2_minus_M0_raw"]["delta_r"]
    print(f"  LOCO-CV cardiac (5/5) : d_r={d_loco['mean']:+.4f} {d_loco['ci95']}")

    m10a = gc["module10a_generalization_ablation"]["generalization_ladder"]["S07_enriched"]
    m10e = gc["module10e_clinical_monitoring"]
    print("\n=== 3. Generalization Ladder & Passive Clinical Monitoring (Cold-Start) ===")
    for rkey in (
        "R1_random_5fold",
        "R2_within_person_new_themes",
        "R3_loso_seen_themes",
        "R4_double_unseen_cold_start",
    ):
        auc = m10a[rkey]["stress_ridge_va"]["stress"]["auc"]
        print(f"  {rkey:30s}: AUC={auc:.4f}")
    for ckey in (
        "stress_standard_vs_nonstress",
        "acute_threat_vs_calm_neutral",
        "stress_strict37_vs_relaxed64",
    ):
        by_k = m10e[ckey]["S07_enriched"]["by_k"]
        print(
            f"  {ckey:30s}: 10s AUC={by_k['1']['auc']['mean']:.4f} "
            f"-> 100s AUC={by_k['10']['auc']['mean']:.4f} "
            f"(Sens@90%Spec={100 * by_k['10']['sens_at_spec90']['mean']:.1f}%)"
        )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Reproduce Team 12 tables and figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    data = load_results(args.results_dir)
    print_summary_tables(data)
    if not args.no_figures:
        paths = generate_all(args.results_dir, args.figures_dir)
        print("\nGenerated figures:")
        for p in paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
