"""Build bilingual LaTeX results directly from checked aggregate JSON files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "study11"
R = ["R1_random_pool", "R2_seen_id_new_themes", "R3_new_id_seen_themes", "R4_double_unseen"]


def data(name):
    return json.loads((DATA / f"{name}.json").read_text())


def render(fr):
    controls = data("controls")
    raw = data("raw_scores")
    coverage = data("raw_coverage")
    p2 = controls["priority2_post_ablation"]["by_pca"]["pca_64"]
    p3 = controls["priority3_order_and_confound_controls"]["baselines"]
    ridge = data("ladder_ridge")["models"]["S07_with_post"]
    reve = data("ladder_reve")["models"]["REVE_frozen_vid10"]
    same_ridge = data("same_ridge")["models"]["S07_with_post"]
    same_reve = data("same_reve")["models"]["REVE_frozen_vid10"]
    values = {}

    def number(value, precision=3):
        txt = f"{value:.{precision}f}"
        return txt.replace(".", "{,}") if fr else txt

    def estimate(item, field="mean", precision=3):
        lo, hi = item["ci95"]
        return (
            number(item[field], precision)
            + r"\;["
            + number(lo, precision)
            + r";\,"
            + number(hi, precision)
            + "]"
        )

    def put(key, text):
        values[key] = str(text)

    def row(*cells):
        return " & ".join(cells) + r" \\"

    def math(x):
        return "$" + x + "$"

    def tex_table(cols, rows):
        return (
            "\\begin{tabular}{"
            + cols
            + "}\n\\toprule\n"
            + rows[0]
            + "\n\\midrule\n"
            + "\n".join(rows[1:])
            + "\n\\bottomrule\n\\end{tabular}"
        )

    main_ladder = ridge["stress_va"]["matched_n_train"]["contrasts"]["R4_minus_R1"]
    put("RidgeGap", estimate(main_ladder, "diff", 4))
    put("RidgeGapShort", number(main_ladder["diff"], 3))
    put(
        "ReveGap",
        estimate(reve["stress_logreg"]["matched_n_train"]["contrasts"]["R4_minus_R1"], "diff", 4),
    )
    put(
        "RidgeSame",
        estimate(same_ridge["stress_va"]["matched_n_train"]["contrasts"]["R4_minus_R1"], "diff", 4),
    )
    put(
        "ReveSame",
        estimate(
            same_reve["stress_logreg"]["matched_n_train"]["contrasts"]["R4_minus_R1"], "diff", 4
        ),
    )
    put(
        "RidgeFullGap",
        estimate(ridge["stress_va"]["full_pool_same_cells"]["contrasts"]["R4_minus_R1"], "diff", 4),
    )
    for with_post, key in [(True, "FourPhase"), (False, "ThreePhase")]:
        item = p2["subsets"]["all_with_post" if with_post else "all_no_post"]["stress_va"]
        put(key, number(item["within_id_auc"], 4))
    put(
        "PostDiff",
        estimate(p2["paired_with_minus_no_post"]["all_with_minus_no_post"]["stress_va"], "diff", 4),
    )
    study = data("order_diagnostic")
    put("TreeRaw", number(study["tree_rank"]["within_id"]["mean"]))
    put("TreeSame", estimate(study["tree_rank"]["within_id_and_same_outer_model"]))
    put("FoldPrior", number(study["training_prevalence_only"]["within_id"]["mean"]))
    rows = [
        row(
            "Modèle" if fr else "Model",
            "R1",
            "R2",
            "R3",
            "R4",
            r"$\Delta$ R4$-$R1 (95\% CI)" if not fr else r"$\Delta$ R4$-$R1 (IC 95\,\%)",
        )
    ]
    for label, m in [
        ("Ridge multimodal" if fr else "Multimodal Ridge", ridge["stress_va"]),
        ("REVE + Ridge", reve["stress_va"]),
        ("REVE + logit", reve["stress_logreg"]),
    ]:
        mat = m["matched_n_train"]
        rows.append(
            row(
                label,
                *[math(number(mat["regimes"][k]["within_id_auc"])) for k in R],
                math(estimate(mat["contrasts"]["R4_minus_R1"], "diff", 3)),
            )
        )
    put("LadderTable", tex_table("lccccc", rows))
    order_keys = [
        "p3a_index_linear",
        "p3b_cohort_x_index_poly",
        "p3c_pre0_multimodal",
        "p3d_prev_trial_s07_nonfirst",
        "ref_S07_no_post",
    ]
    labels = (
        [
            "Indice seul",
            "Cohorte $\\times$ indice",
            "Pré-stimulus nominal",
            "Essai précédent",
            "Essai courant sans post",
        ]
        if fr
        else [
            "Trial index",
            "Cohort $\\times$ index",
            "Nominal pre-stimulus",
            "Previous trial",
            "Current clip, no post",
        ]
    )
    rows = [
        row(
            "Entrée" if fr else "Input",
            r"AUC (95\% CI)" if not fr else r"AUC (IC 95\,\%)",
            "C1--2",
            "C3--5",
        )
    ]
    for k, label in zip(order_keys, labels, strict=True):
        item = p3[k]["stress_va"]
        rows.append(
            row(
                label,
                math(estimate(item["overall"], "within_id_auc")),
                math(
                    number(item["by_order_protocol"]["cohorts_1_2_fully_random"]["within_id_auc"])
                ),
                math(
                    number(
                        item["by_order_protocol"]["cohorts_3_5_randomized_blocks"]["within_id_auc"]
                    )
                ),
            )
        )
    put("OrderTable", tex_table("lccc", rows))
    put("CalIDs", coverage["calibration_ids"])
    put("NonRawRows", coverage["non_curated_alignment_trials_excluded"])
    put("NonRawRecords", coverage["non_curated_alignment_recordings_excluded"])
    put("BeforeQC", f"{coverage['evaluation_trials_before_qc']:,}".replace(",", r"\,"))
    put("AfterQC", f"{coverage['evaluation_trials_after_qc']:,}".replace(",", r"\,"))
    put("RawIDs", raw["common_evaluation_ids"])
    put("RawEvaluable", raw["models"]["body_strong"]["n_people"])
    put("RawRows", f"{raw['common_evaluation_trials']:,}".replace(",", r"\,"))
    put("BodyRMSE", number(raw["models"]["body_strong"]["rmse"]["mean"], 2))
    put("BodyR", number(raw["models"]["body_strong"]["r"]["mean"], 3))
    put("OrigRMSE", number(raw["models"]["body57"]["rmse"]["mean"], 3))
    put("TreeRMSE", number(raw["models"]["body_gbdt"]["rmse"]["mean"], 3))
    put("TreeWithEEG", number(raw["models"]["gbdt_corrected"]["rmse"]["mean"], 3))
    put("TreeGain", estimate(raw["paired"]["nonlinear_eeg_gain"]["reduction_rmse"], precision=4))
    put("TreeGainR", estimate(raw["paired"]["nonlinear_eeg_gain"]["delta_r"], precision=4))
    put(
        "RawOriginalGain",
        estimate(raw["paired"]["57_net_before-body"]["reduction_rmse"], precision=4),
    )
    put(
        "ArchiveOriginalGain",
        estimate(raw["paired"]["57_release-body"]["reduction_rmse"], precision=4),
    )
    primary = raw["paired"]["strong_net_after-body"]
    put("RawGainR", estimate(primary["delta_r"], precision=4))
    put("RawGainRMSE", estimate(primary["reduction_rmse"], precision=4))
    put("RawGainShort", number(primary["reduction_rmse"]["mean"], 3))
    put(
        "RawBefore",
        estimate(raw["paired"]["strong_net_before-body"]["reduction_rmse"], precision=3),
    )
    put("RawCleanup", estimate(raw["paired"]["corrected_minus_uncorrected"]["reduction_rmse"]))
    put("RawVsRelease", estimate(raw["paired"]["raw_minus_release"]["reduction_rmse"]))
    put("ResidualGain", estimate(raw["paired"]["residual_corrected_gain"]["reduction_rmse"]))
    put("StrongBodyImprove", estimate(raw["paired"]["strong_body_minus_57"]["reduction_rmse"]))
    put("ReleaseMatched", estimate(raw["paired"]["strong_release-body"]["reduction_rmse"]))
    put("RawOther", estimate(raw["paired"]["matched_minus_other"]["delta_r"]))
    put("RawShuffle", estimate(raw["paired"]["matched_minus_shuffle"]["delta_r"]))
    put("RawInteraction", estimate(raw["distress_interaction_mse_bpm2"], precision=2))
    put("InteractionN", raw["distress_interaction_mse_bpm2"]["n"])
    put("CouplingBefore", number(raw["ecg_coupling"]["median_abs_ecg_r_before"]["mean"]))
    put("CouplingAfter", number(raw["ecg_coupling"]["median_abs_ecg_r_after"]["mean"]))
    put("StrictPre", estimate(raw["strict_pre0_classification"]["joint_pre0"]))
    put("StrictPreN", raw["strict_pre0_classification"]["joint_pre0"]["n"])
    rows = [row("Modèle" if fr else "Model", "$r$", "RMSE (bpm)")]
    for k, en, french in [
        ("last_rr", "Recent RR sequence", "Séquence RR récente"),
        ("body57", "Archived body (57 features)", "Corps historique (57 variables)"),
        ("body_strong", "Body plus raw trajectories", "Corps et trajectoires brutes"),
        (
            "strong_hc_after",
            "Body + bands after ECG regression",
            "Corps + bandes après régression ECG",
        ),
        ("strong_net_before", "Body + raw REVE", "Corps + REVE brut"),
        (
            "strong_net_after",
            "Body + raw REVE after ECG regression",
            "Corps + REVE brut après régression ECG",
        ),
        ("body_gbdt", "Body only, trees", "Corps seul, arbres"),
        (
            "gbdt_corrected",
            "Body + REVE after ECG regression, trees",
            "Corps + REVE après régression ECG, arbres",
        ),
    ]:
        m = raw["models"][k]
        rows.append(
            row(
                french if fr else en,
                math(number(m["r"]["mean"])),
                math(number(m["rmse"]["mean"], 3)),
            )
        )
    put("BodyTable", tex_table("lcc", rows))
    suffix = "_fr" if fr else ""
    (ROOT / "paper" / f"study11_numbers{suffix}.tex").write_text(
        "% Derived from results/study11 by scripts/study11_tables.py.\n"
        + "\n".join("\\newcommand{\\" + key + "}{" + val + "}" for key, val in values.items())
        + "\n"
    )


if __name__ == "__main__":
    render(False)
    render(True)
