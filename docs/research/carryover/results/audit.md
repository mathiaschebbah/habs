# Cells: `audit/02_emotional_carryover.py`

## Cell 0: `# %% [markdown]`

# Saved carryover estimates and grouped J2 forecasts
#
Read the completed source/freeze evidence and cached reports at their original
absolute workspace path. No model is fitted and no report is overwritten.

## Cell 1: `# %%`

```python
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import skore
from scipy.linalg import cho_factor, cho_solve

from habs.evaluate import participant_macro_mae
from habs.temporal.data import verify_inputs
from habs.temporal.inference import _marginal, _mean_design

ROOT = Path(os.environ.get("HABS_ANALYSIS_OUT", "/out"))
audit_state = {
    "source": False,
    "likelihood": False,
    "predictions": False,
    "paired": False,
}
PUBLIC = ROOT / "public"
PRIVATE = ROOT / "temporal"
result = json.loads((PUBLIC / "temporal_results.json").read_text())
frozen = json.loads((PUBLIC / "temporal_frozen.json").read_text())
producer = importlib.import_module("experiments.02_emotional_carryover")
verify_inputs(ROOT / "data")
for name, expected in frozen["source_files_sha256"].items():
    assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == expected
assert (
    hashlib.sha256((ROOT / "data/features.csv.gz").read_bytes()).hexdigest()
    == frozen["input_hashes"]["input_features_sha256"]
)
assert frozen["status"] == "frozen_before_empirical_carryover_fits"
assert result["status"] == "empirical_carryover_complete"
assert (
    result["frozen_sha256"]
    == hashlib.sha256((PUBLIC / "temporal_frozen.json").read_bytes()).hexdigest()
)
assert (
    result["scientific_code_sha256"]
    == frozen["scientific_code_sha256"]
    == producer._code_hash()
)
audit_state["source"] = True
{
    "source_revision": frozen["source_revision"],
    "frozen_at_utc": frozen["frozen_at_utc"],
    "source_and_freeze_verified": True,
    "audited_result_sha256": hashlib.sha256(
        (PUBLIC / "temporal_results.json").read_bytes()
    ).hexdigest(),
    "audited_frozen_sha256": hashlib.sha256(
        (PUBLIC / "temporal_frozen.json").read_bytes()
    ).hexdigest(),
    "structure": result["structure"],
    "real_data_smoke": result["real_data_smoke"],
}
```

**stdout:**
```
Out[0]: 
{'source_revision': '5b079a836da54652240c00c43555a1624400ae0e',
 'frozen_at_utc': '2026-11-09T12:37:36.537708+00:00',
 'source_and_freeze_verified': True,
 'audited_result_sha256': '32d84ed0de2a4487a6cb6f1896ed0f4926b890250496962adbf00cf662f90c6c',
 'audited_frozen_sha256': '7f9054aa4b4d7a2dd652ea837ecdcd54cc5aa967e9965bc90cd8b6d7357b640f',
 'structure': {'people': 53,
  'transitions': 4554,
  'by_day': {'1': 2294, '2': 2260},
  'minimum_pairs': 20,
  'population_scales': False,
  'reference_gap_s': 15.956999999999994,
  'j1_gap_iqr': [13.144249999999829, 19.63600000000008],
  'j1_scale_ranges': {'sd_a': [0.9904138262210453,
    1.6661063479996414,
    2.0750939692516313,
    2.378856427036421,
    3.5086615549338336],
   'sd_p': [0.13691476535044217,
    0.21853521274383345,
    0.2572572843373988,
    0.30883250002814555,
    0.507295758996839]},
  'scope': 'Exploratory cohort-2 identifiers, fixed video catalogue; novel current videos for each person.'},
 'real_data_smoke': '2 passed in 15.59s'}
```

## Cell 2: `# %% [markdown]`

## Likelihood and sampling-covariance reconstruction

## Cell 3: `# %%`

```python
saved = np.load(PRIVATE / "estimates.npz")
values, sampling, counts, ratio = (
    saved[name] for name in ("values", "sampling", "counts", "scale_ratio")
)
parameters, mean = saved["reml_parameters"], saved["mean_parameters"]
matrix, sd, rho, covariance = _marginal(parameters, sampling, None)
design = _mean_design(counts, True, ratio)
factor = cho_factor(matrix, lower=True)
information = design.T @ cho_solve(factor, design)
mean_factor = cho_factor(information, lower=True)
residual = values - design @ mean
objective = float(
    np.sum(np.log(np.diag(factor[0])))
    + np.sum(np.log(np.diag(mean_factor[0])))
    + 0.5 * residual @ cho_solve(factor, residual)
)
assert abs(objective - result["variance_components"]["reml_objective"]) < 1e-8
assert abs(covariance - result["variance_components"]["covariance"]) < 1e-12
assert (
    np.max(
        np.abs(values.reshape(2, -1) - np.asarray(result["anonymous_delta_estimates"]))
    )
    < 1e-12
)
assert np.linalg.eigvalsh(sampling).min() > 0
audit_state["likelihood"] = True
{
    "saved_likelihood_verified": True,
    "covariance": covariance,
    "covariance_ci95": result["variance_components"]["covariance_ci95"],
    "null_bootstrap_p_two_sided": result["variance_components"][
        "null_bootstrap_p_two_sided"
    ],
    "random_effect_sd": sd.tolist(),
    "rho_point": rho,
    "rho_profile95_grid": result["variance_components"]["rho_profile95_grid"],
    "correlation_weakly_identified": result["variance_components"][
        "correlation_weakly_identified"
    ],
}
```

**stdout:**
```
Out[0]: 
{'saved_likelihood_verified': True,
 'covariance': 0.003495956009668797,
 'covariance_ci95': [-0.010277306157027757, 0.015054966339341981],
 'null_bootstrap_p_two_sided': 0.253,
 'random_effect_sd': [0.04500069862544692, 0.07807279443796387],
 'rho_point': 0.9950547536867305,
 'rho_profile95_grid': [-0.995, 0.9950547536867305],
 'correlation_weakly_identified': True}
```

## Cell 4: `# %% [markdown]`

## Stored skore reports

## Cell 5: `# %%`

```python
report_workspace = PRIVATE / "reports"
project = skore.Project(
    "habs-emotional-carryover", mode="local", workspace=str(report_workspace.resolve())
)
summary = project.summarize()
summary.frame()
```

**stdout:**
```
Out[0]: 
                                       skore-version  \
  id                                                   
0 01a1c051-f82c-7243-b346-6ec220bedc67        0.29.1   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8        0.29.1   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8        0.29.1   

                                                           creation-date  \
  id                                                                       
0 01a1c051-f82c-7243-b346-6ec220bedc67  2026-11-09T12:39:19.853034+00:00   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8  2026-11-09T12:40:58.820025+00:00   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8  2026-11-09T12:42:38.784937+00:00   

                                             report_type  \
  id                                                       
0 01a1c051-f82c-7243-b346-6ec220bedc67  cross-validation   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8  cross-validation   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8  cross-validation   

                                                                   report_id  \
  id                                                                           
0 01a1c051-f82c-7243-b346-6ec220bedc67  01a1c051-f82c-7243-b346-6ec220bedc67   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8  01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8  01a1c055-0140-7558-ba9d-1ffddcbf84e8   

                                                                                 learner  \
  id                                                                                       
0 01a1c051-f82c-7243-b346-6ec220bedc67  SkrubLearner(data_op=<Apply TemporalForecaster>)   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8  SkrubLearner(data_op=<Apply TemporalForecaster>)   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8  SkrubLearner(data_op=<Apply TemporalForecaster>)   

                                                                    name  \
  id                                                                       
0 01a1c051-f82c-7243-b346-6ec220bedc67  02_emotional_carryover__ablation   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8      02_emotional_carryover__full   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8     02_emotional_carryover__refit   

                                                                                                                                                                                                                           local_path  \
  id                                                                                                                                                                                                                                    
0 01a1c051-f82c-7243-b346-6ec220bedc67  /out/temporal/reports/projects/habs-emotional-carryover/reports/2026-11-09T12-39-19.853034+00-00__id_01a1c051-f82c-7243-b346-6ec220bedc67__cross-validation__02_emotional_carryover__ablation   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8      /out/temporal/reports/projects/habs-emotional-carryover/reports/2026-11-09T12-40-58.820025+00-00__id_01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8__cross-validation__02_emotional_carryover__full   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8     /out/temporal/reports/projects/habs-emotional-carryover/reports/2026-11-09T12-42-38.784937+00-00__id_01a1c055-0140-7558-ba9d-1ffddcbf84e8__cross-validation__02_emotional_carryover__refit   

                                                                   date  \
  id                                                                      
0 01a1c051-f82c-7243-b346-6ec220bedc67 2026-11-09 12:39:19.853034+00:00   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8 2026-11-09 12:40:58.820025+00:00   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8 2026-11-09 12:42:38.784937+00:00   

                                                                     key  \
  id                                                                       
0 01a1c051-f82c-7243-b346-6ec220bedc67  02_emotional_carryover__ablation   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8      02_emotional_carryover__full   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8     02_emotional_carryover__refit   

                                                                 dataset  \
  id                                                                       
0 01a1c051-f82c-7243-b346-6ec220bedc67  a436f2cd3f24b5f6daf72d3e9657df2a   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8  a436f2cd3f24b5f6daf72d3e9657df2a   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8  a436f2cd3f24b5f6daf72d3e9657df2a   

                                        default_score  fit_time       mae  \
  id                                                                        
0 01a1c051-f82c-7243-b346-6ec220bedc67       0.045424  5.859654  1.215195   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8       0.036770  5.992246  1.221250   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8       0.043004  5.932466  1.216713   

                                            mape  participant_macro_mae  \
  id                                                                      
0 01a1c051-f82c-7243-b346-6ec220bedc67  2.294760               1.215406   
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8  2.271597               1.220952   
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8  2.298757               1.216828   

                                        predict_time        r2      rmse  
  id                                                                      
0 01a1c051-f82c-7243-b346-6ec220bedc67      5.873243  0.045424  1.588795  
1 01a1c053-7ac3-7e68-8ab2-2e5dd3d753f8      5.856397  0.036770  1.596061  
2 01a1c055-0140-7558-ba9d-1ffddcbf84e8      5.963004  0.043004  1.590914  
```

## Cell 6: `# %%`

```python
frame = summary.frame().reset_index()
id_column = "id" if "id" in frame else "report_id"
verified, checks, metrics = {}, {}, {}
for mode in ("ablation", "full", "refit"):
    rows = frame.loc[frame["key"] == "02_emotional_carryover__" + mode]
    assert len(rows) == 1
    report = project.get(str(rows.iloc[0][id_column]))
    grid, target = report.X, np.asarray(report.y)
    prediction = producer._oof_predictions(report, len(grid))
    groups = grid["participant_id"].to_numpy()
    people = sorted(set(groups))
    errors = producer._person_errors(grid, target, prediction, people)
    assert (
        np.max(
            np.abs(
                errors - result["prediction"]["models"][mode]["anonymous_person_errors"]
            )
        )
        < 1e-10
    )
    for child, (_, test) in zip(report.reports_, report.split_indices, strict=True):
        registered = float(child.metrics.get("participant_macro_mae"))
        reconstructed = participant_macro_mae(
            target[test], prediction[test], groups=groups[test]
        )
        assert abs(registered - reconstructed) < 1e-10
    verified[mode] = {
        "rows": len(grid),
        "people": len(people),
        "all_finite": bool(np.isfinite(prediction).all()),
        "participant_macro_mae": float(errors.mean()),
    }
    metrics[mode] = report.metrics.summarize().frame()
    try:
        checks[mode] = report.checks.summarize().frame()
    except (ValueError, TypeError, AttributeError, FileNotFoundError) as error:
        checks[mode] = {"status": "generic_checks_unavailable", "reason": str(error)}
audit_state["predictions"] = set(verified) == {"ablation", "full", "refit"}
verified
```

**stdout:**
```






























Out[0]: 
{'ablation': {'rows': 2260,
  'people': 53,
  'all_finite': True,
  'participant_macro_mae': 1.212365638382822},
 'full': {'rows': 2260,
  'people': 53,
  'all_finite': True,
  'participant_macro_mae': 1.2179046253244266},
 'refit': {'rows': 2260,
  'people': 53,
  'all_finite': True,
  'participant_macro_mae': 1.2137791776748255}}
```

**stderr:**
```
/opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/skore/_checks/skd008_correlated_features.py:54: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  corr_statistic = spearmanr(X.to_numpy()).statistic
/opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/skore/_checks/skd008_correlated_features.py:54: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  corr_statistic = spearmanr(X.to_numpy()).statistic
/opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/skore/_checks/skd008_correlated_features.py:54: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  corr_statistic = spearmanr(X.to_numpy()).statistic
```

## Cell 7: `# %% [markdown]`

## Metrics summary

## Cell 8: `# %%`

```python
metrics
```

**stdout:**
```
Out[0]: 
{'ablation':                        skrublearner_mean  skrublearner_std
 metric                                                    
 participant_macro_mae           1.215406          0.084693
 default_score                   0.045424          0.052591
 r2                              0.045424          0.052591
 rmse                            1.588795          0.168235
 mae                             1.215195          0.085628
 mape                            2.294760          0.966683
 fit_time                        5.859654          0.030336
 predict_time                    5.873243          0.017837,
 'full':                        skrublearner_mean  skrublearner_std
 metric                                                    
 participant_macro_mae           1.220952          0.084777
 default_score                   0.036770          0.050991
 r2                              0.036770          0.050991
 rmse                            1.596061          0.169057
 mae                             1.221250          0.084834
 mape                            2.271597          0.832604
 fit_time                        5.992246          0.024940
 predict_time                    5.856397          0.013314,
 'refit':                        skrublearner_mean  skrublearner_std
 metric                                                    
 participant_macro_mae           1.216828          0.085336
 default_score                   0.043004          0.053050
 r2                              0.043004          0.053050
 rmse                            1.590914          0.170028
 mae                             1.216713          0.086312
 mape                            2.298757          0.974026
 fit_time                        5.932466          0.026000
 predict_time                    5.963004          0.017630}
```

## Cell 9: `# %% [markdown]`

## Checks summary

## Cell 10: `# %%`

```python
checks
```

**stdout:**
```
Out[0]: 
{'ablation': {'status': 'generic_checks_unavailable',
  'reason': 'array must be at least 2-d'},
 'full': {'status': 'generic_checks_unavailable',
  'reason': 'array must be at least 2-d'},
 'refit': {'status': 'generic_checks_unavailable',
  'reason': 'array must be at least 2-d'}}
```

## Cell 11: `# %% [markdown]`

## Paired forecast intervals and prespecified sensitivities

## Cell 12: `# %%`

```python
for name, reference in (
    ("full_vs_ablation", "ablation"),
    ("full_vs_refitted_restriction", "refit"),
):
    a = np.asarray(result["prediction"]["models"][reference]["anonymous_person_errors"])
    b = np.asarray(result["prediction"]["models"]["full"]["anonymous_person_errors"])
    actual = producer._paired(
        a,
        b,
        frozen["settings"]["seed"],
        frozen["settings"]["predictive_bootstrap_draws"],
    )
    expected = result["prediction"][name]
    assert abs(actual["gain"] - expected["gain"]) < 1e-12
    assert np.max(np.abs(np.asarray(actual["ci95"]) - expected["ci95"])) < 1e-12
audit_state["paired"] = True
assert all(audit_state.values())
completion = {
    "status": "complete_verified",
    "primary_verified": all(audit_state.values()),
    "verification_stages": audit_state,
    "audited_result_sha256": hashlib.sha256(
        (PUBLIC / "temporal_results.json").read_bytes()
    ).hexdigest(),
    "audited_frozen_sha256": hashlib.sha256(
        (PUBLIC / "temporal_frozen.json").read_bytes()
    ).hexdigest(),
}
completion_path = Path("scratch/audit/02_emotional_carryover/completion.json")
completion_path.parent.mkdir(parents=True, exist_ok=True)
completion_path.write_text(json.dumps(completion, indent=2))
{
    **completion,
    "prediction": result["prediction"],
    "sensitivities": result["sensitivities"],
    "figure_count": len(list((PUBLIC / "figures").glob("figure*.png"))),
    "scope": result["scope"],
}
```

**stdout:**
```
Out[0]: 
{'status': 'complete_verified',
 'primary_verified': True,
 'verification_stages': {'source': True,
  'likelihood': True,
  'predictions': True,
  'paired': True},
 'audited_result_sha256': '32d84ed0de2a4487a6cb6f1896ed0f4926b890250496962adbf00cf662f90c6c',
 'audited_frozen_sha256': '7f9054aa4b4d7a2dd652ea837ecdcd54cc5aa967e9965bc90cd8b6d7357b640f',
 'prediction': {'people': 53,
  'trials': 2260,
  'models': {'ablation': {'participant_macro_mae': 1.212365638382822,
    'fold_registered_macro_mae': [1.182983738387153,
     1.1454167767794468,
     1.1566579455605739,
     1.3521306310254988,
     1.2398429456035682],
    'anonymous_person_errors': [1.275623183377335,
     1.4941346559404882,
     0.8675805142549596,
     1.2120245634220046,
     1.1099409561036517,
     1.122779526752939,
     1.4211029379723235,
     1.2583964662052058,
     1.323830599049746,
     0.837405646874741,
     0.807255753574603,
     1.0580067639561879,
     1.0210484452289068,
     1.2437399432853333,
     0.8323074544604716,
     1.141988039475614,
     1.029179176138735,
     1.0386048120126594,
     1.3421178907914217,
     1.3620687309905226,
     0.7696574446945156,
     1.029996482493373,
     1.0340551067600394,
     0.94268011817047,
     0.9001118850859169,
     0.9728698637646959,
     0.7260850861201867,
     0.9575031415730266,
     0.9758747047089458,
     0.9164548092310106,
     1.4604494007924045,
     1.6407376123513941,
     0.8167371054184293,
     1.1050992882611612,
     1.0219003508741753,
     1.0572050560178403,
     1.0424810842971362,
     1.400236025243528,
     1.3521075048717024,
     1.4351475342955873,
     1.5527457204346877,
     2.411955144407988,
     3.670941998858885,
     1.143286565558931,
     1.4484647404575566,
     1.1905945303409988,
     1.2122605470872658,
     1.2783451261081629,
     1.223858251697802,
     1.3421713531180575,
     1.171428800660911,
     1.1901652835968366,
     1.0626351070681044]},
   'full': {'participant_macro_mae': 1.2179046253244266,
    'fold_registered_macro_mae': [1.190710007055352,
     1.1453468957587931,
     1.165304452353823,
     1.3563842389001053,
     1.2470127846345924],
    'anonymous_person_errors': [1.2574301056307515,
     1.4811364359668153,
     0.8892426099287852,
     1.210411131353409,
     1.1083342407700298,
     1.1223934220206868,
     1.4998434822106417,
     1.263139829779824,
     1.3338224458716532,
     0.8330516997964879,
     0.8788156105705331,
     1.0370969821188591,
     1.063272042092161,
     1.2091519299037896,
     0.8587456858970894,
     1.1575252783012917,
     1.0396165207661112,
     1.0395586258877034,
     1.347588253058698,
     1.3712241871579463,
     0.812599923697703,
     1.0312610891959344,
     1.0640203384232474,
     0.9430929532575821,
     0.8945622759226626,
     0.9724795300487852,
     0.7199287823632745,
     0.9543324049066858,
     0.9491103129579525,
     0.915316145751994,
     1.4617960404461754,
     1.6452438081284309,
     0.8124910334443533,
     1.1004603663729342,
     1.0338334693710918,
     1.0472098314873275,
     1.0465124131720962,
     1.4392319781414329,
     1.287698445678057,
     1.449838728177037,
     1.5628101517489599,
     2.4146604148504034,
     3.713960436288997,
     1.1438068322511055,
     1.4462200589359038,
     1.171121759255451,
     1.2305465504870112,
     1.2964894392680253,
     1.2351847720657259,
     1.359505951877391,
     1.1238930362343316,
     1.1907504745554802,
     1.0775748743478104]},
   'refit': {'participant_macro_mae': 1.2137791776748255,
    'fold_registered_macro_mae': [1.1843633325163871,
     1.1460917560476866,
     1.1584378765694479,
     1.3550971847023536,
     1.2401501953273482],
    'anonymous_person_errors': [1.2735397768900563,
     1.4931858264618603,
     0.8663994143350545,
     1.2117689763741843,
     1.1098978519813771,
     1.1212343117476888,
     1.4074095530289032,
     1.2555577894251497,
     1.3224585701532061,
     0.8364041909461473,
     0.8133859185292935,
     1.0573849429717568,
     1.0214419903422742,
     1.2425082188949121,
     0.8312779453294448,
     1.144702250220687,
     1.028459826833391,
     1.0384011137805147,
     1.3326728984313578,
     1.3613066794872746,
     0.769625146217799,
     1.0254305330867135,
     1.0361122679189831,
     0.9426902749423496,
     0.9000053275714852,
     0.9731799230319382,
     0.7262421227277043,
     0.9576507875258348,
     0.9796074923065329,
     0.9165573743448026,
     1.463075361839518,
     1.6426168215285253,
     0.8170071541735289,
     1.1124010721906281,
     1.0283035686229534,
     1.0572714607161162,
     1.04299201101943,
     1.4073060827398016,
     1.3702713245364662,
     1.4396850497915,
     1.5547399709668202,
     2.4143879885837385,
     3.7023273564680372,
     1.1431894479477138,
     1.4489952362246803,
     1.2025722103815841,
     1.2202211998006696,
     1.2797724081739739,
     1.2199896745842107,
     1.3429045635617525,
     1.1623536738397378,
     1.1908982511541573,
     1.0705152320815314]}},
  'full_vs_ablation': {'gain': -0.005538986941604621,
   'ci95': [-0.012251070631019042, 0.000938389896886013],
   'people_helped': 21,
   'people': 53,
   'bootstrap_draws': 10000},
  'full_vs_refitted_restriction': {'gain': -0.004125447649601302,
   'ci95': [-0.011034453141787632, 0.0025344335236614605],
   'people_helped': 22,
   'people': 53,
   'bootstrap_draws': 10000},
  'scope': 'Grouped one-step J2 prediction with already observed predecessor responses and J1-only personal calibration.',
  'macro_note': 'Global equal-person OOF error; parent fold-mean metrics can weight folds differently.'},
 'sensitivities': {'sampling_half': {'status': 'completed',
   'people': 53,
   'transitions': 4554,
   'covariance': 0.013231864664672391,
   'rho_point': 0.6901942792459794,
   'rho_profile95_grid': [0.0, 0.995],
   'correlation_weakly_identified': False,
   'random_effect_sd': [0.1118664077314503, 0.17137600033594272],
   'likelihood_ratio': 3.5377688088006494},
  'sampling_double': {'status': 'completed',
   'people': 53,
   'transitions': 4554,
   'covariance': 4.734776005286056e-10,
   'rho_point': 0.46000000000000013,
   'rho_profile95_grid': [-0.995, 0.995],
   'correlation_weakly_identified': True,
   'random_effect_sd': [5.9139489011858477e-05, 1.7404599680892286e-05],
   'likelihood_ratio': 1.7001525264959128e-05},
  'without_count_moderator': {'status': 'completed',
   'people': 53,
   'transitions': 4554,
   'covariance': 0.0027546141232259393,
   'rho_point': 0.9950547536867305,
   'rho_profile95_grid': [-0.995, 0.9950547536867305],
   'correlation_weakly_identified': True,
   'random_effect_sd': [0.0367534996120835, 0.07532082925255983],
   'likelihood_ratio': 0.2991059864809813},
  'hc1_no_serial_kernel': {'status': 'completed',
   'people': 53,
   'transitions': 4554,
   'covariance': 0.009355013682997218,
   'rho_point': 0.9950547536867305,
   'rho_profile95_grid': [-0.995, 0.9950547536867305],
   'correlation_weakly_identified': True,
   'random_effect_sd': [0.07250243463901039, 0.12967159647684878],
   'likelihood_ratio': 2.2074514435856827},
  'minimum_30': {'status': 'completed',
   'people': 46,
   'transitions': 4105,
   'covariance': 0.003957927921244953,
   'rho_point': 0.9950547536867305,
   'rho_profile95_grid': [-0.995, 0.9950547536867305],
   'correlation_weakly_identified': True,
   'random_effect_sd': [0.041643467458302975, 0.09551553620467773],
   'likelihood_ratio': 0.44286669932418476},
  'common_j1_scales': {'status': 'completed',
   'people': 53,
   'transitions': 4554,
   'covariance': 0.005658005625586766,
   'rho_point': 0.9950547536867305,
   'rho_profile95_grid': [-0.995, 0.9950547536867305],
   'correlation_weakly_identified': True,
   'random_effect_sd': [0.05168662885876092, 0.11001152598657031],
   'likelihood_ratio': 0.8423920911143057},
  'stable_j1_scales': {'status': 'identical_primary_grid_and_scales',
   'people': 53,
   'transitions': 4554},
  'j1_gap_iqr': {'status': 'completed',
   'people': 17,
   'transitions': 969,
   'covariance': -9.986244327312669e-08,
   'rho_point': -0.1399999999999999,
   'rho_profile95_grid': [-0.995, 0.995],
   'correlation_weakly_identified': True,
   'random_effect_sd': [0.0006912677400244993, 0.0010318768328626755],
   'likelihood_ratio': 8.867111631616353e-07}},
 'figure_count': 5,
 'scope': 'Exploratory study after experiment-01 outcomes were opened; no independent confirmatory holdout is claimed.'}
```
