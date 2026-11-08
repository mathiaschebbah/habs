# Cells: `audit/01_aversive_reactivity.py`

## Cell 0: `# %% [markdown]`

# Reserved aversive-reactivity reports
#
Read the saved cohort-2 predictions and metrics without training or modifying
the reports. The output tree is restored at its original absolute location.

## Cell 1: `# %%`

```python
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skore

from habs import PROJECT_ROOT
from habs.model import INPUT_COLUMNS

out_dir = Path(os.environ["HABS_ANALYSIS_OUT"]).resolve()
project = skore.Project(
    "habs-aversive-reactivity",
    mode="local",
    workspace=str((out_dir / "reports").resolve()),
)
summary = project.summarize().frame().reset_index()
summary[["id", "key", "report_type"]]
```

**stdout:**
```
Out[0]: 
                                      id  \
0   01a1bd72-24db-7395-b591-d8aef6ce725b   
1   01a1bd72-26b6-7a73-8d16-d146cee70a18   
2   01a1bd72-28d6-7cf4-bb9d-d023526b89a8   
3   01a1bd72-2bd1-7105-b1ee-ec9aaceb5e3e   
4   01a1bd72-2e98-7f47-9164-5fdcfc7ecbc8   
5   01a1bd72-3169-7437-a5e8-22edafa27377   
6   01a1bd72-3480-7ac3-b12b-51ca56c4f2b4   
7   01a1bd72-3748-7b69-bc08-fbf37fe6b76b   
8   01a1bd72-3a4d-79ae-9d6f-304f7aca9444   
9   01a1bd72-3daf-7636-9fd2-9adb9898755d   
10  01a1bd72-40f5-7efb-8269-5d44edd58565   

                                                    key report_type  
0                   01_aversive_reactivity__cohort2__M1   estimator  
1                   01_aversive_reactivity__cohort2__M0   estimator  
2                   01_aversive_reactivity__cohort2__M2   estimator  
3                   01_aversive_reactivity__cohort2__M3   estimator  
4                   01_aversive_reactivity__cohort2__M4   estimator  
5          01_aversive_reactivity__cohort2__missingness   estimator  
6               01_aversive_reactivity__cohort2__no_EDA   estimator  
7         01_aversive_reactivity__cohort2__history_only   estimator  
8            01_aversive_reactivity__cohort2__nonlinear   estimator  
9                   01_aversive_reactivity__cohort2__M5   estimator  
10  01_aversive_reactivity__cohort2__continuous_profile   estimator  
```

## Cell 2: `# %% [markdown]`

## Input and parameter provenance

## Cell 3: `# %%`

```python
frozen = json.loads((out_dir / "frozen.json").read_text())
evidence = json.loads((out_dir / "public/study_summary.json").read_text())
source_files = sorted((PROJECT_ROOT / "habs").glob("*.py")) + [
    PROJECT_ROOT / "experiments/01_aversive_reactivity.py",
    PROJECT_ROOT / "run_config.json",
]
current_code = hashlib.sha256(
    b"".join(p.name.encode() + p.read_bytes() for p in source_files)
).hexdigest()
assert current_code == frozen["code_sha256"]
assert (
    hashlib.sha256((out_dir / "frozen.json").read_bytes()).hexdigest()
    == evidence["frozen_sha256"]
)
assert frozen["status"] == "frozen_before_reserved_scoring"
assert not frozen["selected"]["M4"].get("conditional_arousal", False)
assert not {
    "rating_valence",
    "rating_arousal",
    "final_emotion_label",
    "aversive_score",
}.intersection(INPUT_COLUMNS)
{
    "source_matches_frozen": True,
    "freeze_digest_matches": True,
    "prospective_response_columns_excluded": True,
    "allocation_sha256": frozen["feature_manifest"]["allocation_sha256"],
    "labels_sha256": frozen["feature_manifest"]["labels_sha256"],
    "test_grid_sha256": frozen["test_grid_sha256"],
}
```

**stdout:**
```
Out[0]: 
{'source_matches_frozen': True,
 'freeze_digest_matches': True,
 'prospective_response_columns_excluded': True,
 'allocation_sha256': '6717f50a5c86658a35f175417c463fca9f2c46ba5b98c3c1bdb11edab52f9027',
 'labels_sha256': 'c981a6836dfcd6af2b4c68ca3fab0283453f934522217b7b5c66e03119430fcd',
 'test_grid_sha256': '2ddb698938aeaef51ac85f56373e1a51d364738a74b1b00e8233de717bea4dfd'}
```

## Cell 4: `# %% [markdown]`

## Reloaded participant-macro metrics

## Cell 5: `# %%`

```python
reports = {}
verified = []
for name in frozen["selected"]:
    key = "01_aversive_reactivity__cohort2__" + name
    matching = summary.loc[summary["key"] == key]
    if len(matching) == 0:
        raise KeyError(f"Missing saved report for {key}")
    report = project.get(str(matching.iloc[0]["id"]))
    prediction = np.asarray(report.get_predictions(data_source="test"))
    assert np.isfinite(prediction).all()
    assert len(prediction) == evidence["primary"]["test_trials"]
    macro = float(report.metrics.get("participant_macro_mae"))
    expected = evidence["primary"]["models"][name]["macro_mae"]
    assert abs(macro - expected) < 1e-10
    reports[name] = report
    verified.append(
        {
            "model": name,
            "participant_macro_mae": macro,
            "predictions": len(prediction),
            "matches_export": True,
        }
    )
pd.DataFrame(verified)
```

**stdout:**
```
Out[0]: 
                 model  participant_macro_mae  predictions  matches_export
0                   M1               0.112443          997            True
1                   M0               0.115680          997            True
2                   M2               0.111879          997            True
3                   M3               0.113070          997            True
4                   M4               0.112826          997            True
5          missingness               0.112259          997            True
6               no_EDA               0.112597          997            True
7         history_only               0.112694          997            True
8            nonlinear               0.112631          997            True
9                   M5               0.112825          997            True
10  continuous_profile               0.112814          997            True
```

## Cell 6: `# %% [markdown]`

## Checks summary

## Cell 7: `# %%`

```python
reports["M4"].checks.summarize().frame()
```

**stdout:**
```

Out[0]: 
      code                                                title  \
0   SKD001                                Potential overfitting   
1   SKD002                               Potential underfitting   
2   SKD003               Inconsistent performance across splits   
3   SKD004                                 High class imbalance   
4   SKD005                             Underrepresented classes   
5   SKD006                           Coefficient interpretation   
6   SKD007             MDI biased for high-cardinality features   
7   SKD008                     Highly correlated input features   
8   SKD009  Model performance vs. HistGradientBoosting baseline   
9   SKD010                           Model slower than baseline   
10  SKD011                                       Golden feature   
11  SKD012                                     Useless features   
12  SKD013                    Train-test overlap in time series   
13  SKD014                       Hyperparameters at search edge   
14  SKD015                         Hyperparameters worth tuning   
15  SKD016                                  Estimator not tuned   

           section  \
0   not_applicable   
1   not_applicable   
2   not_applicable   
3   not_applicable   
4   not_applicable   
5   not_applicable   
6   not_applicable   
7   not_applicable   
8   not_applicable   
9   not_applicable   
10  not_applicable   
11  not_applicable   
12  not_applicable   
13  not_applicable   
14  not_applicable   
15  not_applicable   

                                                                                  explanation  \
0                                                                  Train data is unavailable.   
1                                                                  Train data is unavailable.   
2                                                        Not applicable to estimator reports.   
3                                       ML task is not binary classification. Got regression.   
4                                   ML task is not multiclass classification. Got regression.   
5                      Estimator is not a linear model: it does not have a `coef_` attribute.   
6   Estimator is not a tree-based model: it does not have a `feature_importances_` attribute.   
7                                                                  Train data is unavailable.   
8                                                                  Train data is unavailable.   
9                                                                  Train data is unavailable.   
10                                                                 Train data is unavailable.   
11                                                  Failed to compute permutation importance.   
12                                                                 Train data is unavailable.   
13                                Estimator is not a BaseSearchCV instance. Got SkrubLearner.   
14                                Estimator is not a BaseSearchCV instance. Got SkrubLearner.   
15                                                                 No parameter to recommend.   

                                                                                          documentation_url  
0                    https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd001-overfitting  
1                   https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd002-underfitting  
2       https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd003-inconsistent-performance  
3           https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd004-high-class-imbalance  
4       https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd005-underrepresented-classes  
5          https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd006-unscaled-coefficients  
6           https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd007-mdi-cardinality-bias  
7            https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd008-correlated-features  
8            https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd009-worse-than-baseline  
9           https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd010-slower-than-baseline  
10                https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd011-golden-feature  
11              https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd012-useless-features  
12       https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd013-train-test-time-overlap  
13    https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd014-hyperparams-at-search-edge  
14  https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd015-hyperparameters-worth-tuning  
15           https://docs.skore.probabl.ai/0.29/user_guide/automated_checks.html#skd016-estimator-not-tuned  
```

## Cell 8: `# %% [markdown]`

## Metrics summary

## Cell 9: `# %%`

```python
reports["M4"].metrics.summarize().frame()
```

**stdout:**
```
Out[0]: 
metric
participant_macro_mae    1.128258e-01
default_score                     NaN
r2                       4.818720e-01
rmse                     1.882699e-01
mae                      1.140501e-01
mape                     9.103547e+13
fit_time                          NaN
predict_time             2.102010e-01
Name: SkrubLearner, dtype: float64
```

**stderr:**
```
<ipython-input-1-fb6288f996b5>:1: UserWarning: Metric 'default_score' has failed: ValueError('Calibration must be a separate first-session-only table.')
  reports["M4"].metrics.summarize().frame()
```

## Cell 10: `# %% [markdown]`

## Paired uncertainty and interpretation boundaries

## Cell 11: `# %%`

```python
contrast = evidence["primary"]["comparisons"]["M1_minus_M4"]
reference = np.asarray(contrast["anonymous_reference_errors"], dtype=float)
candidate = np.asarray(contrast["anonymous_candidate_errors"], dtype=float)
assert len(reference) == evidence["primary"]["test_people"]
assert len(candidate) == len(reference)
assert np.isfinite(reference).all() and np.isfinite(candidate).all()
assert (
    abs(reference.mean() - reports["M1"].metrics.get("participant_macro_mae")) < 1e-10
)
assert (
    abs(candidate.mean() - reports["M4"].metrics.get("participant_macro_mae")) < 1e-10
)
difference = reference - candidate
rng = np.random.default_rng(frozen["seed"])
means = difference[
    rng.integers(0, len(difference), size=(10000, len(difference)))
].mean(axis=1)
interval = np.quantile(means, [0.025, 0.975])
assert abs(difference.mean() - contrast["delta_mae"]) < 1e-12
assert np.max(np.abs(interval - np.asarray(contrast["ci95"]))) < 1e-12
{
    "primary_verified": True,
    "people": len(difference),
    "reference_macro_mae": float(reference.mean()),
    "multimodal_macro_mae": float(candidate.mean()),
    "paired_gain": float(difference.mean()),
    "ci95": interval.tolist(),
    "scope": (
        "Internal released-identifier hold-out, fixed stimulus catalogue, "
        "experimental proxy."
    ),
}
```

**stdout:**
```
Out[0]: 
{'primary_verified': True,
 'people': 22,
 'reference_macro_mae': 0.11244304617514045,
 'multimodal_macro_mae': 0.11282579422390664,
 'paired_gain': -0.00038274804876618963,
 'ci95': [-0.0028766096502642006, 0.001822109922505052],
 'scope': 'Internal released-identifier hold-out, fixed stimulus catalogue, experimental proxy.'}
```
