# Software environment notes

The released modeling scripts depend on NumPy, pandas, SciPy, scikit-learn, joblib, openpyxl, XGBoost, LightGBM, CatBoost, and PyTorch. The full 30-algorithm workflow requires the optional booster/deep-learning packages to be installed.

`requirements.txt` and `environment.yml` define a **recommended reproduction environment**, not a claim that the exact historical package versions used for all archived model fits were preserved.

For the strongest provenance record, run:

```bash
python environment/capture_environment.py
```

from the environment in which the released code is verified. Commit the resulting `environment_report.json` and `requirements-lock.txt` to the repository. Both files are normally only a few kilobytes.

The capture script intentionally avoids environment variables, usernames, full filesystem paths, and the complete `pip freeze` inventory to reduce accidental disclosure of private information.
