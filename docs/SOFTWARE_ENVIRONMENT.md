# Software environment notes

The released modeling scripts depend on NumPy, pandas, SciPy, scikit-learn, joblib, openpyxl, XGBoost, LightGBM, CatBoost, and PyTorch. The full 30-algorithm workflow requires the booster and deep-learning packages listed in `environment/requirements-lock.txt`.

`environment/environment_report.json` records the environment in which the released scripts were verified. `environment/requirements-lock.txt` provides the captured package versions used for that verification environment.

The captured environment is a reproducibility record for the released code. It should not be interpreted as proof that the exact package build, operating-system patch level, CPU/GPU implementation, or numerical backend of every historical archived model fit was preserved.

For a new reproduction, use a compatible Python 3.8 environment, install the locked dependencies, validate the package, and record any unavoidable version substitutions in the resulting analysis report.
