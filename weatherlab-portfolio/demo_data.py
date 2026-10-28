"""Synthetic data only. No real observations, forecasts or trading history."""
import numpy as np
import pandas as pd

def make_dataset(n=500):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n)
    doy = dates.dayofyear.to_numpy()
    seasonal = 16 + 9 * np.sin(2 * np.pi * (doy - 90) / 365.25)
    common = rng.normal(0, 1.2, n)
    frame = pd.DataFrame({"date": dates, "doy": doy})
    models = ["forecast_a", "forecast_b", "forecast_c"]
    for idx, model in enumerate(models):
        frame[model] = seasonal + common + rng.normal(0, 0.8 + idx * .25, n) + idx * .2
    frame["tmax_c"] = seasonal + common + rng.normal(0, 1.1, n)
    frame["err_prev"] = (frame["tmax_c"] - frame[models].mean(axis=1)).shift(1).fillna(0)
    return frame, models
