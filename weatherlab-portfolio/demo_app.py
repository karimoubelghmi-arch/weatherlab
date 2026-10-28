"""Offline portfolio demo using the original WeatherLab statistical engine."""
import numpy as np
import pandas as pd
import streamlit as st
from demo_data import make_dataset
from weatherlab.engine import PipelineConfig, RegConfig, LawConfig, run_pipeline, pdf_T

st.set_page_config(page_title="WeatherLab portfolio demo", layout="wide")
st.title("WeatherLab")
st.caption("Probabilistic weather modelling • offline portfolio demonstration")
st.info("All data and market prices below are synthetic. No API connection, credentials or order execution.")
method = st.sidebar.selectbox("Regression", ["OLS", "Ridge", "BEST"])
family = st.sidebar.selectbox("Residual distribution", ["Normal", "Student-t", "Laplace"])
fraction = st.sidebar.slider("Fraction of Kelly", 0.0, 1.0, .25, .05)
cap = st.sidebar.slider("Maximum hypothetical allocation", 0.0, .10, .02, .005)
frame, models = make_dataset()
train = frame.iloc[:-1].copy()
target = {m: float(frame.iloc[-1][m]) for m in models}
cfg = PipelineConfig(reg=RegConfig(method=method, use_err_prev=False), law=LawConfig(family=family))
@st.cache_data
def calculate(method, family):
    cfg = PipelineConfig(reg=RegConfig(method=method, use_err_prev=False), law=LawConfig(family=family))
    return run_pipeline(train, models, "C", cfg, target_forecasts=target, target_doy=int(frame.iloc[-1]["doy"]))
ctx = calculate(method, family)
a,b,c = st.columns(3)
a.metric("Predicted location", f"{ctx['mu']:.2f} °C")
b.metric("Scale parameter", f"{ctx['sigma']:.2f} °C")
c.metric("Historical training rows", len(train))
st.subheader("Historical synthetic observations and forecasts")
st.line_chart(frame.set_index("date")[["tmax_c"] + models].tail(90))
st.subheader("Predictive density")
x = np.linspace(ctx["mu"]-6, ctx["mu"]+6, 250)
y = pdf_T(x, ctx["mu"], ctx["sigma"], ctx["dist"])
st.line_chart(pd.DataFrame({"Temperature °C": x, "Density": y}).set_index("Temperature °C"))
st.subheader("Illustrative probability versus price")
threshold = st.slider("Temperature threshold °C", float(ctx["mu"]-4), float(ctx["mu"]+4), float(ctx["mu"]))
prob = float(ctx["dist"].cdf((threshold-ctx["mu"])/ctx["sigma"]))
price = st.slider("Synthetic YES price", .01, .99, .40, .01)
edge = prob-price
allocation = min(cap, fraction * max(0.0, edge/(1-price)))
st.write({"Event": "Temperature at or below threshold", "Model probability": round(prob,4), "Synthetic price": price, "Probability minus price": round(edge,4), "Capped fractional Kelly allocation": round(allocation,4)})
st.caption("Illustration only: no fees, slippage, liquidity constraints or correlated-position risk. Synthetic examples do not establish predictive accuracy or profitability. The scale parameter is not necessarily the distribution standard deviation.")
with st.expander("Design and validation responsibilities"):
    st.write("Karim defined the objective, statistical approach and workflow, instructed AI tools to implement the software, and manually checked data and statistical results. Code implementation was generated with AI assistance. This public demo is a curated subset; live integrations are preserved separately as reference source.")
