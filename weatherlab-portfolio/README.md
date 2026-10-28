# WeatherLab

**A personal project connecting probabilistic weather forecasts, prediction-market prices and automated decision rules.**

Designed and quantitatively reviewed by **Karim Belghmi**, MSc in Actuarial Science candidate at HEC Lausanne. All master’s coursework passed; thesis defence planned for January 2027.

## What the original project does

The original application collects weather forecasts through APIs, fits statistical models, estimates probabilities for temperature intervals and compares them with prediction-market prices. It includes fractional Kelly sizing, exposure limits, scheduled workflows, interactive analysis and decision journals.

```text
Weather forecasts → statistical model → predictive distribution
                                            ↓
Market prices → probability comparison → position sizing → audit trail
```

The timing of live forecasts must be consistent with the forecast horizon used for training. The project includes work on forecast freshness, local time zones and traceability of calculations.

## My contribution and use of AI

I defined the project objective, proposed the statistical approach and specified the workflows. AI tools generated the implementation from my instructions. I manually checked collected data and statistical outputs, identified issues including an unsuitable data API and a regression implementation error, and directed corrections. This is an AI-assisted development project, not a claim that I manually authored every line of code.

## Run the offline demonstration

Python 3.11 or 3.12 is recommended. From the repository folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run demo_app.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

The demonstration uses **generated synthetic data and simulated market prices only**. No API keys, wallet, network data requests or real orders are needed. It uses the original statistical engine and a new simplified interface prepared for portfolio presentation. It does not reproduce the full original application.

## Repository contents

- `demo_app.py`: interactive offline demonstration.
- `demo_data.py`: reproducible synthetic dataset generator.
- `weatherlab/engine.py`: statistical engine extracted from the original project.
- `reference_source/wlab/`: selected original modules showing data access, market scanning, calibration, journals and analysis. These modules are for code review, are not a standalone runnable application, and some include optional network calls. The demo does not import them.
- `PROJECT_NOTES.md`: scope, exclusions and limitations.

## Methods

OLS/Ridge regression and best-model selection; spread-dependent predictive scale; residual distributions; temperature interval probabilities; probability/price comparisons; fractional Kelly position sizing.

The full source contains validation utilities and historical diagnostic comments. Their numeric claims have **not been independently reproduced for this public release**. The demo is an illustration, not evidence of live forecasting performance or profitability. Whole-training-sample results must not be confused with held-out validation.

## Scope

This repository presents a personal quantitative and automation project. The active original workflow is primarily deterministic automation. An experimental LLM guardrail was archived in the original project and is not presented here as a deployed agent. No MCP implementation is claimed.

No original trading history, account addresses, private settings, historical weather datasets or order-execution components are included. No open-source licence has been selected for this initial portfolio release; public visibility alone does not grant a general reuse licence.
