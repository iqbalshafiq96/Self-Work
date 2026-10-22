# Refinery Production Planning Optimizer – interactive flow

A Streamlit app that shows the planning pipeline as connected blocks with flowing particles
(inputs → validate → process model → LP optimizer → outputs). Click any block to open it:

| Block | Click to… |
|---|---|
| 7 input blocks (CSV) | **Edit** values in a table → *Apply & re-optimize* |
| Validate & build | See data checks and errors |
| Process model | See product equations + margin per barrel of each decision |
| **LP optimizer** | See the objective function (live coefficients), constraints C1–C16, run settings (rolling horizon, soft tank limits) |
| Decision variables | x[c,t], r[t], f[t] per day + crude slate chart |
| Plan economics | Margin, revenue, cost, product make – current vs optimised |
| Tank balance | Opening + inflow − outflow = closing per tank and day, C14 check |
| Tank level curves | Closing inventory vs min heel, capacity and end target for all tanks |
| Why? | Binding constraints and shadow prices in plain language |

Data are loaded at start-up from the GitHub folder (CSV files directly in the folder or in `data/`);
any file that can't be fetched falls back to the bundled `data/` copy. **Sample data are illustrative only.**

## Files
```
app.py                    Streamlit host: loads data, runs the solver, talks to the canvas
flow_component/index.html The interactive canvas (HTML/CSS/vanilla JS, no external libraries)
data_loader.py            Read CSVs (GitHub or local), validate, build day-by-day arrays
model.py                  Linear process model, economics, tank balance, constraint checks
optimizer.py              LP (SciPy HiGHS), binding constraints, shadow prices, explanation
run_optimizer.py          Same optimizer from the command line / Excel VBA (RunOptimizer.bas)
data/*.csv                Fallback copy of the input CSVs
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to GitHub (e.g. `Self-Work/Refinery_Optimizer/`).
2. share.streamlit.io → **Create app** → repo `Self-Work`, branch `main`, main file `Refinery_Optimizer/app.py`.
3. Optional: point to another data folder with a secret `DATA_URL = "https://github.com/<user>/<repo>/tree/main/<folder>"`.

Edits made in the app live only in that browser session. To make a scenario the new base case,
use **⬇ Edited input CSVs** and commit the files to GitHub.
