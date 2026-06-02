# cquant_exercise

# Generate README.md for cquant_exercise repository

readme = """# cQuant Energy Analyst Programming Exercise
**Candidate:** Luis  
**Data:** ERCOT Day-Ahead Hourly Settlement Point Prices, 2016–2019  
**Language:** Python 3  

---

## Project Structure
cquant_exercise/
├── main.py                        # All analysis code (cell-per-task structure)
├── historicalPriceData/           # Input CSVs (4 files, 2016–2019)
│   ├── ERCOT_DA_Prices_2016.csv
│   ├── ERCOT_DA_Prices_2017.csv
│   ├── ERCOT_DA_Prices_2018.csv
│   └── ERCOT_DA_Prices_2019.csv
└── output/                        # All generated outputs
├── AveragePriceByMonth.csv
├── HourlyVolatilityByYear.csv
├── MaxVolatilityByYear.csv
├── OU_parameters_by_hub.csv
├── SettlementHubAveragePriceByMonth.png
├── LoadZoneAveragePriceByMonth.png
├── VolatilityByHub_BarChart.png
├── VolatilityByHub_Heatmap.png
├── OU_MeanReversion_Analysis.png
├── formattedSpotHistory/      # 15 spot_<node>.csv files
└── hourlyShapeProfiles/       # 15 profile_<node>.csv files

---

## Dependencies

```bash
pip install pandas numpy matplotlib scipy
```

Developed and tested on Python 3.14. No additional packages required.

---

## How to Run

All code is in `main.py`, structured as independent `#%%` cells that can be
run sequentially end-to-end or individually in VS Code / Jupyter.

```bash
python main.py
```

Each cell prints progress messages and validates its own output before saving.
Tasks must be run in order (1→2→3...) as each reads the previous task's output.

---

## Task Summary

### Task 1 — Load and Combine Data
Reads all 4 CSV files from `historicalPriceData/` using glob pattern matching,
concatenates them into a single DataFrame, parses timestamps, and validates
completeness. Saves `combined_ercot_da_prices.csv`.

**Data:** 497,320 rows × 3 columns (Date, SettlementPoint, Price)  
**Nodes:** 15 settlement points — 7 hubs (HB_), 8 load zones (LZ_)

---

### Task 2 — Monthly Average Prices
Computes the mean hourly price for each (SettlementPoint, YearMonth) pair.
Negative prices are retained — they are physically valid in ERCOT (wind
generators accept negative prices due to production tax credits).

**Output:** 681 rows (not 720 — see Missing Data note below)  
**Hub avg:** $29.60/MWh | **Load zone avg:** $30.74/MWh

---

### Task 3 — Write AveragePriceByMonth.csv
Reshapes Task 2 output into the required 4-column format:
`SettlementPoint, Year, Month, AveragePrice`.

**Output:** `AveragePriceByMonth.csv` — 681 rows × 4 columns

---

### Task 4 — Hourly Volatility by Hub and Year
Computes hourly price volatility as the standard deviation of log returns
for each hub and year. Zero and negative prices are filtered before computing
log returns (ln is undefined for P ≤ 0). Load zones excluded per instructions.

**Formula:** σ = std(ln(P_t / P_{t-1})) per hub-year  
**Rows removed (price ≤ 0):** 1,024 (0.472% of hub rows)

---

### Task 5 — Write HourlyVolatilityByYear.csv
Writes the Task 4 results to the required 3-column format:
`SettlementPoint, Year, HourlyVolatility` (column names are case-sensitive).

**Output:** `HourlyVolatilityByYear.csv` — 25 rows × 3 columns

---

### Task 6 — Write MaxVolatilityByYear.csv
Identifies the hub with the highest hourly volatility for each year using
`idxmax()` and extracts those rows. Same column format as Task 5.

| Year | Hub      | HourlyVolatility |
|------|----------|-----------------|
| 2016 | HB_SOUTH | 0.209026        |
| 2017 | HB_WEST  | 0.247628        |
| 2018 | HB_WEST  | 0.302512        |
| 2019 | HB_PAN   | 0.631635        |

---

### Task 7 — cQuant Model-Ready Format (formattedSpotHistory/)
Translates hourly data from long format into cQuant's wide format:
one row per day, 24 hourly price columns (X1–X24), one file per node.

**Convention:** Hour-beginning — X1 = 00:00, X2 = 01:00, ..., X24 = 23:00  
**Files:** 15 CSV files named `spot_<SettlementPoint>.csv`  
**Columns:** `Variable, Date, X1, X2, ..., X24` (26 columns)  
**Rows:** 1,461 days for full-history nodes; 270 days for HB_PAN

---

## Bonus Analyses

### Bonus 1 — Monthly Average Price Line Plots
Two line plots showing monthly average prices in chronological order:
- `SettlementHubAveragePriceByMonth.png` — all 7 HB_ hubs overlaid
- `LoadZoneAveragePriceByMonth.png` — all 8 LZ_ load zones overlaid

**Key finding:** LZ_WEST and HB_WEST spike dramatically above all other
nodes in summer 2018 (~$101/MWh) and 2019 (~$128/MWh), reflecting
West Texas transmission congestion during peak AC load periods.

---

### Bonus 2 — Volatility Comparison Plots
Two complementary visualizations of hub volatility across years:
- `VolatilityByHub_BarChart.png` — grouped bars by year, one per hub
- `VolatilityByHub_Heatmap.png` — hub × year matrix with color intensity

**Key finding:** Volatility increased year-over-year from 2017→2019 for
all hubs, consistent with rising renewable penetration in ERCOT.
HB_WEST is persistently the most volatile full-history hub.

---

### Bonus 3 — Hourly Shape Profiles (hourlyShapeProfiles/)
Normalized 24-hour price profiles for each settlement point, broken out
by month of year and day of week. Each profile averages exactly 1.0 across
the 24 hours (verified to machine epsilon: max error 4.44e-16).

**Files:** 15 CSV files named `profile_<SettlementPoint>.csv`  
**Format:** Wide — columns: `Month, DayOfWeek, H1, H2, ..., H24`  
**Rows:** 84 per full-history node (12 months × 7 days); 63 for HB_PAN  
**Normalization:** `shape[h] = avg_price[h] / mean(avg_price[1..24])`

---

### Bonus 4 — Ornstein-Uhlenbeck Mean Reversion Model
Fits the OU stochastic process to log prices for each hub using OLS
regression on the discrete-time SDE approximation. Recovers three
parameters: mean reversion speed (κ), long-run mean (μ), and volatility
(σ). Runs a 200-path Monte Carlo simulation for HB_BUSAVG over a 30-day
horizon and produces a 5-panel diagnostic plot.

**SDE:** dP_t = κ(μ - P_t)dt + σdW_t  
**Fitting method:** OLS on Δln(P_t) = a + b·ln(P_{t-1}) + ε  
**Key finding:** κ is extremely large (thousands per year), implying
half-lives of just a few hours — consistent with electricity being the
most mean-reverting commodity class.

**Outputs:** `OU_MeanReversion_Analysis.png`, `OU_parameters_by_hub.csv`

---

## Missing Data & Data Quality Notes

### 1. HB_PAN — Partial History
**What:** HB_PAN has only 9 months of data, all in 2019. It is absent
from 2016, 2017, and 2018 entirely.  
**How found:** Task 2 threw an AssertionError (expected 720 rows, got 681).
A diagnostic cell built a complete expected grid of every node × every month
and left-joined it against the actual data, isolating HB_PAN as the only
node with missing months (exactly 39 missing).  
**Why:** HB_PAN (Panhandle hub) was added to ERCOT's settlement system
partway through the data window. It represents the West Texas Panhandle
region — an area with extreme wind generation capacity and significant
transmission constraints.  
**Decision:** Never filled or dropped. Carried through transparently with
adjusted assertions in every downstream task. The model operator is
responsible for deciding how to handle partial-history nodes.

### 2. DST Spring-Forward Days — Missing Hour
**What:** 4 days per year (second Sunday of March) have only 23 hours.
The 02:00–03:00 hour does not exist on those days.  
**How found:** Task 7 threw an AssertionError on NaN values after pivoting
to wide format. A diagnostic cell counted hours per day and identified the
4 affected dates (2016-03-13, 2017-03-12, 2018-03-11, 2019-03-10).  
**Why:** Daylight Saving Time — clocks jump from 02:00 directly to 03:00.
No energy was settled in that hour because it never occurred.  
**Decision:** X3 is left as NaN on those 4 days. Validation was updated to
confirm NaNs appear only in X3, only on DST days, with exact count matching.
Filling would mean fabricating a price that never existed in the market.

---

## Key Coding Decisions

| Decision | Rationale |
|----------|-----------|
| `glob` for file discovery | Dynamic — picks up new years automatically |
| Wide CSV format (X1–X24) | Matches cQuant model-ready convention |
| Hour-beginning: X1=00:00 | ERCOT standard; `dt.hour + 1` maps 0→1, 23→24 |
| Negative prices retained | Physically valid — wind PTCs in ERCOT |
| `.div(row_means, axis=0)` | Avoids `groupby().apply()` pandas version issues |
| Assertions throughout | Code proves its own correctness at every step |
| Load from CSV between tasks | Each task is self-contained and reproducible |

---

## Analytical Highlights

- **LZ_WEST / HB_WEST** dominate summer price spikes — West Texas
  transmission congestion during peak AC load is a structural feature
- **2017 was the lowest volatility year** across all hubs without exception
- **HB_NORTH** is consistently the least volatile hub — well-connected
  Dallas/Fort Worth load center with no transmission constraints
- **HB_PAN's 0.632 volatility** in 2019 is nearly 2× HB_WEST — Panhandle
  wind intermittency combined with partial-year data
- **OU half-lives of hours** (not days) confirm electricity is the most
  mean-reverting commodity — a fundamental distinction from equities
- **Machine epsilon normalization** (4.44e-16) confirms shape profiles are
  mathematically exact to floating point precision limits
"""

import os
OUTPUT_DIR = "output/"
readme_path = "README.md"

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

assert os.path.exists(readme_path), "README.md was not created"
print(f"Saved → {readme_path}")
