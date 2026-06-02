# ============================================================
# cQuant Energy Analyst Exercise
# Author: Luyang Zhang
# Date: 06/02/2026
# ============================================================

# %%
# --- SYSTEM PARAMETERS — UPDATE THESE BEFORE RUNNING ---
DATA_DIR = "historicalPriceData/"
OUTPUT_DIR = "output/"

# --- REQUIRED PACKAGES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm
import os
import glob
import itertools
import warnings
warnings.filterwarnings("ignore")

# --- SETUP OUTPUT DIRECTORY ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Setup complete")
print(f"Reading data from: {DATA_DIR}")
print(f"Saving outputs to: {OUTPUT_DIR}")

# ============================================================
# TASK 1: Read in all the historical data files from the historicalPriceData folder and 
# combine them into a single data structure
# ============================================================
# %%
print("Scanning for CSV files...")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "ERCOT_DA_Prices_*.csv")))

print(f"  Found {len(csv_files)} files:")
for f in csv_files:
    print(f"    {f}")

# Sanity check — we expect exactly 4 files (2016–2019)
assert len(csv_files) == 4, (
    f"Expected 4 CSV files, found {len(csv_files)}. "
    "Check that DATA_DIR points to the correct folder."
)

# --- Read each file, tag it with its source filename, then stack them ---
# We add a 'source_file' column so we can trace any row back to its origin
# if we ever need to debug a data quality issue.
frames = []
for filepath in csv_files:
    df_temp = pd.read_csv(filepath)
    df_temp["source_file"] = os.path.basename(filepath)
    frames.append(df_temp)
    print(f"  Loaded {filepath}  →  {df_temp.shape[0]:,} rows, "
          f"{df_temp.shape[1]} columns")

# Stack all years into one DataFrame; reset index to 0…N
df = pd.concat(frames, ignore_index=True)

print(f"\nCombined shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# --- Parse the Date column into a proper datetime ---
# The raw format is "2016-01-01 00:00:00"; pandas infers this automatically.
df["Date"] = pd.to_datetime(df["Date"])

# --- Basic validation ---
print("\nColumn dtypes:")
print(df.dtypes)

print("\nDate range:")
print(f"  Min : {df['Date'].min()}")
print(f"  Max : {df['Date'].max()}")

print("\nSettlementPoint unique values:")
print(f"  {df['SettlementPoint'].nunique()} unique nodes")
print(f"  {df['SettlementPoint'].unique()[:10]}")   # first 10

print("\nPrice summary statistics:")
print(df["Price"].describe().round(2))

# Negative prices do occur in ERCOT (wind oversupply) so we allow them,
# but we flag and count them so we are aware.
n_negative = (df["Price"] < 0).sum()
print(f"\nNegative price hours : {n_negative:,}  "
      f"({100 * n_negative / len(df):.2f}% of all rows)")

# Assert no missing values in the three core columns
assert df["Date"].isna().sum()            == 0, "NaT values found in Date"
assert df["SettlementPoint"].isna().sum() == 0, "Nulls found in SettlementPoint"
assert df["Price"].isna().sum()           == 0, "Nulls found in Price"
print("\nValidation passed: no missing values in Date, SettlementPoint, or Price.")

# --- Preview ---
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# --- Save combined dataset ---
out_path = os.path.join(OUTPUT_DIR, "combined_ercot_da_prices.csv")
df.to_csv(out_path, index=False)
assert os.path.exists(out_path), "Output file was not created!"
print(f"\nSaved combined dataset → {out_path}")

# =============================================================================
# ANALYTICAL NOTE:
# ERCOT DA (Day-Ahead) prices are hourly Settlement Point Prices for hubs
# and load zones. HB_BUSAVG is the "bus average" hub — a load-weighted
# average across all buses in ERCOT, often used as the benchmark price.
# Four years (2016–2019) covers ~35,040 hourly observations per node.
# Negative prices are physically meaningful in ERCOT: they occur when
# wind generation is high and generators pay to dispatch rather than curtail
# (due to PTCs making negative prices profitable for wind farms).
# A negative price rate above ~5% would warrant investigation.
# =============================================================================

# ============================================================
# TASK 2: [task name here]
# ============================================================
# %%
df = pd.read_csv(os.path.join(OUTPUT_DIR, "combined_ercot_da_prices.csv"),
                 parse_dates=["Date"])
df["YearMonth"] = df["Date"].dt.to_period("M")

# Build the complete expected grid of every node × every month
all_nodes  = df["SettlementPoint"].unique()
all_months = pd.period_range("2016-01", "2019-12", freq="M")

expected = pd.DataFrame(
    list(itertools.product(all_nodes, all_months)),
    columns=["SettlementPoint", "YearMonth"]
)

# What we actually have
actual = df.groupby(["SettlementPoint", "YearMonth"]).size().reset_index()
actual.columns = ["SettlementPoint", "YearMonth", "HourCount"]

# Left join to find gaps
merged = expected.merge(actual, on=["SettlementPoint", "YearMonth"], how="left")
missing = merged[merged["HourCount"].isna()]

print(f"Missing combinations: {len(missing)}")
print("\nMissing by SettlementPoint:")
print(missing["SettlementPoint"].value_counts().to_string())
print("\nMissing by YearMonth:")
print(missing["YearMonth"].value_counts().to_string())
print("\nFull missing list:")
print(missing.to_string(index=False))

# %%
# --- Classify each settlement point as Hub or Load Zone ---
# HB_ prefix = hub (e.g. HB_BUSAVG, HB_NORTH)
# LZ_ prefix = load zone (e.g. LZ_HOUSTON, LZ_WEST)
df["NodeType"] = df["SettlementPoint"].apply(
    lambda sp: "Hub"      if sp.startswith("HB_")
    else       "LoadZone" if sp.startswith("LZ_")
    else       "Unknown"
)

# Report what we found
type_counts = df.groupby("NodeType")["SettlementPoint"].nunique()
print("\nSettlement point counts by type:")
print(type_counts.to_string())

unknown_nodes = df[df["NodeType"] == "Unknown"]["SettlementPoint"].unique()
if len(unknown_nodes) > 0:
    print(f"\n  ⚠ WARNING: {len(unknown_nodes)} unclassified node(s) found:")
    print(f"    {unknown_nodes}")
else:
    print("\n  All settlement points are HB_ or LZ_ — no unknowns.")

# --- Create a YearMonth period column ---
# dt.to_period("M") converts a timestamp to a monthly period label
# e.g. 2016-01-15 00:00 → 2016-01
df["YearMonth"] = df["Date"].dt.to_period("M")

# --- Compute mean price for every (SettlementPoint, YearMonth) pair ---
# NOTE: negative prices are included as-is — no filtering applied
# Only months where a node actually has data will appear in the output —
# we do NOT fill missing months with NaN or zero, as that would
# misrepresent the market (a missing node is not a $0 price node)
monthly_avg = (
    df.groupby(["SettlementPoint", "NodeType", "YearMonth"])["Price"]
    .mean()
    .round(4)
    .reset_index()
    .rename(columns={"Price": "AvgPrice"})
)

# Sort for readability: hubs first, then load zones, both alphabetical
monthly_avg = monthly_avg.sort_values(
    ["NodeType", "SettlementPoint", "YearMonth"]
).reset_index(drop=True)

# --- Validate shape ---
# We expect 681 rows because HB_PAN is only present for 9 out of 48 months.
# All other 14 nodes have all 48 months → 14×48 + 9 = 681
n_nodes_full   = 14   # nodes with all 48 months
n_months_pan   = monthly_avg[
                     monthly_avg["SettlementPoint"] == "HB_PAN"
                 ]["YearMonth"].nunique()
expected_rows  = (n_nodes_full * 48) + n_months_pan

print(f"\nHB_PAN months in data    : {n_months_pan}")
print(f"Full-coverage nodes      : {n_nodes_full}")
print(f"Expected rows in output  : {expected_rows:,}")
print(f"Actual rows in output    : {monthly_avg.shape[0]:,}")

assert monthly_avg.shape[0] == expected_rows, (
    f"Row count mismatch — expected {expected_rows}, "
    f"got {monthly_avg.shape[0]}"
)

# --- Validate month coverage for all nodes except HB_PAN ---
nodes_except_pan = monthly_avg[monthly_avg["SettlementPoint"] != "HB_PAN"]
months_per_node  = nodes_except_pan.groupby("SettlementPoint")["YearMonth"].nunique()
assert (months_per_node == 48).all(), (
    "Some non-HB_PAN nodes are missing months:\n"
    f"{months_per_node[months_per_node != 48]}"
)
print("Validation passed: all non-HB_PAN nodes have exactly 48 months.")
print(f"Validation passed: HB_PAN has {n_months_pan} months (partial history — expected).")

# --- Validate no NaN averages ---
assert monthly_avg["AvgPrice"].isna().sum() == 0, \
    "NaN values found in AvgPrice — check for months with no hourly data"
print("Validation passed: no NaN values in AvgPrice.")

# --- Summary statistics split by node type ---
print("\nAvgPrice summary by NodeType:")
print(
    monthly_avg.groupby("NodeType")["AvgPrice"]
    .describe()
    .round(2)
    .to_string()
)

# --- Flag any negative monthly averages (rare but valid — report don't remove) ---
neg_months = monthly_avg[monthly_avg["AvgPrice"] < 0]
if len(neg_months) > 0:
    print(f"\n  ⚠ {len(neg_months)} month(s) with negative average price "
          f"(retained per instructions):")
    print(neg_months.to_string(index=False))
else:
    print("\n  No negative monthly averages found.")

# --- Preview ---
print("\nFirst 10 rows of output:")
print(monthly_avg.head(10).to_string(index=False))

# --- Save ---
out_path = os.path.join(OUTPUT_DIR, "avg_price_by_node_yearmonth.csv")
monthly_avg.to_csv(out_path, index=False)
assert os.path.exists(out_path), "Output file was not created!"
print(f"\nSaved → {out_path}")

# =============================================================================
# ANALYTICAL NOTE:
# HB_PAN (Panhandle hub) was introduced to ERCOT's settlement system
# partway through our data window — it does not have data for all 48 months.
# This is a known characteristic of ERCOT's hub history, not a data error.
# HB_PAN represents the West Texas Panhandle region, which has extremely
# high wind generation capacity. When it does appear, its prices are often
# lower than other hubs due to wind oversupply and transmission congestion
# limiting exports out of the region.
#
# For downstream tasks that require a balanced panel (all nodes × all months),
# consider either dropping HB_PAN or restricting the analysis window to only
# the months where HB_PAN exists.
# =============================================================================

#%%
# =============================================================================
# TASK 3 — Write monthly average prices to AveragePriceByMonth.csv
#           Columns: SettlementPoint, Year, Month, AveragePrice
# =============================================================================


# --- Extract Year and Month as separate integer columns ---
# YearMonth was saved as a string like "2016-01" — split on "-" to get both parts
# We cast to int so the output is clean integers (2016, 1) not strings ("2016", "01")
monthly_avg["YearMonth"] = monthly_avg["YearMonth"].astype(str)
monthly_avg["Year"]  = monthly_avg["YearMonth"].str.split("-").str[0].astype(int)
monthly_avg["Month"] = monthly_avg["YearMonth"].str.split("-").str[1].astype(int)

# --- Select and rename columns to match required output format ---
output_df = (
    monthly_avg[["SettlementPoint", "Year", "Month", "AvgPrice"]]
    .rename(columns={"AvgPrice": "AveragePrice"})
    .sort_values(["SettlementPoint", "Year", "Month"])
    .reset_index(drop=True)
)

# --- Validate ---
print(f"\nOutput shape : {output_df.shape[0]:,} rows × {output_df.shape[1]} columns")
print(f"Columns      : {list(output_df.columns)}")

# Year must be in [2016, 2019]
assert output_df["Year"].min()  == 2016, "Unexpected minimum year"
assert output_df["Year"].max()  == 2019, "Unexpected maximum year"

# Month must be in [1, 12]
assert output_df["Month"].min() == 1,  "Unexpected minimum month"
assert output_df["Month"].max() == 12, "Unexpected maximum month"

# No nulls anywhere
assert output_df.isna().sum().sum() == 0, "Unexpected null values in output"

print("Validation passed: Year in [2016–2019], Month in [1–12], no nulls.")

# --- Preview ---
print("\nFirst 10 rows:")
print(output_df.head(10).to_string(index=False))

# --- Save ---
out_path = os.path.join(OUTPUT_DIR, "AveragePriceByMonth.csv")
output_df.to_csv(out_path, index=False)
assert os.path.exists(out_path), "Output file was not created!"
print(f"\nSaved → {out_path}")

# =============================================================================
# ANALYTICAL NOTE:
# Separating Year and Month into distinct integer columns (rather than a
# single "2016-01" string) makes the file easier to consume in downstream
# tools — e.g. pivot tables in Excel, ggplot/seaborn by month, or SQL joins.
# This format is also standard in energy reporting (FERC, EIA datasets use
# the same convention).
# =============================================================================

#%%
# =============================================================================
# TASK 4 — Hourly price volatility per hub (HB_) and year
#           Volatility = standard deviation of log returns of hourly prices
#           Log return at hour t = ln(P_t / P_{t-1})
#           Zero and negative prices are removed before computing log returns
#           Load zones (LZ_) are excluded
# =============================================================================

# --- Filter to hubs only (HB_ prefix) ---
# Load zones are excluded per task instructions
hubs = df[df["SettlementPoint"].str.startswith("HB_")].copy()

print(f"\nHub rows (before price filter) : {hubs.shape[0]:,}")
print(f"Unique hubs                    : {hubs['SettlementPoint'].nunique()}")
print(f"Hubs found: {sorted(hubs['SettlementPoint'].unique())}")

# --- Filter out zero and negative prices before computing log returns ---
# ln(P) is only defined for P > 0; zero/negative prices would produce
# -inf or NaN in the log return and must be removed before this step.
# We report how many rows are removed so the evaluator can judge materiality.
n_before = len(hubs)
hubs = hubs[hubs["Price"] > 0].copy()
n_removed = n_before - len(hubs)
print(f"\nRows removed (price ≤ 0)       : {n_removed:,} "
      f"({100 * n_removed / n_before:.3f}% of hub rows)")

# --- Extract year for grouping ---
hubs["Year"] = hubs["Date"].dt.year

# --- Compute log returns and volatility per (SettlementPoint, Year) ---
# For each hub-year group:
#   1. Sort by Date so hours are in chronological order
#   2. Compute log return: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
#      This is the continuously compounded return between consecutive hours
#   3. Volatility = standard deviation of those log returns (ddof=1, sample std)
#
# NOTE: we use .shift(1) within each group to avoid computing a log return
# that crosses the year boundary (Dec 31 hour 23 → Jan 1 hour 0 of next year)
# That cross-year return would be meaningless and is automatically dropped
# because groupby isolates each year before shifting.

results = []

for (node, year), group in hubs.groupby(["SettlementPoint", "Year"]):
    # Sort chronologically within this hub-year window
    group = group.sort_values("Date")

    # Log return: ln(P_t) - ln(P_{t-1})
    # np.log returns NaN for the first row (no previous price) — that's correct
    log_returns = np.log(group["Price"]) - np.log(group["Price"].shift(1))

    # Drop the first NaN (no return available for the very first hour)
    log_returns = log_returns.dropna()

    # Standard deviation of log returns (ddof=1 = sample standard deviation)
    volatility = log_returns.std(ddof=1)

    results.append({
        "SettlementPoint" : node,
        "Year"            : year,
        "Volatility"      : round(volatility, 6),
        "NHours"          : len(group),        # hours used after price filter
        "NReturns"        : len(log_returns),  # number of log returns computed
    })

volatility_df = pd.DataFrame(results).sort_values(
    ["SettlementPoint", "Year"]
).reset_index(drop=True)

# --- Validate ---
n_hubs  = hubs["SettlementPoint"].nunique()
n_years = 4   # 2016, 2017, 2018, 2019

# HB_PAN only exists for part of the data — it may not span all 4 years
n_pan_years = volatility_df[
    volatility_df["SettlementPoint"] == "HB_PAN"
]["Year"].nunique()

expected_rows = (n_hubs - 1) * n_years + n_pan_years
print(f"\nUnique hubs          : {n_hubs}")
print(f"HB_PAN years in data : {n_pan_years}")
print(f"Expected rows        : {expected_rows}")
print(f"Actual rows          : {volatility_df.shape[0]}")

assert volatility_df.shape[0] == expected_rows, (
    f"Row count mismatch — expected {expected_rows}, "
    f"got {volatility_df.shape[0]}"
)

assert volatility_df["Volatility"].isna().sum() == 0, \
    "NaN volatilities found — check for hubs with insufficient price data"

assert (volatility_df["Volatility"] > 0).all(), \
    "Zero or negative volatilities found — check log return calculation"

print("Validation passed: correct row count, no NaN, all volatilities > 0.")

# --- Summary ---
print("\nVolatility summary:")
print(volatility_df["Volatility"].describe().round(6))

print("\nFull output:")
print(volatility_df.to_string(index=False))

# --- Save ---
out_path = os.path.join(OUTPUT_DIR, "hub_volatility_by_year.csv")
volatility_df.to_csv(out_path, index=False)
assert os.path.exists(out_path), "Output file was not created!"
print(f"\nSaved → {out_path}")

# =============================================================================
# ANALYTICAL NOTE:
# Hourly log return volatility is the standard measure of price risk in
# power markets — analogous to daily return volatility in equities but
# computed on hourly intervals to capture intraday price dynamics.
#
# Expected volatility range for ERCOT hubs (2016–2019): roughly 0.20–0.60
# (i.e. 20%–60% on a per-return basis). Power markets have far higher
# volatility than equities or gas because electricity cannot be stored
# economically — a supply/demand imbalance in a single hour causes an
# immediate price spike with no buffer from inventory.
#
# What to look for:
#   - Higher volatility in summer years (2018/2019 in ERCOT were hot)
#   - HB_WEST and HB_PAN typically show higher volatility than other hubs
#     due to wind intermittency in West Texas — wind output can swing
#     dramatically hour to hour, causing large log returns
#   - HB_BUSAVG (load-weighted average) should show lower volatility than
#     individual hubs because averaging across buses smooths local spikes
#
# Limitation: removing zero/negative prices before computing log returns
# slightly understates true volatility, since price spikes from near-zero
# TO positive (or vice versa) are among the most volatile moves in the
# series. This is a known trade-off when applying log returns to power prices.
# =============================================================================

#%%
# =============================================================================
# TASK 5 — Write hub volatilities to HourlyVolatilityByYear.csv
#           Columns: SettlementPoint, Year, HourlyVolatility (case-sensitive)
# =============================================================================

# --- Select only the three required columns and rename Volatility ---
# We drop the diagnostic columns (NHours, NReturns) added in Task 4 —
# those were for our own validation, not part of the deliverable
output_df = (
    volatility_df[["SettlementPoint", "Year", "Volatility"]]
    .rename(columns={"Volatility": "HourlyVolatility"})
    .sort_values(["SettlementPoint", "Year"])
    .reset_index(drop=True)
)

# --- Validate ---
# Confirm exact column names (task specifies they are case-sensitive)
expected_columns = ["SettlementPoint", "Year", "HourlyVolatility"]
assert list(output_df.columns) == expected_columns, (
    f"Column mismatch — expected {expected_columns}, "
    f"got {list(output_df.columns)}"
)

assert output_df.shape[0] == 25, (
    f"Expected 25 rows, got {output_df.shape[0]}"
)

assert output_df.isna().sum().sum() == 0, \
    "Unexpected null values in output"

assert (output_df["HourlyVolatility"] > 0).all(), \
    "Non-positive volatility found — check Task 4 output"

print(f"\nOutput shape : {output_df.shape[0]} rows × {output_df.shape[1]} columns")
print(f"Columns      : {list(output_df.columns)}")
print("Validation passed: correct columns, 25 rows, no nulls, all volatilities > 0.")

# --- Preview ---
print("\nFull output:")
print(output_df.to_string(index=False))

# --- Save ---
out_path = os.path.join(OUTPUT_DIR, "HourlyVolatilityByYear.csv")
output_df.to_csv(out_path, index=False)
assert os.path.exists(out_path), "Output file was not created!"
print(f"\nSaved → {out_path}")

# =============================================================================
# ANALYTICAL NOTE:
# HourlyVolatility is expressed as a unitless ratio (standard deviation of
# log returns). It is NOT annualized here — it reflects the typical
# hour-to-hour price variability within a given year.
# To annualize (for comparison with financial vol conventions), you would
# multiply by sqrt(8760) — the number of hours in a year. For example,
# a hub with hourly vol of 0.23 would have an annualized vol of ~21.5,
# or 2,150% — illustrating why power is considered the most volatile
# commodity class.
# =============================================================================

#%%
# =============================================================================
# TASK 6 — Find the hub with highest hourly volatility for each year
#           Output: MaxVolatilityByYear.csv
# =============================================================================

# --- Load the volatility file saved in Task 5 ---
print("Loading HourlyVolatilityByYear.csv from Task 5...")
vol_df = pd.read_csv(os.path.join(OUTPUT_DIR, "HourlyVolatilityByYear.csv"))
print(f"  Loaded {vol_df.shape[0]:,} rows × {vol_df.shape[1]} columns")

# === TASK 6: MAX VOLATILITY HUB PER YEAR ===

# --- For each year, find the row with the highest HourlyVolatility ---
# .idxmax() returns the index label of the maximum value within each group.
# We use it to select the full row (not just the max value) so we keep
# SettlementPoint alongside the volatility figure.
max_vol_idx = vol_df.groupby("Year")["HourlyVolatility"].idxmax()
max_vol_df  = vol_df.loc[max_vol_idx].reset_index(drop=True)

# Sort by Year for clean output
max_vol_df = max_vol_df.sort_values("Year").reset_index(drop=True)

# --- Validate ---
# Expect exactly one row per year (2016, 2017, 2018, 2019)
expected_years = [2016, 2017, 2018, 2019]

assert list(max_vol_df["Year"].values) == expected_years, (
    f"Year mismatch — expected {expected_years}, "
    f"got {list(max_vol_df['Year'].values)}"
)

assert max_vol_df.shape[0] == 4, (
    f"Expected 4 rows (one per year), got {max_vol_df.shape[0]}"
)

assert max_vol_df.isna().sum().sum() == 0, \
    "Unexpected null values in output"

# Confirm each selected volatility is truly the max for that year
for _, row in max_vol_df.iterrows():
    year_max = vol_df[vol_df["Year"] == row["Year"]]["HourlyVolatility"].max()
    assert row["HourlyVolatility"] == year_max, (
        f"Row for {row['Year']} is not the maximum — check idxmax logic"
    )

print("Validation passed: 4 rows, one per year, each is the true maximum.")

# --- Preview ---
print("\nFull output:")
print(max_vol_df.to_string(index=False))

# --- Save ---
out_path = os.path.join(OUTPUT_DIR, "MaxVolatilityByYear.csv")
max_vol_df.to_csv(out_path, index=False)
assert os.path.exists(out_path), "Output file was not created!"
print(f"\nSaved → {out_path}")

# =============================================================================
# ANALYTICAL NOTE:
# The max volatility hub reveals which part of the ERCOT grid experienced
# the most extreme hour-to-hour price swings each year.
# Expected result:
#   2016–2018: HB_WEST — West Texas wind intermittency dominates
#   2019:      HB_PAN  — Panhandle wind, even more exposed than HB_WEST,
#                        and its only year in the dataset
# If HB_PAN wins 2019 (volatility 0.63 vs HB_WEST 0.34), that is a large
# margin — worth flagging as a data artifact since HB_PAN only has 9 months
# of data in 2019. A partial year with extreme seasonal months could inflate
# its annual volatility relative to a full-year hub.
# This is a meaningful limitation: comparing a partial-year hub to full-year
# hubs on an annual volatility basis is not strictly apples-to-apples.
# =============================================================================

# =============================================================================
# TASK 7 — Translate hourly data into cQuant model-ready format
#           One CSV per settlement point, saved in output/formattedSpotHistory/
#           Format: Variable, Date, X1, X2, ..., X24
#           Hour-beginning convention: X1=00:00, X2=01:00, ..., X24=23:00
#           Filename convention: spot_<SettlementPoint>.csv
# =============================================================================
#%%
df = pd.read_csv(os.path.join(OUTPUT_DIR, "combined_ercot_da_prices.csv"),
                 parse_dates=["Date"])

df["DateOnly"]  = df["Date"].dt.date
df["HourLabel"] = "X" + (df["Date"].dt.hour + 1).astype(str)

# Check every node for missing hours on any given day
for node in sorted(df["SettlementPoint"].unique()):
    node_df = df[df["SettlementPoint"] == node]
    
    # Count hours per day — should always be 24
    hours_per_day = node_df.groupby("DateOnly")["HourLabel"].count()
    bad_days = hours_per_day[hours_per_day != 24]
    
    if len(bad_days) > 0:
        print(f"{node}: {len(bad_days)} days with != 24 hours")
        print(bad_days.to_string())
        print()
    else:
        print(f"{node}: OK — all days have exactly 24 hours")
#%%
SPOT_DIR   = os.path.join(OUTPUT_DIR, "formattedSpotHistory")

os.makedirs(SPOT_DIR, exist_ok=True)
print(f"Output subdirectory: {SPOT_DIR}")

# --- Load the combined dataset from Task 1 ---
print("\nLoading combined dataset from Task 1...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "combined_ercot_da_prices.csv"),
                 parse_dates=["Date"])
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# === TASK 7: PIVOT TO WIDE FORMAT AND WRITE ONE FILE PER SETTLEMENT POINT ===

# --- Extract Date and HourLabel (X1–X24) ---
# Hour-beginning: 00:00 → X1, 01:00 → X2, ..., 23:00 → X24
df["DateOnly"]  = df["Date"].dt.date
df["HourLabel"] = "X" + (df["Date"].dt.hour + 1).astype(str)

# --- Identify DST spring-forward days (23-hour days) ---
# These are the 4 days per year where clocks jump 02:00 → 03:00
# X3 (the 02:00 hour) will be NaN on these days — this is correct
hours_per_day = df.groupby(["SettlementPoint", "DateOnly"])["HourLabel"].count()
dst_days = hours_per_day[hours_per_day == 23].reset_index()["DateOnly"].unique()
print(f"\nDST spring-forward days found (23-hour days): {len(dst_days)}")
for d in sorted(dst_days):
    print(f"  {d}")

# --- Get all settlement points ---
all_nodes = sorted(df["SettlementPoint"].unique())
print(f"\nSettlement points to process : {len(all_nodes)}")
assert len(all_nodes) == 15, \
    f"Expected 15 settlement points, found {len(all_nodes)}"

# --- Process each settlement point ---
files_written = []
hour_cols = [f"X{i}" for i in range(1, 25)]   # X1, X2, ..., X24

for node in all_nodes:
    # Filter to this node only
    node_df = df[df["SettlementPoint"] == node].copy()

    # Pivot: rows = DateOnly, columns = X1..X24, values = Price
    wide = node_df.pivot(index="DateOnly", columns="HourLabel", values="Price")

    # Reorder columns numerically (X1, X2, ..., X24)
    # pivot() sorts alphabetically by default: X1, X10, X11 ... X9
    wide = wide.reindex(columns=hour_cols)

    # Reset index and rename DateOnly → Date
    wide = wide.reset_index().rename(columns={"DateOnly": "Date"})

    # Insert Variable as the first column
    wide.insert(0, "Variable", node)

    n_rows = wide.shape[0]
    n_cols = wide.shape[1]   # 26: Variable + Date + X1..X24

    # --- Validate shape ---
    assert n_cols == 26, (
        f"{node}: expected 26 columns, got {n_cols}"
    )
    assert wide["Variable"].nunique() == 1, \
        f"{node}: Variable column contains more than one settlement point"

    # --- Validate NaNs ---
    # Only X3 on DST spring-forward days should be NaN — nothing else
    nan_counts = wide.iloc[:, 2:].isna().sum()   # sum NaNs per hour column
    nan_cols   = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        # X3 NaNs are expected — count should equal number of DST days
        # For HB_PAN (partial year) DST days may fall outside its date range
        date_min = pd.to_datetime(wide["Date"]).min().date()
        date_max = pd.to_datetime(wide["Date"]).max().date()
        n_dst_in_range = sum(
            date_min <= d <= date_max
            for d in dst_days
        )
        unexpected = nan_cols[nan_cols.index != "X3"]
        assert len(unexpected) == 0, (
            f"{node}: unexpected NaN values in non-DST columns: "
            f"{unexpected.to_dict()}"
        )
        assert nan_cols.get("X3", 0) == n_dst_in_range, (
            f"{node}: expected {n_dst_in_range} NaN in X3 "
            f"(DST days), got {nan_cols.get('X3', 0)}"
        )
        print(f"  {node:20s} → {n_rows:,} days × {n_cols} cols "
              f"[X3 NaN on {nan_cols.get('X3',0)} DST day(s) — expected]")
    else:
        print(f"  {node:20s} → {n_rows:,} days × {n_cols} cols")

    # Write CSV
    filename  = f"spot_{node}.csv"
    file_path = os.path.join(SPOT_DIR, filename)
    wide.to_csv(file_path, index=False)
    files_written.append(file_path)

# --- Confirm all 15 files written ---
print(f"\nFiles written : {len(files_written)}")
assert len(files_written) == 15, \
    f"Expected 15 files, wrote {len(files_written)}"
for fp in files_written:
    assert os.path.exists(fp), f"File not found after write: {fp}"
print("Validation passed: all 15 files confirmed on disk.")

# --- Spot-check one file ---
print("\nSpot-check — first 2 rows of spot_HB_BUSAVG.csv:")
check = pd.read_csv(os.path.join(SPOT_DIR, "spot_HB_BUSAVG.csv"))
print(check.head(2).to_string(index=False))
print(f"Shape: {check.shape[0]:,} rows × {check.shape[1]} columns")

# =============================================================================
# ANALYTICAL NOTE:
# DST spring-forward days are a known data characteristic in all North
# American power markets. On the second Sunday of March, clocks jump from
# 02:00 to 03:00 — the 02:00 hour physically does not exist.
# In ERCOT DA prices, this hour is simply absent from the dataset.
# After pivoting to wide format, this appears as NaN in the X3 column
# (the 02:00–03:00 slot) on those 4 days.
#
# How models typically handle DST NaNs:
#   1. Leave as NaN and let the model skip that hour (our approach)
#   2. Forward-fill X3 from X2 (conservative — assumes prices held steady)
#   3. Interpolate between X2 and X4 (smoother but adds synthetic data)
# Option 1 is the most transparent for a data preparation step —
# we do not fabricate prices that never existed in the market.
# The model operator can decide how to handle missing DST hours downstream.
# =============================================================================

# =============================================================================
# BONUS — Monthly average price line plots
#         Plot 1: All settlement hubs (HB_) — one curve per hub
#         Plot 2: All load zones (LZ_)      — one curve per load zone
#         X-axis: chronological date (first day of each month)
#         Saved as PNG to OUTPUT_DIR
# =============================================================================

#%%
# --- Load monthly averages from Task 3 ---
print("Loading AveragePriceByMonth.csv from Task 3...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "AveragePriceByMonth.csv"))
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# === BONUS: MEAN PRICE LINE PLOTS ===

# --- Create a proper date column for the x-axis ---
# Assign each month to its first day so matplotlib can plot chronologically
# e.g. Year=2016, Month=1 → 2016-01-01
df["Date"] = pd.to_datetime(
    df["Year"].astype(str) + "-" +
    df["Month"].astype(str).str.zfill(2) + "-01"
)

# --- Split into hubs and load zones ---
hubs  = df[df["SettlementPoint"].str.startswith("HB_")]
zones = df[df["SettlementPoint"].str.startswith("LZ_")]

print(f"\nHubs to plot      : {sorted(hubs['SettlementPoint'].unique())}")
print(f"Load zones to plot: {sorted(zones['SettlementPoint'].unique())}")

# --- Shared plot settings ---
# Use a colormap with enough distinct colors for up to 8 nodes per plot
HUB_CMAP  = plt.cm.tab10
ZONE_CMAP = plt.cm.tab10

def make_avg_price_plot(data, title, filename, cmap):
    """
    Plot monthly average price curves for each settlement point in `data`.
    One curve per SettlementPoint, colored by cmap, saved to OUTPUT_DIR.
    """
    nodes     = sorted(data["SettlementPoint"].unique())
    n_nodes   = len(nodes)
    colors    = [cmap(i / n_nodes) for i in range(n_nodes)]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, node in enumerate(nodes):
        node_data = data[data["SettlementPoint"] == node].sort_values("Date")
        ax.plot(
            node_data["Date"],
            node_data["AveragePrice"],
            label=node,
            color=colors[i],
            linewidth=1.8,
            marker="o",
            markersize=3
        )

    # --- Axes formatting ---
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Average Price ($/MWh)", fontsize=11)

    # Show one x-tick per quarter (every 3 months) to avoid crowding
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right", fontsize=8)

    # Add minor ticks for every month (no label) so grid aligns to months
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", which="minor", linestyle=":", alpha=0.3)

    # Place legend outside the plot area so it doesn't obscure curves
    ax.legend(
        title="Settlement Point",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9
    )

    # Add a note about HB_PAN partial history if it's in this plot
    if "HB_PAN" in data["SettlementPoint"].values:
        ax.annotate(
            "HB_PAN: partial history (9 months in 2019 only)",
            xy=(0.01, 0.02), xycoords="axes fraction",
            fontsize=8, color="gray", style="italic"
        )

    plt.tight_layout()

    # Save and close
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert os.path.exists(out_path), f"Plot file not created: {out_path}"
    print(f"  Saved → {out_path}")
    return out_path

# --- Plot 1: Settlement Hubs ---
print("\nGenerating hub plot...")
make_avg_price_plot(
    data     = hubs,
    title    = "ERCOT DA Monthly Average Price — Settlement Hubs (2016–2019)",
    filename = "SettlementHubAveragePriceByMonth.png",
    cmap     = HUB_CMAP
)

# --- Plot 2: Load Zones ---
print("Generating load zone plot...")
make_avg_price_plot(
    data     = zones,
    title    = "ERCOT DA Monthly Average Price — Load Zones (2016–2019)",
    filename = "LoadZoneAveragePriceByMonth.png",
    cmap     = ZONE_CMAP
)

print("\nBoth plots saved successfully.")

# =============================================================================
# ANALYTICAL NOTE:
# These plots reveal several key ERCOT market dynamics:
#
# 1. SEASONAL PATTERN: Clear summer peaks (Jul–Aug) each year driven by
#    Texas heat and AC load, with a secondary winter peak. Spring and fall
#    show the lowest prices due to mild weather and strong wind output.
#
# 2. YEAR-OVER-YEAR TREND: 2018–2019 should show higher peaks than 2016–2017
#    reflecting both load growth and increased renewable intermittency.
#
# 3. HUB vs LOAD ZONE SPREAD: Load zones (LZ_) should track closely with
#    their corresponding hubs (HB_) but sit slightly higher on average,
#    reflecting the cost of transmission from generation to load.
#
# 4. HB_PAN: Will appear as a short stub in the hub plot starting mid-2019.
#    Its prices may diverge significantly from other hubs due to wind
#    oversupply and transmission congestion in the Panhandle region.
#
# 5. CONVERGENCE: During low-demand months, hub and load zone prices
#    converge as transmission constraints relax — visible as tightly
#    clustered curves in spring/fall months.
# =============================================================================

# =============================================================================
# BONUS — Volatility comparison plots across settlement hubs by year
#         Plot 1: Grouped bar chart — hubs side by side for each year
#                 Best for comparing hub rankings within and across years
#         Plot 2: Heatmap — hubs vs years, color = volatility magnitude
#                 Best for spotting patterns at a glance across the full matrix
# =============================================================================
#%%
# --- Load volatility data from Task 5 ---
print("Loading HourlyVolatilityByYear.csv...")
vol_df = pd.read_csv(os.path.join(OUTPUT_DIR, "HourlyVolatilityByYear.csv"))
print(f"  Loaded {vol_df.shape[0]} rows × {vol_df.shape[1]} columns")

# Pivot to wide format: rows = SettlementPoint, columns = Year
# This matrix layout is used by both plots
vol_pivot = vol_df.pivot(
    index="SettlementPoint",
    columns="Year",
    values="HourlyVolatility"
)

# Sort hubs by their mean volatility across all years (ascending)
# so the least volatile hub is at the bottom/left and most volatile at top/right
vol_pivot = vol_pivot.loc[vol_pivot.mean(axis=1).sort_values().index]

print("\nVolatility matrix (rows=hub, cols=year):")
print(vol_pivot.round(4).to_string())

years = sorted(vol_df["Year"].unique())
hubs  = list(vol_pivot.index)

# =============================================================================
# PLOT 1: Grouped bar chart
# Each year is a group of bars, one bar per hub within the group
# Makes it easy to compare hub rankings within a year and trends over time
# =============================================================================
print("\nGenerating Plot 1: Grouped bar chart...")

n_hubs  = len(hubs)
n_years = len(years)

# Bar positioning: cluster bars by year, offset each hub within the cluster
bar_width   = 0.11                          # width of each individual bar
group_gap   = 0.05                          # extra space between year groups
offsets     = np.arange(n_hubs) * bar_width # offset within each year group
group_width = n_hubs * bar_width + group_gap
x_centers   = np.arange(n_years) * group_width   # center of each year group

# One color per hub, consistent across both plots
cmap   = plt.cm.tab10
colors = {hub: cmap(i / n_hubs) for i, hub in enumerate(hubs)}

fig1, ax1 = plt.subplots(figsize=(14, 6))

for i, hub in enumerate(hubs):
    # x position for this hub's bar in each year group
    x_positions = x_centers + offsets[i] - (n_hubs * bar_width / 2)
    values = [vol_pivot.loc[hub, yr] if yr in vol_pivot.columns
              else 0 for yr in years]

    bars = ax1.bar(
        x_positions,
        values,
        width=bar_width * 0.9,    # slight gap between bars
        label=hub,
        color=colors[hub],
        edgecolor="white",
        linewidth=0.5
    )

    # Label each bar with its value if it's notably high (> 0.35)
    # to draw attention to outliers without cluttering all bars
    for bar, val in zip(bars, values):
        if val > 0.35:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color="black"
            )

# X-axis ticks at center of each year group
ax1.set_xticks(x_centers)
ax1.set_xticklabels([str(yr) for yr in years], fontsize=11)
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Hourly Volatility (σ of log returns)", fontsize=11)
ax1.set_title(
    "ERCOT DA Hourly Price Volatility by Settlement Hub and Year",
    fontsize=13, fontweight="bold", pad=12
)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.set_ylim(0, vol_pivot.max().max() * 1.15)   # headroom for labels

# Legend outside plot area
ax1.legend(
    title="Settlement Hub",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=10,
    framealpha=0.9
)

# Annotate HB_PAN as partial year
ax1.annotate(
    "HB_PAN: 2019 only (9 months of data)",
    xy=(0.01, 0.96), xycoords="axes fraction",
    fontsize=8, color="gray", style="italic"
)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "VolatilityByHub_BarChart.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
assert os.path.exists(out1)
print(f"  Saved → {out1}")

# =============================================================================
# PLOT 2: Heatmap
# Rows = hubs (sorted by mean volatility), columns = years
# Color intensity = volatility magnitude
# Makes the full hub × year matrix readable at a glance
# NaN cells (HB_PAN missing years) shown in light gray
# =============================================================================
print("Generating Plot 2: Heatmap...")

fig2, ax2 = plt.subplots(figsize=(8, 6))

# Build the matrix as a numpy array for imshow
matrix = vol_pivot.values.astype(float)   # shape: (n_hubs, n_years)

# Use a masked array so NaN cells (HB_PAN missing years) render as gray
masked = np.ma.masked_invalid(matrix)

# Color scale: white = low volatility, deep red = high volatility
cmap_heat = plt.cm.YlOrRd
cmap_heat.set_bad(color="#d0d0d0")   # gray for NaN cells

im = ax2.imshow(
    masked,
    cmap=cmap_heat,
    aspect="auto",
    vmin=0,
    vmax=vol_pivot.max().max()
)

# Axis labels
ax2.set_xticks(range(n_years))
ax2.set_xticklabels([str(yr) for yr in years], fontsize=11)
ax2.set_yticks(range(len(hubs)))
ax2.set_yticklabels(hubs, fontsize=10)
ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Settlement Hub", fontsize=11)
ax2.set_title(
    "ERCOT DA Hourly Volatility Heatmap\nSettlement Hubs × Year",
    fontsize=13, fontweight="bold", pad=12
)

# Annotate each cell with its numeric value
for row_idx in range(len(hubs)):
    for col_idx in range(n_years):
        val = matrix[row_idx, col_idx]
        if not np.isnan(val):
            # Use white text on dark cells, black on light cells for readability
            text_color = "white" if val > 0.35 else "black"
            ax2.text(
                col_idx, row_idx,
                f"{val:.3f}",
                ha="center", va="center",
                fontsize=9, color=text_color, fontweight="bold"
            )
        else:
            ax2.text(
                col_idx, row_idx, "N/A",
                ha="center", va="center",
                fontsize=9, color="#888888"
            )

# Colorbar
cbar = fig2.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label("Hourly Volatility (σ of log returns)", fontsize=10)

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "VolatilityByHub_Heatmap.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
assert os.path.exists(out2)
print(f"  Saved → {out2}")

print("\nBoth volatility plots saved successfully.")

# =============================================================================
# ANALYTICAL NOTE:
# Two complementary visualizations are used because each answers a different
# question:
#
# Bar chart: "How does each hub rank relative to others within a given year?"
#   → Easy to see that HB_WEST and HB_PAN dominate, and that 2019 was
#     the most volatile year across almost all hubs
#
# Heatmap: "What is the overall pattern across the full hub × year matrix?"
#   → Immediately shows the top-right corner (HB_WEST/HB_PAN, 2019) as the
#     hottest cell, and the bottom-left (HB_NORTH, 2017) as the coolest —
#     the volatility gradient is visible at a glance
#
# Key findings:
#   - Volatility increased year-over-year from 2017→2019 for all hubs,
#     consistent with rising renewable penetration in ERCOT
#   - HB_WEST is persistently the most volatile full-history hub —
#     West Texas wind intermittency is a structural feature, not a one-off
#   - HB_PAN's 0.632 in 2019 is an outlier — partially explained by it
#     being a partial year, but also reflective of extreme Panhandle wind
#   - HB_NORTH consistently shows the lowest volatility — the Dallas/Fort
#     Worth load center is well-connected and transmission-unconstrained
# =============================================================================

#%%
# =============================================================================
# BONUS — Hourly shape profiles by settlement point, month, and day of week
#         For each (SettlementPoint, Month, DayOfWeek) group:
#           1. Compute the average price for each of the 24 hours
#           2. Divide each hour's average by the mean of all 24 averages
#           → 24 normalized values that average to exactly 1.0
#         Output: wide format — one row per (Month, DayOfWeek) combination
#                 columns: Month, DayOfWeek, H1, H2, ..., H24
#         One CSV per settlement point (15 files total)
#         Saved in output/hourlyShapeProfiles/
# =============================================================================


PROFILE_DIR = os.path.join(OUTPUT_DIR, "hourlyShapeProfiles")

os.makedirs(PROFILE_DIR, exist_ok=True)
print(f"Output subdirectory: {PROFILE_DIR}")

# --- Load the combined dataset from Task 1 ---
print("\nLoading combined dataset from Task 1...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "combined_ercot_da_prices.csv"),
                 parse_dates=["Date"])
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# === BONUS: HOURLY SHAPE PROFILES ===

# --- Extract time components needed for grouping ---
df["Month"]     = df["Date"].dt.month        # 1–12
df["DayOfWeek"] = df["Date"].dt.day_name()   # Monday, Tuesday, ..., Sunday
df["Hour"]      = df["Date"].dt.hour + 1     # 1–24 (hour-beginning convention)

# Define ordered day-of-week list so output rows sort Mon→Sun
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Hour columns in output: H1, H2, ..., H24
hour_cols = [f"H{h}" for h in range(1, 25)]

# --- Get all settlement points ---
all_nodes = sorted(df["SettlementPoint"].unique())
print(f"\nSettlement points to process: {len(all_nodes)}")
assert len(all_nodes) == 15, \
    f"Expected 15 settlement points, found {len(all_nodes)}"

EXPECTED_PROFILES = 84   # 12 months × 7 days of week
files_written = []

for node in all_nodes:
    node_df = df[df["SettlementPoint"] == node].copy()

    # --- Step 1: Average price by (Month, DayOfWeek, Hour) ---
    # Computes the mean price for every (month, day-of-week, hour) combination
    # e.g. average price at H18 across all Mondays in January 2016–2019
    hourly_avg = (
        node_df.groupby(["Month", "DayOfWeek", "Hour"])["Price"]
        .mean()
        .reset_index()
        .rename(columns={"Price": "AvgPrice"})
    )

    # --- Step 2: Pivot to wide format FIRST ---
    # Rows: (Month, DayOfWeek), Columns: Hour 1–24, Values: AvgPrice
    # We pivot before normalizing so we can operate on full rows cleanly
    wide = hourly_avg.pivot(
        index=["Month", "DayOfWeek"],
        columns="Hour",
        values="AvgPrice"
    )

    # Rename columns from integers (1–24) to H1–H24
    wide.columns = [f"H{int(h)}" for h in wide.columns]

    # Reorder columns numerically (pivot may sort alphabetically)
    wide = wide[hour_cols]

    # --- Step 3: Normalize each row so its 24 values average to exactly 1.0 ---
    # row_means is a Series with one mean per (Month, DayOfWeek) row
    # Dividing wide by row_means.values[:,None] broadcasts across all 24 columns
    row_means = wide[hour_cols].mean(axis=1)
    wide[hour_cols] = wide[hour_cols].div(row_means, axis=0)

    # Reset index so Month and DayOfWeek become regular columns
    wide = wide.reset_index()

    # Sort rows: Month 1–12, within each month Monday→Sunday
    wide["DayOfWeek"] = pd.Categorical(
        wide["DayOfWeek"], categories=DOW_ORDER, ordered=True
    )
    wide = wide.sort_values(["Month", "DayOfWeek"]).reset_index(drop=True)

# --- Validate shape ---
    # HB_PAN only has 9 months of data so it will have fewer than 84 profiles
    # For all other nodes, exactly 84 (12 months × 7 days) are required
    if node == "HB_PAN":
        assert wide.shape[0] <= EXPECTED_PROFILES, (
            f"{node}: expected at most {EXPECTED_PROFILES} rows, "
            f"got {wide.shape[0]}"
        )
        print(f"  ⚠ HB_PAN: {wide.shape[0]} profiles (partial history — "
              f"{wide.shape[0] // 7} months of data, expected 9)")
    else:
        assert wide.shape[0] == EXPECTED_PROFILES, (
            f"{node}: expected {EXPECTED_PROFILES} rows, got {wide.shape[0]}"
        )
    assert wide.shape[1] == 26, (
        f"{node}: expected 26 columns (Month, DayOfWeek, H1–H24), "
        f"got {wide.shape[1]}"
    )

    # --- Validate normalization: every row mean must equal exactly 1.0 ---
    row_means_check = wide[hour_cols].mean(axis=1)
    max_deviation = (row_means_check - 1.0).abs().max()
    assert max_deviation < 1e-10, \
        f"{node}: normalization failed — max deviation from 1.0: {max_deviation:.2e}"

    # --- Check for NaN values ---
    n_nan = wide[hour_cols].isna().sum().sum()
    if n_nan > 0:
        print(f"  ⚠ {node}: {n_nan} NaN shape values — partial history node")

    print(f"  {node:20s} → {wide.shape[0]} profiles × {wide.shape[1]} columns "
          f"| max normalization error: {max_deviation:.2e}")

    # --- Save ---
    filename  = f"profile_{node}.csv"
    file_path = os.path.join(PROFILE_DIR, filename)
    wide.to_csv(file_path, index=False)
    files_written.append(file_path)

# --- Confirm all 15 files written ---
print(f"\nFiles written: {len(files_written)}")
assert len(files_written) == 15, \
    f"Expected 15 files, wrote {len(files_written)}"
for fp in files_written:
    assert os.path.exists(fp), f"File not found: {fp}"
print("Validation passed: all 15 files confirmed on disk.")

# --- Spot-check: first 3 rows of HB_BUSAVG profile ---
print("\nSpot-check — first 3 rows of profile_HB_BUSAVG.csv:")
check = pd.read_csv(os.path.join(PROFILE_DIR, "profile_HB_BUSAVG.csv"))
print(check.head(3).to_string(index=False))
print(f"\nShape: {check.shape[0]} rows × {check.shape[1]} columns")

hour_cols_check = [f"H{h}" for h in range(1, 25)]
loaded_means = check[hour_cols_check].mean(axis=1)
print(f"Row mean (min): {loaded_means.min():.10f}")
print(f"Row mean (max): {loaded_means.max():.10f}")

# =============================================================================
# ANALYTICAL NOTE:
# Wide format (one row per Month-DayOfWeek pair, H1–H24 as columns) was
# chosen over long format for two reasons:
#   1. It mirrors the cQuant model-ready convention established in Task 7
#      (Variable, Date, X1–X24), making the files consistent across the
#      submission and easy for a model to ingest row by row
#   2. Each row is a complete 24-hour shape profile — a self-contained
#      record that can be read, plotted, or applied without any further
#      pivoting
#
# Interpretation of shape values:
#   - ShapeValue > 1.0: this hour is more expensive than the daily average
#   - ShapeValue < 1.0: this hour is cheaper than the daily average
#   - ShapeValue = 1.0: this hour matches the daily average exactly
#
# Expected patterns in ERCOT:
#   - Weekday profiles: morning ramp (H7–H9 rising), evening peak (H18–H20),
#     overnight trough (H1–H5 lowest shape values ~0.7–0.8)
#   - Weekend profiles: flatter shape, lower peak-to-trough ratio
#   - Summer months (7–8): most pronounced peak shape, driven by AC load
#   - Spring/fall months (3–5, 9–10): flattest profiles, mild demand
#
# Shape profiles are used in forward curve construction to distribute a
# monthly average price forecast into 24 hourly prices — a core step in
# energy trading and risk management.
# =============================================================================
#%%
# =============================================================================
# BONUS — Open-Ended Analysis
#         Ornstein-Uhlenbeck (OU) Mean Reversion Model for ERCOT Hub Prices
#
# MOTIVATION:
# Electricity prices are mean-reverting — unlike stocks, power prices cannot
# drift to infinity because high prices incentivize new generation supply,
# and low prices cause generators to shut down. The Ornstein-Uhlenbeck process
# is the canonical continuous-time model for mean-reverting commodities and
# is the mathematical foundation of most power price simulation models,
# including those built by cQuant.
#
# The OU process is defined by the SDE:
#   dP_t = κ(μ - P_t)dt + σ dW_t
# where:
#   κ = mean reversion speed (how fast prices snap back to the long-run mean)
#   μ = long-run mean price (the equilibrium level prices revert toward)
#   σ = volatility (magnitude of random price shocks)
#   dW_t = Wiener process increment (random noise)
#
# We fit this model to log prices for each hub using OLS regression on the
# discrete-time approximation of the OU SDE, then visualize the parameters
# and simulate future price paths.
# =============================================================================


print("Loading combined dataset from Task 1...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "combined_ercot_da_prices.csv"),
                 parse_dates=["Date"])
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# === PART 1: FIT OU PARAMETERS PER HUB ===
# ---------------------------------------------------------------------------
# The discrete-time OU process (Euler-Maruyama discretization) is:
#   ln(P_t) - ln(P_{t-1}) = κ·μ·Δt - κ·Δt·ln(P_{t-1}) + σ·√Δt·ε_t
#
# This is a linear regression of the form:
#   Δy_t = a + b·y_{t-1} + ε_t
# where:
#   y_t    = ln(P_t)        (log price)
#   Δy_t   = y_t - y_{t-1} (log return)
#   a      = κ·μ·Δt         → μ = a / (κ·Δt) = -a/b
#   b      = -κ·Δt           → κ = -b/Δt
#   σ_ε    = std(residuals)  → σ = σ_ε / √Δt
#   Δt     = 1/8760          (one hour as fraction of a year)
#
# We fit this regression separately for each hub using all available
# hourly data, filtering out non-positive prices before taking logs.
# ---------------------------------------------------------------------------

DT = 1 / 8760   # one hourly timestep as a fraction of a year

# Filter to hubs only, remove non-positive prices
hubs_df = df[df["SettlementPoint"].str.startswith("HB_")].copy()
hubs_df = hubs_df[hubs_df["Price"] > 0].copy()
hubs_df = hubs_df.sort_values(["SettlementPoint", "Date"])

hub_list = sorted(hubs_df["SettlementPoint"].unique())
print(f"\nFitting OU model for {len(hub_list)} hubs...")

ou_params = []

for hub in hub_list:
    h = hubs_df[hubs_df["SettlementPoint"] == hub].copy()

    # Compute log price and log return
    h["LogPrice"]  = np.log(h["Price"])
    h["LogReturn"] = h["LogPrice"].diff()

    # Drop first row (NaN log return) and any remaining NaNs
    h = h.dropna(subset=["LogPrice", "LogReturn"])

    # OLS regression: Δy_t = a + b·y_{t-1} + ε_t
    # y_{t-1} is the lagged log price
    y_lag  = h["LogPrice"].shift(1).dropna()
    dy     = h["LogReturn"].loc[y_lag.index]

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_lag, dy)

    # Recover OU parameters from regression coefficients
    # b = slope = -κ·Δt  →  κ = -slope / Δt
    # a = intercept = κ·μ·Δt  →  μ = intercept / (-slope)  = -a/b
    kappa = -slope / DT                          # mean reversion speed (per year)
    mu    = intercept / (-slope)                 # long-run mean (in log price space)
    mu_price = np.exp(mu)                        # convert back to $/MWh

    # σ from residual std: σ = std(residuals) / √Δt
    residuals = dy - (intercept + slope * y_lag)
    sigma = residuals.std() / np.sqrt(DT)        # annualized volatility

    # Half-life: time for price deviation to decay by 50%
    # half_life = ln(2) / κ  (in years) × 8760 = hours
    half_life_hours = np.log(2) / kappa * 8760

    ou_params.append({
        "Hub"             : hub,
        "Kappa"           : round(kappa,    4),   # mean reversion speed
        "Mu_LogPrice"     : round(mu,       4),   # long-run mean (log scale)
        "Mu_Price"        : round(mu_price, 4),   # long-run mean ($/MWh)
        "Sigma"           : round(sigma,    4),   # annualized volatility
        "HalfLife_Hours"  : round(half_life_hours, 2),
        "R_Squared"       : round(r_value**2, 4),
        "N_Obs"           : len(dy)
    })

    print(f"  {hub:20s} | κ={kappa:7.2f}/yr | μ=${mu_price:6.2f} | "
          f"σ={sigma:.4f} | half-life={half_life_hours:.1f}h | R²={r_value**2:.4f}")

ou_df = pd.DataFrame(ou_params)

# Save OU parameters
ou_path = os.path.join(OUTPUT_DIR, "OU_parameters_by_hub.csv")
ou_df.to_csv(ou_path, index=False)
print(f"\nSaved OU parameters → {ou_path}")

# === PART 2: MONTE CARLO SIMULATION FOR HB_BUSAVG ===
# ---------------------------------------------------------------------------
# Using the fitted OU parameters for HB_BUSAVG, simulate N_PATHS future
# hourly price paths over a 30-day horizon.
# Discrete simulation:
#   ln(P_t) = ln(P_{t-1}) + κ(μ - ln(P_{t-1}))Δt + σ√Δt·ε_t
#   where ε_t ~ N(0,1)
# ---------------------------------------------------------------------------

print("\nRunning Monte Carlo simulation for HB_BUSAVG...")

N_PATHS   = 200     # number of simulated price paths
N_HOURS   = 24 * 30 # 30-day horizon
np.random.seed(42)  # reproducibility

# Get fitted parameters for HB_BUSAVG
params    = ou_df[ou_df["Hub"] == "HB_BUSAVG"].iloc[0]
kappa_sim = params["Kappa"]
mu_sim    = params["Mu_LogPrice"]     # long-run mean in log space
sigma_sim = params["Sigma"]

# Starting price = last observed price for HB_BUSAVG
last_price = (hubs_df[hubs_df["SettlementPoint"] == "HB_BUSAVG"]
              .sort_values("Date")["Price"].iloc[-1])
log_p0 = np.log(last_price)

print(f"  κ={kappa_sim:.2f}/yr | μ=${np.exp(mu_sim):.2f} | "
      f"σ={sigma_sim:.4f} | P0=${last_price:.2f}")

# Simulate paths
log_paths = np.zeros((N_PATHS, N_HOURS + 1))
log_paths[:, 0] = log_p0

for t in range(1, N_HOURS + 1):
    # OU increment: mean reversion pull + random shock
    dW = np.random.normal(0, 1, N_PATHS)
    log_paths[:, t] = (log_paths[:, t-1]
                       + kappa_sim * (mu_sim - log_paths[:, t-1]) * DT
                       + sigma_sim * np.sqrt(DT) * dW)

# Convert log paths back to price space
price_paths = np.exp(log_paths)

# === PART 3: VISUALIZATION ===
print("\nGenerating OU analysis plots...")

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Plot 1 (top left): Mean Reversion Speed (κ) by hub ---
ax1 = fig.add_subplot(gs[0, 0])
colors_bar = plt.cm.RdYlGn_r(
    (ou_df["Kappa"] - ou_df["Kappa"].min()) /
    (ou_df["Kappa"].max() - ou_df["Kappa"].min())
)
bars = ax1.barh(ou_df["Hub"], ou_df["Kappa"],
                color=colors_bar, edgecolor="white")
ax1.set_xlabel("κ — Mean Reversion Speed (per year)", fontsize=10)
ax1.set_title("Mean Reversion Speed by Hub\n"
              "(higher = faster reversion to long-run mean)", fontsize=10,
              fontweight="bold")
ax1.axvline(ou_df["Kappa"].mean(), color="black",
            linestyle="--", linewidth=1, label=f"Mean κ={ou_df['Kappa'].mean():.0f}")
ax1.legend(fontsize=8)
ax1.grid(axis="x", linestyle="--", alpha=0.4)

# --- Plot 2 (top right): Long-run mean price (μ) by hub ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(ou_df["Hub"], ou_df["Mu_Price"],
         color="steelblue", edgecolor="white")
ax2.set_xlabel("μ — Long-Run Mean Price ($/MWh)", fontsize=10)
ax2.set_title("Long-Run Equilibrium Price by Hub\n"
              "(price level that OU process reverts toward)", fontsize=10,
              fontweight="bold")
ax2.axvline(ou_df["Mu_Price"].mean(), color="black",
            linestyle="--", linewidth=1,
            label=f"Mean μ=${ou_df['Mu_Price'].mean():.2f}")
ax2.legend(fontsize=8)
ax2.grid(axis="x", linestyle="--", alpha=0.4)

# --- Plot 3 (middle left): Half-life of mean reversion ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.barh(ou_df["Hub"], ou_df["HalfLife_Hours"],
         color="darkorange", edgecolor="white")
ax3.set_xlabel("Half-Life (hours)", fontsize=10)
ax3.set_title("Mean Reversion Half-Life by Hub\n"
              "(hours for price shock to decay 50%)", fontsize=10,
              fontweight="bold")
ax3.grid(axis="x", linestyle="--", alpha=0.4)

# Annotate each bar with its value
for bar, val in zip(ax3.patches, ou_df["HalfLife_Hours"]):
    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}h", va="center", fontsize=8)

# --- Plot 4 (middle right): Annualized volatility (σ) by hub ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.barh(ou_df["Hub"], ou_df["Sigma"],
         color="crimson", edgecolor="white")
ax4.set_xlabel("σ — Annualized Volatility", fontsize=10)
ax4.set_title("OU Volatility Parameter by Hub\n"
              "(consistent with Task 4 hourly vol findings)", fontsize=10,
              fontweight="bold")
ax4.grid(axis="x", linestyle="--", alpha=0.4)

# --- Plot 5 (bottom, full width): Monte Carlo simulation ---
ax5 = fig.add_subplot(gs[2, :])

hours = np.arange(N_HOURS + 1)

# Plot all simulated paths in light gray
for i in range(N_PATHS):
    ax5.plot(hours, price_paths[i], color="steelblue",
             alpha=0.08, linewidth=0.5)

# Plot percentile bands
p5   = np.percentile(price_paths, 5,  axis=0)
p25  = np.percentile(price_paths, 25, axis=0)
p50  = np.percentile(price_paths, 50, axis=0)
p75  = np.percentile(price_paths, 75, axis=0)
p95  = np.percentile(price_paths, 95, axis=0)

ax5.fill_between(hours, p5,  p95, alpha=0.15, color="steelblue",
                 label="5th–95th percentile")
ax5.fill_between(hours, p25, p75, alpha=0.30, color="steelblue",
                 label="25th–75th percentile")
ax5.plot(hours, p50, color="darkblue",
         linewidth=2, label="Median simulated path")
ax5.axhline(np.exp(mu_sim), color="red", linestyle="--",
            linewidth=1.5, label=f"Long-run mean μ=${np.exp(mu_sim):.2f}")
ax5.axhline(last_price, color="green", linestyle=":",
            linewidth=1.5, label=f"Starting price P0=${last_price:.2f}")

ax5.set_xlabel("Hours into Future", fontsize=10)
ax5.set_ylabel("Simulated Price ($/MWh)", fontsize=10)
ax5.set_title(
    f"OU Monte Carlo Simulation — HB_BUSAVG | {N_PATHS} Paths | 30-Day Horizon\n"
    f"κ={kappa_sim:.1f}/yr  μ=${np.exp(mu_sim):.2f}  σ={sigma_sim:.4f}",
    fontsize=10, fontweight="bold"
)
ax5.legend(fontsize=9, loc="upper right")
ax5.set_xlim(0, N_HOURS)
ax5.set_ylim(0, np.percentile(price_paths, 99) * 1.1)
ax5.grid(linestyle="--", alpha=0.4)

fig.suptitle(
    "Ornstein-Uhlenbeck Mean Reversion Analysis — ERCOT DA Hub Prices (2016–2019)",
    fontsize=13, fontweight="bold", y=1.01
)

out_plot = os.path.join(OUTPUT_DIR, "OU_MeanReversion_Analysis.png")
fig.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close(fig)
assert os.path.exists(out_plot)
print(f"Saved → {out_plot}")

# --- Print final summary table ---
print("\nOU Parameter Summary:")
print(ou_df[["Hub", "Kappa", "Mu_Price", "Sigma",
             "HalfLife_Hours", "R_Squared"]].to_string(index=False))

# =============================================================================
# ANALYTICAL NOTE:
#
# MEAN REVERSION SPEED (κ):
# A κ of ~5,000–50,000/year is typical for hourly power prices — far higher
# than equities (κ~1–5/year) or even natural gas (κ~10–30/year). This means
# electricity price shocks decay within hours, not days or months. The
# half-life plot makes this intuitive: a price spike in ERCOT typically
# reverts 50% within just a few hours.
#
# LONG-RUN MEAN (μ):
# The equilibrium price μ should be close to the historical average price
# (~$25–35/MWh for ERCOT 2016–2019). If μ deviates significantly from the
# observed mean, it suggests structural price shifts (e.g. fuel cost changes,
# capacity additions) that a single-regime OU model cannot fully capture.
#
# VOLATILITY (σ):
# The OU σ is annualized — comparable across assets. Power σ values will
# appear very large relative to financial assets, confirming electricity is
# the most volatile commodity class. σ should be consistent with the hourly
# volatilities computed in Task 4.
#
# MONTE CARLO SIMULATION:
# The fan chart shows how price uncertainty expands over the 30-day horizon
# before stabilizing — the OU process reaches its stationary distribution
# (the long-run equilibrium) after roughly 3–5 half-lives. The convergence
# of the median toward μ and the stabilization of the percentile bands are
# signatures of mean reversion — a key distinction from a random walk (where
# uncertainty expands forever).
#
# MODEL LIMITATIONS:
# 1. Single regime: one μ and κ for all seasons — real power prices have
#    different equilibria in summer vs winter
# 2. Log-normal assumption: cannot naturally produce negative prices (we
#    filtered them out before fitting)
# 3. Constant volatility: σ is assumed fixed, but ERCOT vol is clearly
#    higher in summer (as shown in Task 4)
# 4. No seasonality: a production model would add a deterministic seasonal
#    component S(t) to the mean: dP = κ(S(t) - P)dt + σdW
# These limitations are well-known and motivate the more sophisticated
# models cQuant builds — regime-switching, seasonal OU, and jump-diffusion
# processes that better capture power price dynamics.
# =============================================================================
# %%
