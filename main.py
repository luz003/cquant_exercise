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

# %%
