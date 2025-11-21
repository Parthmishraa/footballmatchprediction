# --- FULL COLAB-SAFE STREAMLIT APP ---

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Sports Prediction & Betting Toolkit", layout="wide")


# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------

def best_series(df, cols):
    if not cols:
        return pd.Series([np.nan]*len(df))
    return df[cols].astype(float).max(axis=1, skipna=True)


def normalize_implied(odds_h, odds_d, odds_a):
    imp_h = 1.0 / odds_h
    imp_d = 1.0 / odds_d
    imp_a = 1.0 / odds_a

    df_imp = pd.DataFrame({"h": imp_h, "d": imp_d, "a": imp_a}).replace([np.inf, -np.inf], np.nan).fillna(0)
    s = df_imp.sum(axis=1).replace(0, np.nan)
    df_norm = df_imp.div(s, axis=0).fillna(df_imp.mean())
    return df_norm["h"], df_norm["d"], df_norm["a"]


def kelly_fraction(p, odds, cap):
    if odds <= 1:
        return 0.0
    b = odds - 1
    q = 1 - p
    f = (b*p - q)/b if b>0 else 0
    if f <= 0:
        return 0
    return min(f, cap)


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader("Upload sports dataset (CSV)", type=["csv"])

edge_thr = st.sidebar.slider("Edge Threshold", 0.0, 0.5, 0.14, step=0.01)
kelly_cap = st.sidebar.slider("Kelly Cap", 0.0, 0.2, 0.05, step=0.01)
abs_cap = st.sidebar.slider("Absolute Stake Cap", 0.01, 0.5, 0.10, step=0.01)
min_odds = st.sidebar.slider("Minimum Odds", 1.0, 3.0, 1.0, step=0.05)
avoid_draws = st.sidebar.checkbox("Avoid Draw Bets", True)

strategy = st.sidebar.selectbox(
    "Choose Strategy",
    ["Ensemble (Recommended)", "Value Edge (ML)", "Poisson Only", "Dynamic Kelly"]
)

run_button = st.sidebar.button("Run Simulation")


# -----------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------

st.title("⚽ Sports Prediction & Betting Toolkit (Colab Version)")

if uploaded is None:
    st.warning("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded)
st.success("Dataset loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head())


# -----------------------------------------------------
# AUTO-DETECT COMMON COLUMNS
# -----------------------------------------------------

possible_home = ["HomeTeam","home","team_home"]
possible_away = ["AwayTeam","away","team_away"]
possible_result = ["FTR","Result","result"]
possible_date = ["Date","date"]

def detect(col_list):
    for c in col_list:
        if c in df.columns:
            return c
    return None


home_col = detect(possible_home)
away_col = detect(possible_away)
res_col  = detect(possible_result)
date_col = detect(possible_date)

if home_col is None or away_col is None:
    st.error("Cannot detect home/away team columns.")
    st.stop()

# -----------------------------------------------------
# ODDS DETECTION
# -----------------------------------------------------

all_cols = df.columns.tolist()

home_odds = [c for c in all_cols if c.upper().endswith("H") and c not in ["FTHG"]]
draw_odds = [c for c in all_cols if c.upper().endswith("D")]
away_odds = [c for c in all_cols if c.upper().endswith("A") and c not in ["FTAG"]]

df["odds_home_best"] = best_series(df, home_odds)
df["odds_draw_best"] = best_series(df, draw_odds)
df["odds_away_best"] = best_series(df, away_odds)

df["imp_h"], df["imp_d"], df["imp_a"] = normalize_implied(
    df["odds_home_best"], df["odds_draw_best"], df["odds_away_best"]
)


# -----------------------------------------------------
# MODEL PROBS (fallback random probabilities)
# -----------------------------------------------------

if not {"p_H","p_D","p_A"}.issubset(df.columns):
    st.warning("No model probs found! Using fallback pseudo probabilities.")
    np.random.seed(42)
    raw = np.random.rand(len(df), 3)
    raw = raw / raw.sum(axis=1, keepdims=True)
    df["p_H"], df["p_D"], df["p_A"] = raw[:,0], raw[:,1], raw[:,2]


# -----------------------------------------------------
# ENSEMBLE EDGE
# -----------------------------------------------------

df["edge_ML_H"] = df["p_H"] - df["imp_h"]
df["edge_ML_D"] = df["p_D"] - df["imp_d"]
df["edge_ML_A"] = df["p_A"] - df["imp_a"]

df["best_edge"] = df[["edge_ML_H","edge_ML_D","edge_ML_A"]].max(axis=1)
df["best_side"] = df[["edge_ML_H","edge_ML_D","edge_ML_A"]].idxmax(axis=1).str[-1]


# -----------------------------------------------------
# SHOW TOP VALUE MATCHES
# -----------------------------------------------------

st.subheader("Top Value Matches")
top_matches = df[df["best_edge"] > edge_thr].sort_values("best_edge", ascending=False).head(20)
st.dataframe(top_matches[[date_col, home_col, away_col, "best_side", "best_edge", "odds_home_best","odds_away_best"]])


# -----------------------------------------------------
# SIMULATION ENGINE
# -----------------------------------------------------

if run_button:

    bankroll = 10000
    records = []

    for idx, r in top_matches.iterrows():

        side = r["best_side"]
        if side == "D" and avoid_draws:
            continue

        if side == "H":
            odds = r["odds_home_best"]
            p = r["p_H"]
        else:
            odds = r["odds_away_best"]
            p = r["p_A"]

        if odds < min_odds:
            continue

        f = kelly_fraction(p, odds, kelly_cap)
        f = min(f, abs_cap)
        stake = bankroll * f

        if res_col in r:
            actual = r[res_col]
        else:
            actual = np.random.choice(["H","D","A"])

        win = (actual == side)
        profit = stake*(odds-1) if win else -stake
        bankroll += profit

        records.append({
            "Date": r.get(date_col),
            "Home": r.get(home_col),
            "Away": r.get(away_col),
            "Side": side,
            "Odds": odds,
            "Stake": stake,
            "Profit": profit,
            "Bankroll": bankroll
        })

    audit = pd.DataFrame(records)

    st.subheader("Simulation Results")
    st.dataframe(audit)

    st.success(f"Final Bankroll: {bankroll:.2f}")
    st.success(f"Net Profit: {bankroll - 10000:.2f}")

    st.line_chart(audit["Bankroll"])

    st.download_button(
        "Download Audit CSV",
        audit.to_csv(index=False).encode("utf-8"),
        "audit.csv",
        "text/csv"
    )
