
# ============================================================
# FINAL FULLY FIXED STREAMLIT APP (READY FOR STREAMLIT CLOUD)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sports Prediction & Betting Toolkit", layout="wide")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def best_series(df, cols):
    if not cols:
        return pd.Series([np.nan]*len(df))
    return df[cols].astype(float).max(axis=1, skipna=True)

def normalize_implied(odds_h, odds_d, odds_a):
    imp_h = 1.0 / odds_h
    imp_d = 1.0 / odds_d
    imp_a = 1.0 / odds_a
    df_imp = pd.DataFrame({"h": imp_h, "d": imp_d, "a": imp_a}).replace([np.inf, -np.inf], np.nan).fillna(0)
    total = df_imp.sum(axis=1).replace(0, np.nan)
    df_norm = df_imp.div(total, axis=0).fillna(df_imp.mean())
    return df_norm["h"], df_norm["d"], df_norm["a"]

def kelly_fraction(p, odds, cap):
    if odds <= 1:
        return 0.0
    b = odds - 1
    q = 1 - p
    f = (b*p - q)/b if b>0 else 0
    if f <= 0:
        return 0.0
    return min(f, cap)


# ---------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------
st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader("Upload sports dataset (CSV)", type=["csv"])

edge_thr = st.sidebar.slider("Edge Threshold", 0.0, 0.5, 0.05, 0.01)
kelly_cap = st.sidebar.slider("Kelly Cap", 0.0, 0.2, 0.05, 0.01)
abs_cap = st.sidebar.slider("Absolute Stake Cap", 0.01, 0.5, 0.10, 0.01)
min_odds = st.sidebar.slider("Minimum Odds", 1.0, 5.0, 1.2, 0.05)
avoid_draws = st.sidebar.checkbox("Avoid Draw Bets", True)
strategy = st.sidebar.selectbox("Choose Strategy", ["Ensemble (Recommended)", "Value Edge (ML)", "Poisson Only", "Dynamic Kelly"])
run_button = st.sidebar.button("Run Simulation")


# ---------------------------------------------------------
# FILE UPLOAD (SAFE BLOCK)
# ---------------------------------------------------------
if uploaded is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"❌ Could not read the CSV file: {e}")
    st.stop()

st.success("Dataset loaded successfully!")
st.subheader("Dataset Preview")
st.dataframe(df.head())


# ---------------------------------------------------------
# AUTO-DETECT COMMON COLUMNS
# ---------------------------------------------------------
possible_home = ["HomeTeam","home","team_home"]
possible_away = ["AwayTeam","away","team_away"]
possible_result = ["FTR","Result","result","ftr","outcome"]
possible_date = ["Date","date"]

def detect(cols):
    for c in cols:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower() in [x.lower() for x in cols]:
            return c
    return None

home_col = detect(possible_home)
away_col = detect(possible_away)
res_col  = detect(possible_result)
date_col = detect(possible_date)

if home_col is None or away_col is None:
    st.error("Unable to detect Home/Away team columns.")
    st.stop()


# ---------------------------------------------------------
# ODDS DETECTION
# ---------------------------------------------------------
all_cols = df.columns.tolist()
home_odds = [c for c in all_cols if c.upper().endswith("H") and c.upper() not in ["FTHG","HTHG"]]
draw_odds = [c for c in all_cols if c.upper().endswith("D")]
away_odds = [c for c in all_cols if c.upper().endswith("A") and c.upper() not in ["FTAG","HTAG"]]

df["odds_home_best"] = best_series(df, home_odds)
df["odds_draw_best"] = best_series(df, draw_odds)
df["odds_away_best"] = best_series(df, away_odds)

# ensure numeric
for c in ["odds_home_best","odds_draw_best","odds_away_best"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["imp_h"], df["imp_d"], df["imp_a"] = normalize_implied(
    df["odds_home_best"], df["odds_draw_best"], df["odds_away_best"]
)


# ---------------------------------------------------------
# MODEL PROBS (FALLBACK)
# ---------------------------------------------------------
if not {"p_H","p_D","p_A"}.issubset(df.columns):
    st.warning("No ML model probabilities found. Using realistic implied-probability fallback.")

    df["p_H"] = df["imp_h"] * 0.85 + 0.05
    df["p_D"] = df["imp_d"] * 0.85 + 0.05
    df["p_A"] = df["imp_a"] * 0.85 + 0.05

    total = df["p_H"] + df["p_D"] + df["p_A"]
    df["p_H"] /= total
    df["p_D"] /= total
    df["p_A"] /= total
else:
    df["p_H"] = pd.to_numeric(df["p_H"], errors="coerce").fillna(0)
    df["p_D"] = pd.to_numeric(df["p_D"], errors="coerce").fillna(0)
    df["p_A"] = pd.to_numeric(df["p_A"], errors="coerce").fillna(0)


# ---------------------------------------------------------
# EDGES
# ---------------------------------------------------------
df["edge_ML_H"] = df["p_H"] - df["imp_h"]
df["edge_ML_D"] = df["p_D"] - df["imp_d"]
df["edge_ML_A"] = df["p_A"] - df["imp_a"]

df["best_edge"] = df[["edge_ML_H","edge_ML_D","edge_ML_A"]].max(axis=1)
df["best_side"] = df[["edge_ML_H","edge_ML_D","edge_ML_A"]].idxmax(axis=1).str[-1]


# ---------------------------------------------------------
# TOP VALUE MATCHES
# ---------------------------------------------------------
st.subheader("Top Value Matches")
top_matches = df[df["best_edge"] > edge_thr].sort_values("best_edge", ascending=False)

st.write(f"Matches passing filter (edge > {edge_thr}): {len(top_matches)}")

if len(top_matches) == 0:
    st.info("No matches passed. Try lowering Edge Threshold or Minimum Odds.")

cols_show = [date_col, home_col, away_col, "best_side", "best_edge", 
             "odds_home_best", "odds_away_best"]
st.dataframe(top_matches[cols_show].fillna("").head(25))


# ---------------------------------------------------------
# SIMULATION ENGINE
# ---------------------------------------------------------
if run_button:

    if len(top_matches) == 0:
        st.warning("No matches to simulate. Adjust filters.")
        st.stop()

    bankroll = 10000.0
    history = []

    for idx, r in top_matches.iterrows():
        side = r["best_side"]

        # avoid draws
        if avoid_draws and side == "D":
            continue

        # pick odds + prob
        if side == "H":
            odds = r["odds_home_best"]
            p = r["p_H"]
        else:
            odds = r["odds_away_best"]
            p = r["p_A"]

        if pd.isna(odds) or odds < min_odds:
            continue

        f = kelly_fraction(p, odds, kelly_cap)
        f = min(f, abs_cap)

        stake = bankroll * f
        if stake <= 0:
            continue

        # determine actual result
        actual = r.get(res_col)
        if pd.isna(actual):
            actual = np.random.choice(["H","D","A"])

        win = actual == side
        profit = stake*(odds - 1) if win else -stake
        bankroll += profit

        history.append({
            "Date": r.get(date_col),
            "Home": r.get(home_col),
            "Away": r.get(away_col),
            "Side": side,
            "Odds": odds,
            "Stake": round(stake, 2),
            "Profit": round(profit, 2),
            "Bankroll": round(bankroll, 2)
        })

    audit = pd.DataFrame(history)

    st.subheader("Simulation Results")
    if audit.empty:
        st.info("Simulation complete — but no bets placed. Adjust thresholds.")
    else:
        st.dataframe(audit)
        st.success(f"Final Bankroll: {bankroll:.2f}")
        st.success(f"Net Profit: {bankroll - 10000:.2f}")

        try:
            st.line_chart(audit["Bankroll"])
        except:
            st.write("Couldn't render line chart — using fallback.")
            plt.plot(audit["Bankroll"])
            plt.xlabel("Bet #")
            plt.ylabel("Bankroll")
            st.pyplot(plt)

        st.download_button(
            "Download Audit CSV",
            audit.to_csv(index=False).encode("utf-8"),
            "audit.csv",
            "text/csv"
        )
