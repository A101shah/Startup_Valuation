# =============================================================================
# app.py — The Valuation-Survival Paradox: A Martingale Analysis of Seed-Stage
#           Startup Contagion
# =============================================================================
# PURPOSE:
#   This Streamlit application models a seed-stage startup as a Galton-Watson
#   Branching Process, derives the "Fair Market Value" via Martingale Theory,
#   and quantifies extinction probability using the Fixed-Point of the
#   Probability Generating Function (PGF).
#
# THEORETICAL PILLARS:
#   1. Galton-Watson Branching Process  → user referral dynamics
#   2. Martingale W_n = Z_n / μ^n      → unbiased, discounted valuation
#   3. Extinction PGF fixed-point       → VC ruin probability
#   4. Scikit-learn RandomForest        → "unicorn vs zombie" classifier
#
# LIBRARIES:
#   streamlit, numpy, pandas, scipy, plotly, matplotlib, seaborn, scikit-learn
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
import math
import time

# ── Numeric / scientific ──────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import poisson, geom, nbinom, ks_2samp
from scipy.optimize import brentq

# ── Visualisation ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

# ── Machine learning ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st

warnings.filterwarnings("ignore")

# =============================================================================
# 0.  PAGE CONFIG  (must be the very first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Startup Valuation-Survival Paradox",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1.  GLOBAL STYLE INJECTION
#     We inject a small CSS block to give the app a dark, professional look
#     consistent with a quantitative-finance or VC analytics dashboard.
# =============================================================================
STYLE = """
<style>
/* ── Root palette ─────────────────────────────────────────── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #58a6ff;
    --accent2:   #f78166;
    --text:      #e6edf3;
    --muted:     #8b949e;
}

/* ── App background ───────────────────────────────────────── */
.stApp { background-color: var(--bg); color: var(--text); }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
}

/* ── Code blocks ──────────────────────────────────────────── */
code { color: var(--accent2); background: var(--surface); }

/* ── Section dividers ─────────────────────────────────────── */
hr { border-color: var(--border); }

/* ── Tab strip ────────────────────────────────────────────── */
[data-baseweb="tab"] { color: var(--muted); }
[aria-selected="true"] { color: var(--accent) !important; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# =============================================================================
# 2.  SESSION-STATE INITIALISATION
#     Streamlit re-runs the entire script on every widget interaction.
#     We use st.session_state to persist simulation results, the trained ML
#     model, and any user-defined parameters across re-runs without redundant
#     recomputation.
# =============================================================================
_STATE_DEFAULTS = {
    "sim_results":      None,   # DataFrame of 1 000 simulated startup trajectories
    "ml_dataset":       None,   # Feature-engineered DataFrame for ML classifier
    "model_trained":    False,  # Flag: has the classifier been fitted?
    "clf_pipeline":     None,   # Scikit-learn Pipeline object
    "clf_metrics":      {},     # Dict of AUC, AP, etc.
    "extinction_prob":  None,   # η computed from PGF fixed point
    "mu":               1.20,   # Default mean referral rate
    "sigma2":           0.80,   # Default referral variance
    "z0":               10,     # Seed users
    "n_gen":            25,     # Generations to simulate
    "n_sim":            1000,   # Monte Carlo paths
    "dist_name":        "Poisson",
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =============================================================================
# 3.  MATHEMATICAL CORE — PURE FUNCTIONS
#     All heavy computation is isolated here.  Functions are decorated with
#     @st.cache_data so Streamlit only recomputes when arguments change.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  OFFSPRING DISTRIBUTION SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Picklable sampler/PGF classes ─────────────────────────────────────────────
# Streamlit's @st.cache_data serialises return values via pickle.
# Python lambdas and closures are NOT picklable, so we use __call__-based
# classes instead.  Each class stores only primitive scalars (fully picklable)
# and reconstructs the numpy Generator on first use via a lazy property.

class _PoissonSampler:
    """Offspring sampler for Poisson(λ) distribution."""
    def __init__(self, lam: float):
        self.lam = lam
        self._rng = None

    def _get_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def __call__(self, n: int) -> np.ndarray:
        return self._get_rng().poisson(self.lam, size=n)


class _GeomSampler:
    """Offspring sampler for Geometric(p) distribution (support k=0,1,2,…)."""
    def __init__(self, p: float):
        self.p = p
        self._rng = None

    def _get_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def __call__(self, n: int) -> np.ndarray:
        return self._get_rng().geometric(self.p, size=n) - 1


class _NegBinSampler:
    """Offspring sampler for Negative-Binomial(r, p) distribution."""
    def __init__(self, r: float, p: float):
        self.r = r
        self.p = p
        self._rng = None

    def _get_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def __call__(self, n: int) -> np.ndarray:
        return self._get_rng().negative_binomial(self.r, self.p, size=n)


class _PoissonPGF:
    """PGF for Poisson(λ): φ(s) = exp(λ(s-1))."""
    def __init__(self, lam: float):
        self.lam = lam

    def __call__(self, s: float) -> float:
        return float(np.exp(self.lam * (s - 1)))


class _GeomPGF:
    """PGF for Geometric(p): φ(s) = p / (1 - (1-p)s)."""
    def __init__(self, p: float):
        self.p = p

    def __call__(self, s: float) -> float:
        denom = 1.0 - (1.0 - self.p) * s
        if abs(denom) < 1e-12:
            return float("nan")
        return float(self.p / denom)


class _NegBinPGF:
    """PGF for NegBin(r, p): φ(s) = (p / (1-(1-p)s))^r."""
    def __init__(self, r: float, p: float):
        self.r = r
        self.p = p

    def __call__(self, s: float) -> float:
        denom = 1.0 - (1.0 - self.p) * s
        if abs(denom) < 1e-12:
            return float("nan")
        return float((self.p / denom) ** self.r)


# ── Factory — NOT cached (returns callable objects, safe to call anywhere) ────

def make_offspring_sampler(dist_name: str, mu: float, sigma2: float):
    """
    Return a (sampler, pgf) pair for the requested offspring distribution.

    Both returned objects are fully picklable instances of the classes above,
    avoiding the AttributeError that arises when @st.cache_data tries to
    pickle local lambda/closure objects.

    Supported distributions
    -----------------------
    Poisson(λ=μ)      — organic word-of-mouth; Var = Mean.
    Geometric(p)      — heavy right tail; rare influencer referrals.
    NegBinomial(r,p)  — over-dispersed; cohort-clustered fintech referrals.
    """
    if dist_name == "Poisson":
        lam = max(mu, 1e-9)
        return _PoissonSampler(lam), _PoissonPGF(lam)

    elif dist_name == "Geometric":
        # Support k = 0,1,2,…  →  E[X] = (1-p)/p  →  p = 1/(μ+1)
        p = min(1.0 / (mu + 1.0), 1.0 - 1e-9)
        return _GeomSampler(p), _GeomPGF(p)

    else:  # Negative-Binomial — match first two moments
        sigma2_eff = sigma2 if sigma2 > mu else mu + 0.01
        r   = max(mu ** 2 / (sigma2_eff - mu), 0.1)
        p_nb = float(np.clip(mu / sigma2_eff, 1e-6, 1.0 - 1e-6))
        return _NegBinSampler(r, p_nb), _NegBinPGF(r, p_nb)


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  PGF FIXED-POINT SOLVER  →  extinction probability η
# ─────────────────────────────────────────────────────────────────────────────

def _pgf_residual(s, pgf_callable):
    """Module-level residual function for brentq — picklable, no closure."""
    return pgf_callable(s) - s


@st.cache_data(show_spinner=False)
def compute_extinction_probability(dist_name: str, mu: float,
                                   sigma2: float, z0: int) -> float:
    """
    Compute the extinction probability η = P(Z_n → 0) using the smallest
    non-negative fixed point of the PGF φ(s) = s.

    Theory
    ------
    For a Galton-Watson process:
      • If μ ≤ 1  →  η = 1  (certain extinction).
      • If μ > 1  →  η is the unique root of φ(s) = s in [0, 1).
    Starting from z₀ seeds, η_total = η^z₀.

    Implementation
    --------------
    We solve  φ(s) − s = 0  on (0, 1) via Brent's method (brentq).
    _pgf_residual is a module-level def (not a lambda) so it is safely
    picklable by @st.cache_data.
    """
    _, pgf = make_offspring_sampler(dist_name, mu, sigma2)

    if mu <= 1.0:
        return 1.0   # sub-critical or critical → certain extinction

    try:
        eta_single = brentq(
            _pgf_residual, 1e-8, 1 - 1e-8,
            args=(pgf,),
            xtol=1e-10, maxiter=500,
        )
    except (ValueError, RuntimeError):
        eta_single = 0.99   # fallback

    return float(eta_single ** z0)


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  GALTON-WATSON SIMULATION  →  Z_n and W_n trajectories
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_simulation(dist_name: str, mu: float, sigma2: float,
                   z0: int, n_gen: int, n_sim: int) -> pd.DataFrame:
    """
    Monte Carlo simulation of n_sim independent Galton-Watson processes.

    Each process starts with z0 seed users.  At every generation n, each
    alive user independently produces X offspring where X ~ chosen distribution
    with mean μ.

    Stored quantities per path per generation
    -----------------------------------------
    Z_n     : raw user count (integers ≥ 0)
    W_n     : normalised martingale value  Z_n / μ^n
    alive   : 1 if Z_n > 0 else 0

    Returns
    -------
    pd.DataFrame with columns [sim_id, generation, Z, W, alive]
    """
    rng_seed = 42           # reproducible for the cached result
    np.random.seed(rng_seed)

    sampler, _ = make_offspring_sampler(dist_name, mu, sigma2)

    records = []
    for sim_id in range(n_sim):
        z = int(z0)
        for gen in range(n_gen + 1):
            denom = mu ** gen if mu > 0 else 1.0
            w = z / denom if denom > 0 else 0.0
            records.append({
                "sim_id":     sim_id,
                "generation": gen,
                "Z":          z,
                "W":          w,
                "alive":      int(z > 0),
            })
            if z == 0:
                # Once extinct, fill remaining generations with 0
                for g2 in range(gen + 1, n_gen + 1):
                    denom2 = mu ** g2
                    records.append({
                        "sim_id":     sim_id,
                        "generation": g2,
                        "Z":          0,
                        "W":          0.0,
                        "alive":      0,
                    })
                break
            # Propagate to next generation
            # Sum of z independent offspring draws
            z = int(sampler(z).sum())

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  N-STEP TRANSITION PROBABILITY MATRIX  (analytical approximation)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def nstep_transition_matrix(dist_name: str, mu: float, sigma2: float,
                             max_k: int = 30, n_steps: int = 5) -> np.ndarray:
    """
    Compute the n-step transition probability matrix P^n for the chain
    (Z_n) truncated to states {0, 1, …, max_k}.

    Approach
    --------
    Build the 1-step matrix P where P[i, j] = P(Z_{t+1}=j | Z_t=i).
    Row i is the i-fold convolution of the offspring pmf evaluated at j.
    Then raise P to the n-th power.

    This is computationally intensive for large max_k so we limit to max_k=30.
    """
    sampler, _ = make_offspring_sampler(dist_name, mu, sigma2)

    # Approximate offspring pmf by simulation
    offspring_pmf = np.zeros(max_k + 1)
    samples = sampler(200_000)
    for k in range(max_k + 1):
        offspring_pmf[k] = np.mean(samples == k)
    offspring_pmf /= offspring_pmf.sum()   # normalise

    K = max_k + 1
    P = np.zeros((K, K))
    P[0, 0] = 1.0    # 0 is absorbing state

    def convolve_pmf(pmf, k):
        """k-fold convolution of pmf with itself."""
        result = np.zeros(K)
        result[0] = 1.0
        for _ in range(k):
            new = np.zeros(K)
            for x in range(K):
                if result[x] == 0:
                    continue
                for y in range(K - x):
                    if y < len(pmf):
                        new[x + y] += result[x] * pmf[y]
            result = new
        return result

    for i in range(1, K):
        row = convolve_pmf(offspring_pmf, i)
        row_sum = row.sum()
        if row_sum > 0:
            P[i] = row / row_sum
        else:
            P[i, 0] = 1.0

    # Matrix power  P^n_steps
    Pn = np.linalg.matrix_power(P, n_steps)
    return Pn


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  FEATURE ENGINEERING  →  ML dataset for "Unicorn vs Zombie" classifier
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_ml_dataset(sim_df: pd.DataFrame, n_gen: int) -> pd.DataFrame:
    """
    Construct a feature table for the binary classification task:
        label = 1  if startup is 'alive' at generation n_gen  (Unicorn)
                0  otherwise                                   (Zombie/Extinct)

    Features extracted from the EARLY trajectory (generations 0-5 only)
    to simulate the VC diligence scenario where only early KPIs are visible.

    Features
    --------
    w_mean_early    : mean W_n over first 5 generations
    w_std_early     : std  W_n over first 5 generations
    w_slope_early   : OLS slope of W_n vs generation (momentum)
    z_max_early     : peak user count in first 5 gens
    z_at_gen5       : user count at generation 5
    extinct_before5 : 1 if went to 0 before gen 5
    log_z0          : log(Z_0)   (seed size signal)
    """
    early = sim_df[sim_df["generation"] <= 5].copy()
    final = sim_df[sim_df["generation"] == n_gen][["sim_id", "alive"]].copy()
    final = final.rename(columns={"alive": "label"})

    def _features(grp):
        w = grp["W"].values
        z = grp["Z"].values
        g = grp["generation"].values
        slope = np.polyfit(g, w, 1)[0] if len(g) > 1 else 0.0
        return pd.Series({
            "w_mean_early":     float(np.mean(w)),
            "w_std_early":      float(np.std(w) + 1e-9),
            "w_slope_early":    float(slope),
            "z_max_early":      float(np.max(z)),
            "z_at_gen5":        float(z[-1]) if len(z) > 0 else 0.0,
            "extinct_before5":  float(np.any(z == 0)),
            "log_z0":           float(np.log1p(z[0])) if len(z) > 0 else 0.0,
        })

    feats = early.groupby("sim_id").apply(_features).reset_index()
    ml_df = feats.merge(final, on="sim_id")
    return ml_df


# =============================================================================
# 4.  SIDEBAR — USER INPUTS
# =============================================================================

with st.sidebar:
    st.markdown("## 🔬 Model Parameters")
    st.caption("Adjust the underlying stochastic process before running.")

    dist_name = st.selectbox(
        "Offspring Distribution",
        ["Poisson", "Geometric", "NegBinomial"],
        index=["Poisson", "Geometric", "NegBinomial"].index(
            st.session_state["dist_name"]),
        help=(
            "**Poisson** — organic word-of-mouth (σ²=μ). "
            "**Geometric** — heavy-tail viral growth. "
            "**NegBinomial** — cohort-clustered fintech referrals."
        ),
    )

    mu = st.slider(
        "Mean Referral Rate μ",
        min_value=0.50, max_value=3.00, step=0.05,
        value=float(st.session_state["mu"]),
        help="Average number of new users each existing user recruits per generation.",
    )

    sigma2 = st.slider(
        "Referral Variance σ²",
        min_value=0.10, max_value=5.00, step=0.10,
        value=float(st.session_state["sigma2"]),
        help="Higher variance → fatter tail → higher extinction risk despite positive mean.",
    )

    z0 = st.number_input(
        "Seed Users Z₀",
        min_value=1, max_value=500, value=int(st.session_state["z0"]),
        help="Number of users at launch (generation 0).",
    )

    n_gen = st.slider(
        "Generations to Simulate",
        min_value=10, max_value=50, step=5,
        value=int(st.session_state["n_gen"]),
    )

    n_sim = st.select_slider(
        "Monte Carlo Paths",
        options=[200, 500, 1000, 2000],
        value=int(st.session_state["n_sim"]),
    )

    st.markdown("---")

    run_btn = st.button("▶ Run Simulation", width='stretch', type="primary")

    if run_btn:
        # Persist parameters to session state
        for k, v in zip(
            ["dist_name", "mu", "sigma2", "z0", "n_gen", "n_sim"],
            [dist_name, mu, sigma2, z0, n_gen, n_sim],
        ):
            st.session_state[k] = v

        with st.spinner("Running Monte Carlo simulation…"):
            st.session_state["sim_results"] = run_simulation(
                dist_name, mu, sigma2, z0, n_gen, n_sim)

        with st.spinner("Computing extinction probability…"):
            st.session_state["extinction_prob"] = compute_extinction_probability(
                dist_name, mu, sigma2, z0)

        with st.spinner("Building ML feature dataset…"):
            st.session_state["ml_dataset"] = build_ml_dataset(
                st.session_state["sim_results"], n_gen)

        st.session_state["model_trained"] = False
        st.success("Simulation complete!", icon="✅")

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Referencing: Athreya & Ney (1972) "
        "*Branching Processes*; Williams (1991) "
        "*Probability with Martingales*.</small>",
        unsafe_allow_html=True,
    )

# =============================================================================
# 5.  MAIN CONTENT — TABBED LAYOUT
# =============================================================================

st.markdown("# 🚀 The Valuation-Survival Paradox")
st.markdown(
    "### A Martingale Analysis of Seed-Stage Startup Contagion\n"
    "_Galton-Watson Branching Processes · Martingale Theory · VC Risk Analytics_"
)
st.markdown("---")

if st.session_state["sim_results"] is None:
    st.info(
        "👈  Configure the model parameters in the sidebar and click "
        "**▶ Run Simulation** to begin.",
        icon="ℹ️",
    )
    st.stop()

# Alias session-state objects for readability
sim_df       = st.session_state["sim_results"]
eta          = st.session_state["extinction_prob"]
params_mu    = st.session_state["mu"]
params_z0    = st.session_state["z0"]
params_n_gen = st.session_state["n_gen"]
params_n_sim = st.session_state["n_sim"]
params_dist  = st.session_state["dist_name"]
params_sig2  = st.session_state["sigma2"]

# ── Tab structure ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview & Data Quality",
    "🌱 Branching Process",
    "📐 Martingale Valuation",
    "💀 Extinction Risk",
    "🤖 ML Classifier",
    "📖 Theory Reference",
])

# =============================================================================
# TAB 0 — OVERVIEW & DATA QUALITY
# =============================================================================
with tabs[0]:
    st.markdown("## Simulation Overview & Data Quality Audit")
    st.markdown(
        """
        Before diving into the models, we run a quick **health check** on the
        data our simulation just produced. Think of this like a doctor checking
        your vitals before an operation — we want to make sure everything looks
        normal before drawing any big conclusions.

        We check three things:
        * **Row-level completeness** — is every startup tracked at every time step?
          *(Like making sure no page is torn out of a logbook.)*
        * **Value-range checks** — are all user counts and valuations ≥ 0?
          *(A startup can't have negative users — if it does, something went wrong.)*
        * **Distributional sanity** — do the summary numbers match what the math predicts?
          *(If we expect an average of 50 users but see 50,000, something is off.)*
        """
    )

    # ── KPI strip ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    total_rows  = len(sim_df)
    extinct_pct = sim_df[sim_df["generation"] == params_n_gen]["alive"].mean()
    missing_pct = sim_df.isnull().mean().mean() * 100
    max_Z       = sim_df["Z"].max()
    max_W       = sim_df["W"].max()

    c1.metric("Total Rows",
              f"{total_rows:,}",
              help="Total data points recorded across all simulated startups and generations.")
    c2.metric(f"Survival Rate (gen {params_n_gen})",
              f"{extinct_pct:.1%}",
              help="Fraction of startups still alive at the final generation.")
    c3.metric("Missing Values",
              f"{missing_pct:.2f}%",
              help="Percentage of data cells with no recorded value. 0% = perfect dataset.")
    c4.metric("Peak User Count",
              f"{int(max_Z):,}",
              help="The single highest user count seen across all simulated startups.")
    c5.metric("Peak W_n",
              f"{max_W:.2f}",
              help="The highest normalised valuation (W) observed. Large values = unicorn candidates.")

    st.markdown("---")

    # ── Data preview ──────────────────────────────────────────────────────────
    st.markdown("#### Raw Simulation Data (first 200 rows)")
    st.markdown(
        """
        This is the raw output of the simulation — one row per startup per time step.

        | Column | Technical meaning | Plain English |
        |---|---|---|
        | **sim_id** | Simulation path identifier | Which startup (1 to N) |
        | **generation** | Discrete time step n | Which week/cohort cycle |
        | **Z** | Raw user count Z_n | How many active users the startup has right now |
        | **W** | Normalised value Z_n / μⁿ | The startup's "true strength" with growth trend removed |
        | **alive** | 1 if Z > 0, else 0 | Is the startup still operating? (1 = yes, 0 = bankrupt) |

        Once a startup hits Z = 0 it stays there forever — this is called the
        **absorbing state** *(think of it as the startup's lights going out — once
        you've run out of users, you can't spontaneously get them back)*.
        """
    )
    st.dataframe(sim_df.head(200), width='stretch', height=280)

    st.markdown("---")

    # ── Missing Value Analysis ─────────────────────────────────────────────────
    # Since our simulation generates perfectly clean data, a traditional
    # missingness heatmap would be entirely blank (uninformative).
    # Instead we show a per-column completeness bar chart — green = 100% present,
    # which is the expected and correct result for synthetic simulation data.
    # This is still the standard data-quality check; it just renders meaningfully
    # for complete datasets rather than showing an empty grid.

    st.markdown("#### Data Completeness Check")
    st.markdown(
        """
        In real-world analytics, the first thing a data scientist checks is
        **how complete the dataset is** — are there gaps, nulls, or missing
        entries? *(Imagine a survey where some respondents skipped questions —
        those blank answers are 'missing values' and can skew your results.)*

        The chart below shows the **completeness percentage for each column**.
        Green bars at 100% mean the simulation produced a perfectly clean dataset
        with zero gaps — exactly what we expect from a controlled Monte Carlo run.
        If any bar dropped below 100%, we would need to investigate and fix the
        data pipeline before trusting any results.
        """
    )

    # Compute per-column completeness
    completeness = (1 - sim_df.isnull().mean()) * 100
    completeness_df = completeness.reset_index()
    completeness_df.columns = ["Column", "Completeness (%)"]

    # Colour each bar: 100% → green, anything less → red
    bar_colors = [
        "#3fb950" if v == 100.0 else "#f78166"
        for v in completeness_df["Completeness (%)"]
    ]

    fig_mv = go.Figure(go.Bar(
        x=completeness_df["Column"],
        y=completeness_df["Completeness (%)"],
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in completeness_df["Completeness (%)"]],
        textposition="outside",
    ))
    fig_mv.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        title="Column Completeness — 100% = No Missing Values (green = perfect ✓)",
        title_font_size=12,
        yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"),
        showlegend=False,
        height=320,
    )
    st.plotly_chart(fig_mv, width='stretch')

    # Also show a small summary confirmation message
    n_missing_total = int(sim_df.isnull().sum().sum())
    if n_missing_total == 0:
        st.success(
            f"✅ **Perfect data quality confirmed** — 0 missing values across "
            f"all {len(sim_df.columns)} columns and {len(sim_df):,} rows. "
            f"*(Every field in every row has a valid recorded value — the dataset "
            f"is 100% complete and ready for analysis.)*",
        )
    else:
        st.warning(
            f"⚠️ {n_missing_total:,} missing values detected across the dataset. "
            f"Investigate before proceeding."
        )

    st.markdown("---")

    # ── Descriptive statistics ─────────────────────────────────────────────────
    st.markdown("#### Descriptive Statistics")
    st.markdown(
        """
        This table summarises the key numbers across the entire simulation dataset.
        Think of it like a **report card** for the data — it tells you the typical
        values, the spread, and the extremes at a glance.

        *(In plain terms: if Z were exam scores, 'mean' is the class average,
        'std' is how spread out the scores are, 'min/max' are the worst and best
        scores, and '25%/50%/75%' split the class into quarters.)*

        🔍 **What to look for here:**
        - The **mean of Z** should be roughly Z₀ × μⁿ on average across all generations.
        - The **std of Z** will be very large — this is the paradox in action. *(Most
          startups go to zero, but a few explode — that gap between the typical and the
          exceptional is what makes the standard deviation so large.)*
        - The **mean of `alive`** tells you what fraction of startups survived overall
          across all generations. *(E.g. 0.45 means 45% of all recorded snapshots were
          from still-operating startups.)*
        """
    )
    st.dataframe(
        sim_df[["Z", "W", "alive"]].describe().round(4),
        width='stretch',
    )

    st.markdown("---")

    # ── Outlier detection: W_n box-plot (Plotly) ──────────────────────────────
    st.markdown("#### Spread & Outlier Detection — W_n Distribution per Generation")
    st.markdown(
        """
        This **box-plot** shows how the startup valuations (W_n) spread out
        over time. Each box covers the middle 50% of startups at that generation;
        the line inside is the median; dots floating above are the outliers.

        *(Imagine ranking 1,000 students by test score every week. The box shows
        where most students score. The floating dots above are the exceptional
        overachievers — in startup terms, these are the unicorn candidates.)*

        🔍 **What to look for:**
        - Early generations: boxes are tight — startups haven't had time to diverge yet.
        - Later generations: boxes collapse toward zero *(most startups are dead or dying)*
          but extreme outlier dots appear far above *(the survivors are pulling very far
          ahead — the rich-get-richer effect of compounding growth)*.
        - This widening gap is the visual signature of the **Valuation-Survival Paradox**:
          identical starting conditions produce wildly different outcomes.
        """
    )

    sample_gens = sorted(
        sim_df["generation"].unique()[:: max(1, params_n_gen // 8)]
    )
    df_box = sim_df[sim_df["generation"].isin(sample_gens)]

    fig_box = px.box(
        df_box, x="generation", y="W",
        color_discrete_sequence=["#58a6ff"],
        labels={"W": "W_n (normalised valuation)", "generation": "Generation"},
        title="Distribution of Martingale Value W_n Across Generations",
    )
    fig_box.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3", title_font_size=13,
    )
    st.plotly_chart(fig_box, width='stretch')

# =============================================================================
# TAB 1 — BRANCHING PROCESS
# =============================================================================
with tabs[1]:
    st.markdown("## 🌱 Galton-Watson Branching Process")
    st.markdown("### Mathematical Framework")
    st.markdown(
        "A **Galton-Watson Branching Process** {Zₙ}ₙ≥₀ models population "
        "growth where each individual independently produces random offspring. "
        "It is defined recursively:"
    )
    st.latex(r"Z_0 = z_0, \qquad Z_{n+1} = \sum_{i=1}^{Z_n} X_i^{(n)}")
    st.markdown(
        "where the X_i^(n) are i.i.d. non-negative integer-valued random variables "
        "(the **offspring distribution**) with mean and variance:"
    )
    st.latex(r"\mu = \mathbb{E}[X], \qquad \sigma^2 = \operatorname{Var}(X)")
    st.markdown(
        "**Business interpretation:** Zₙ is the number of active users at "
        "generation n (e.g., a weekly cohort cycle). Each user independently "
        "refers X new users — X is the 'viral coefficient' of the startup.\n\n"
        "**Regime classification:**\n\n"
        "| Regime | Condition | Outcome |\n"
        "|---|---|---|\n"
        "| Sub-critical | μ < 1 | Certain extinction |\n"
        "| Critical | μ = 1 | Certain extinction (slower) |\n"
        "| Super-critical | μ > 1 | Positive survival probability |"
    )

    st.markdown("---")

    # ── Fan chart: Z_n trajectories ────────────────────────────────────────────
    st.markdown("#### Z_n Fan Chart — 50 Representative Paths")
    st.markdown(
        """
        Each line on this chart is one simulated startup's full history from
        launch to the final generation. **Blue lines** are startups that survived;
        **red lines** are startups that went bankrupt (hit zero users).

        *(Think of this like watching 50 different horse races on the same track.
        Every horse starts at the same gate with the same average speed — but
        randomness means some bolt ahead while others stumble and stop. The
        yellow dotted line is the median — the "typical" horse at each point
        in the race.)*

        🔍 **What to look for:**
        - Most red lines drop flat to zero early — extinction happens fast when it happens.
        - A small number of blue lines climb steeply — these are the unicorn candidates
          that compounded their user base generation after generation.
        - Notice that both outcomes came from **identical starting conditions** —
          that is the paradox this model is designed to quantify.
        """
    )

    n_show = 50
    sample_ids = sim_df["sim_id"].drop_duplicates().sample(n=n_show, random_state=1)
    df_fan = sim_df[sim_df["sim_id"].isin(sample_ids)]

    final_status = (
        sim_df[sim_df["generation"] == params_n_gen][["sim_id", "alive"]]
        .set_index("sim_id")["alive"].to_dict()
    )

    fig_fan = go.Figure()
    for sid in sample_ids:
        path = df_fan[df_fan["sim_id"] == sid].sort_values("generation")
        alive = final_status.get(sid, 0)
        fig_fan.add_trace(go.Scatter(
            x=path["generation"], y=path["Z"],
            mode="lines",
            line=dict(
                color="rgba(88,166,255,0.35)" if alive else "rgba(247,129,102,0.25)",
                width=1,
            ),
            showlegend=False,
        ))

    # Overlay median
    median_Z = sim_df.groupby("generation")["Z"].median().reset_index()
    fig_fan.add_trace(go.Scatter(
        x=median_Z["generation"], y=median_Z["Z"],
        mode="lines", name="Median Z_n",
        line=dict(color="#f0e130", width=2.5, dash="dot"),
    ))

    fig_fan.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_title="Generation n", yaxis_title="User Count Z_n",
        title=f"Branching Process Trajectories (μ={params_mu:.2f}, "
              f"dist={params_dist})",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    st.plotly_chart(fig_fan, width='stretch')

    st.markdown("---")

    # ── Survival curve ─────────────────────────────────────────────────────────
    st.markdown("#### Empirical Survival Curve — What Fraction of Startups Are Still Alive?")
    st.markdown(
        """
        This chart tracks the **percentage of startups still operating** at each
        generation. The green line falls as more and more startups go bankrupt over time.

        *(Imagine planting 1,000 seedlings in a garden. This curve shows how many
        plants are still alive each week. Some die in the first days; others survive
        for months. The curve flattens when the remaining plants have found their
        footing — those are the ones likely to become trees.)*

        The **red dashed horizontal line** is the theoretical long-run survival rate
        predicted by the mathematics — it is where the curve should eventually flatten.
        *(If the simulation is well-calibrated, the green line should approach the
        red line as generations pass — like a prediction vs reality check.)*

        🔍 **What to look for:**
        - A steep early drop = most startups die young *(normal — most real startups
          also fail in the first 1-2 years)*.
        - The curve flattening = the "survivors' club" has stabilised. Those still alive
          have enough momentum to keep going.
        - If the green line stays well above the red line for many generations, your
          μ and Z₀ settings are giving startups a fighting chance.
        """
    )

    surv_curve = (
        sim_df.groupby("generation")["alive"].mean().reset_index()
        .rename(columns={"alive": "survival_prob"})
    )

    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=surv_curve["generation"], y=surv_curve["survival_prob"],
        mode="lines+markers",
        line=dict(color="#3fb950", width=2),
        marker=dict(size=4),
        name="Empirical P(Z_n > 0)",
    ))
    if eta is not None:
        long_run = 1 - eta
        fig_surv.add_hline(
            y=long_run,
            line_dash="dash", line_color="#f78166",
            annotation_text=f"Theoretical limit: {long_run:.3f}",
            annotation_font_color="#f78166",
        )

    fig_surv.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_title="Generation n",
        yaxis_title="P(Z_n > 0)",
        yaxis_range=[0, 1.05],
        title="Empirical Survival Probability Over Time",
    )
    st.plotly_chart(fig_surv, width='stretch')

    st.markdown("---")

    # ── n-step Transition Matrix heatmap ──────────────────────────────────────
    st.markdown("#### n-Step Transition Matrix — Where Will a Startup Be in n Steps?")
    st.markdown(
        """
        This heatmap answers the question: **"If a startup has i users today,
        what is the probability it will have j users n generations from now?"**

        *(Think of it like a weather forecast table. The row is today's weather
        (sunny = i users). The column is the weather n days ahead (j users).
        Each cell is the probability of that particular future. The darker/hotter
        the colour, the more likely that outcome.)*

        **Row 0 is always 100% red** — once a startup hits zero users *(bankrupt)*,
        it stays there forever. This is called the **absorbing state**.
        *(Like a company that's been dissolved — it doesn't spontaneously
        re-incorporate on its own.)*

        Use the slider below to change how many steps (generations) ahead
        you are forecasting. More steps = more uncertainty = more spread-out colours.
        """
    )

    n_steps_vis = st.slider("n-step horizon", 1, 10, 5,
                             key="nstep_slider")
    with st.spinner("Computing transition matrix…"):
        Pn = nstep_transition_matrix(
            params_dist, params_mu, params_sig2,
            max_k=25, n_steps=n_steps_vis)

    fig_tm, ax_tm = plt.subplots(figsize=(9, 5))
    fig_tm.patch.set_facecolor("#0d1117")
    ax_tm.set_facecolor("#161b22")
    sns.heatmap(
        Pn[:15, :15],
        ax=ax_tm, cmap="YlOrRd",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="#30363d",
        cbar_kws={"shrink": 0.7},
    )
    ax_tm.set_title(
        f"P^{n_steps_vis}[i,j]  (states 0–14)",
        color="#e6edf3", fontsize=11,
    )
    ax_tm.set_xlabel("j  (next-state user count)", color="#8b949e")
    ax_tm.set_ylabel("i  (current user count)", color="#8b949e")
    ax_tm.tick_params(colors="#8b949e")
    st.pyplot(fig_tm, width='stretch')
    plt.close(fig_tm)

# =============================================================================
# TAB 2 — MARTINGALE VALUATION
# =============================================================================
with tabs[2]:
    st.markdown("## 📐 Martingale Valuation Theory")
    st.markdown("### The Fundamental Theorem of Startup Valuation")
    st.markdown(
        """
        A startup growing at 20% per generation will naturally have more users
        each week — but that growth was **already expected**. A smart investor
        wants to know: *after accounting for the expected growth, is this startup
        doing better or worse than it should be?*

        *(Think of it like adjusting a student's grades for course difficulty.
        Getting 70% in the hardest class in school might actually be better than
        getting 90% in the easiest one. We 'normalise' to make fair comparisons.)*

        We define the **normalised valuation** Wₙ — the startup's score after
        removing the expected exponential growth trend:
        """
    )
    st.latex(r"W_n = \frac{Z_n}{\mu^n}")
    st.markdown(
        "**Theorem (Martingale Property):** {Wₙ, Fₙ} is a non-negative martingale.\n\n"
        "**Proof sketch:** Condition on the filtration Fₙ (i.e., on Zₙ):"
    )
    st.latex(
        r"\mathbb{E}[W_{n+1} \mid \mathcal{F}_n]"
        r"= \mathbb{E}\!\left[\frac{Z_{n+1}}{\mu^{n+1}} \,\Big|\, \mathcal{F}_n\right]"
        r"= \frac{1}{\mu^{n+1}}\,\mathbb{E}\!\left[\sum_{i=1}^{Z_n} X_i \,\Big|\, \mathcal{F}_n\right]"
        r"= \frac{Z_n \cdot \mu}{\mu^{n+1}} = \frac{Z_n}{\mu^n} = W_n \qquad \square"
    )
    st.markdown(
        """
        **What the Martingale property means in plain English:**
        The expected value of Wₙ tomorrow equals Wₙ today — no more, no less.
        *(Like a fair coin flip: on average, your total winnings don't go up or
        down. The Martingale is the mathematical definition of a 'fair game' —
        no hidden bias in either direction.)*

        This makes Wₙ the **unbiased fundamental value** of the startup. A VC
        pricing equity proportional to Wₙ is making a mathematically fair bet —
        not overpaying for growth that was already baked in.

        The **Martingale Convergence Theorem** guarantees that Wₙ settles to a
        final value W∞ as time goes on. The gap between W₀ and E[W∞] is the
        **extinction discount** — *(how much value you expect to lose purely
        because some startups will inevitably go bust, even with good average growth)*.
        """
    )

    st.markdown("---")

    # ── Empirical verification of E[W_{n+1} | F_n] = W_n ─────────────────────
    st.markdown("#### Empirical Verification: Is the Average Wₙ Staying Flat?")
    st.markdown(
        """
        If our simulation is mathematically correct, the **average value of Wₙ
        should stay approximately constant** across all generations. This is
        the Martingale property showing up in real data.

        *(Imagine running a casino 1,000 times. If the game is truly fair — no
        house edge — the average total winnings across all players should stay
        roughly the same each round. A rising average means the house is losing;
        a falling average means players are losing. Flat = fair.)*

        The **blue line** is the average Wₙ across all simulations at each generation.
        The **shaded band** is the 95% confidence interval *(the range we'd expect
        the true average to fall in, given random variation in the simulation)*.
        The **green dashed line** is W₀ = Z₀, the starting value.

        🔍 **What to look for:**
        - A **flat or slowly declining** blue line confirms the Martingale property.
        - Any **downward drift** from the green line is the extinction discount
          at work — *(some startups died and dragged the average down, exactly
          as the theory predicts)*.
        - The shaded band should be narrow if n_sim is large — *(more simulations
          = more precise average = tighter band)*.
        """
    )

    mean_W = sim_df.groupby("generation")["W"].mean().reset_index()
    std_W  = sim_df.groupby("generation")["W"].std().reset_index()

    fig_mart = go.Figure()
    fig_mart.add_trace(go.Scatter(
        x=mean_W["generation"], y=mean_W["W"],
        mode="lines+markers",
        name="E[W_n]  (empirical mean)",
        line=dict(color="#58a6ff", width=2.5),
        marker=dict(size=5),
    ))
    fig_mart.add_trace(go.Scatter(
        x=mean_W["generation"],
        y=mean_W["W"] + 1.96 * std_W["W"] / np.sqrt(params_n_sim),
        mode="lines", showlegend=False,
        line=dict(color="#58a6ff", dash="dot", width=1),
    ))
    fig_mart.add_trace(go.Scatter(
        x=mean_W["generation"],
        y=mean_W["W"] - 1.96 * std_W["W"] / np.sqrt(params_n_sim),
        mode="lines", showlegend=False, fill="tonexty",
        fillcolor="rgba(88,166,255,0.12)",
        line=dict(color="#58a6ff", dash="dot", width=1),
    ))
    fig_mart.add_hline(
        y=float(params_z0), line_dash="dash", line_color="#3fb950",
        annotation_text=f"W_0 = Z_0 = {params_z0}",
        annotation_font_color="#3fb950",
    )
    fig_mart.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_title="Generation n", yaxis_title="E[W_n]",
        title="Martingale Property: Mean of W_n Over Time (with 95% CI)",
    )
    st.plotly_chart(fig_mart, width='stretch')

    st.markdown("---")

    # ── W_∞ distribution (surviving paths only) ────────────────────────────────
    st.markdown("#### Final Valuation Distribution — Only the Survivors")
    st.markdown(
        """
        This histogram shows the **spread of final valuations (W∞)** among
        startups that made it to the end without going bankrupt.

        *(Think of this as the prize distribution at a marathon — but only
        showing finishers' times, not those who dropped out. Most finishers
        cluster near the median finish time, but a few elite runners finish
        far earlier — those are the outliers on the right.)*

        The **Kesten-Stigum theorem** (see Theory tab) tells us that surviving
        startups will have a genuinely positive final value — they won't just
        barely survive, they'll have built real momentum. The condition is:
        """
    )
    st.latex(
        r"W_\infty > 0 \;\text{ a.s. on the survival event}"
        r"\quad \Leftrightarrow \quad"
        r"\mathbb{E}[X \log X] < \infty"
    )
    st.markdown(
        """
        *(In plain terms: as long as the referral distribution doesn't have
        an absurdly heavy tail — like one user occasionally referring a billion
        others — surviving startups will end up with a meaningful, positive
        valuation score.)*

        🔍 **What to look for:**
        - A **tall bar near zero** means most survivors barely made it
          *(they survived but didn't really grow)*.
        - A **long right tail** means a few survivors became true unicorns
          *(huge W∞ = massively outperformed expectations)*.
        - The shape of this histogram is what a VC's **portfolio return distribution**
          looks like — most bets return modestly, a few return spectacularly.
        """
    )

    final_gen = sim_df[sim_df["generation"] == params_n_gen]
    survivors = final_gen[final_gen["alive"] == 1]["W"]

    if len(survivors) > 10:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=survivors, nbinsx=60,
            marker_color="#58a6ff", opacity=0.8,
            name="W_∞ | survival",
        ))
        fig_hist.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            xaxis_title="W_∞", yaxis_title="Count",
            title=f"Terminal W_∞ Distribution | Surviving Paths (n={len(survivors)})",
        )
        st.plotly_chart(fig_hist, width='stretch')
    else:
        st.warning("Too few surviving paths to plot terminal distribution.")

    st.markdown("---")

    # ── Valuation calculator ───────────────────────────────────────────────────
    st.markdown("#### 💰 Fair Market Value Calculator")
    st.markdown(
        """
        Now we put a **dollar figure** on the startup using our model.
        The Fair Market Value (FMV) combines three things:

        1. **V₀** — how much you think each seed user is "worth" in dollars
           *(like a price-per-customer baseline — e.g. if each early user
           is worth $10,000 in lifetime revenue)*.
        2. **Wₙ** — the Martingale value at generation n, which tells you if
           this startup is outperforming or underperforming its expected growth
           *(like a performance multiplier — above 1 = beating expectations)*.
        3. **(1 − η^Zₙ)** — a survival discount that shrinks the valuation
           if there is still a meaningful chance the startup goes bankrupt
           *(like a risk haircut a bank applies to a loan with uncertain repayment)*.
        """
    )
    st.latex(
        r"\text{FMV}_n = V_0 \cdot W_n \cdot \underbrace{(1 - \eta^{Z_n})}_{\text{survival adjustment}}"
    )
    st.markdown(
        "Adjust **V₀** (your dollar-per-user assumption) and the **generation** to query, "
        "and the metrics below will update to show the median, optimistic (75th percentile), "
        "and conservative (25th percentile) valuations across all simulated startups at that point."
    )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        V0 = st.number_input(
            "Reference Value per Seed User V₀ (USD)",
            min_value=100, max_value=100_000,
            value=10_000, step=500,
        )
    with col_v2:
        query_gen = st.slider(
            "Query Generation n",
            min_value=1, max_value=params_n_gen,
            value=min(10, params_n_gen),
        )

    W_at_n = sim_df[sim_df["generation"] == query_gen]["W"]
    eta_adj = 1 - (eta if eta is not None else 0.5)
    fmv_median = float(W_at_n.median()) * V0 * eta_adj
    fmv_p75    = float(W_at_n.quantile(0.75)) * V0 * eta_adj
    fmv_p25    = float(W_at_n.quantile(0.25)) * V0 * eta_adj

    c_a, c_b, c_c = st.columns(3)
    c_a.metric("FMV (Median)",   f"${fmv_median:,.0f}")
    c_b.metric("FMV (75th pct)", f"${fmv_p75:,.0f}")
    c_c.metric("FMV (25th pct)", f"${fmv_p25:,.0f}")

# =============================================================================
# TAB 3 — EXTINCTION RISK
# =============================================================================
with tabs[3]:
    st.markdown("## 💀 Extinction Risk & VC Ruin Probability")
    st.markdown("### PGF Fixed-Point Method")
    st.markdown(
        """
        Even if a startup is growing on average (μ > 1), there is still a real
        chance it hits zero users and dies. This section **quantifies exactly
        how large that chance is** using a mathematical tool called the
        Probability Generating Function (PGF).

        *(Think of it this way: a basketball player who scores 20 points on
        average could still score 0 in a particular game if they have a bad
        night. The PGF is a formula that captures the full range of
        possibilities — not just the average — in a single compact expression.)*

        The PGF packs all the probabilities of the referral distribution into
        one formula:
        """
    )
    st.latex(r"\phi(s) = \mathbb{E}[s^X] = \sum_{k=0}^{\infty} p_k s^k, \quad s \in [0,1]")
    st.markdown(
        """
        The extinction probability η is the answer to: *"What is the chance
        this startup eventually dies out completely?"* It satisfies:
        """
    )
    st.latex(r"\eta = \phi(\eta)")
    st.markdown(
        """
        *(In plain English: η is the value that, when plugged into the PGF
        formula, comes back out unchanged — a 'fixed point'. It's like asking:
        what probability of dying leads to itself being reproduced exactly?
        The answer is the true long-run extinction risk.)*

        For a startup with Z₀ independent seed users, since each user's
        lineage is independent:
        """
    )
    st.latex(r"P(\text{extinction}) = \eta^{z_0}")

    st.markdown("---")

    # ── PGF plot with fixed-point annotation ──────────────────────────────────
    st.markdown("#### PGF Fixed-Point Visualisation")
    st.markdown(
        """
        This chart finds the extinction probability **visually** by plotting
        the PGF curve (blue) against the diagonal y = s (green dashes).

        *(Imagine you are looking for the point where a curved slide meets a
        straight ramp. The curved slide is the PGF — it tells you where
        probability 'flows' to. The straight ramp is y = s. Where they cross
        is the fixed point — the extinction probability.)*

        - The crossing at **s = 1** (right edge) is always trivially true
          and not meaningful *(certainty maps to certainty)*.
        - The crossing at **s = η < 1** (the red dotted line) is the real answer —
          the actual probability that a single user's entire family tree eventually
          dies out. *(If η = 0.4, there is a 40% chance that one user's referral
          chain will eventually reach zero.)*
        - For the whole startup launched with Z₀ seed users: extinction chance = η^Z₀.
        """
    )

    _, pgf = make_offspring_sampler(params_dist, params_mu, params_sig2)
    s_vals = np.linspace(0, 0.9999, 500)
    phi_vals = np.array([pgf(s) for s in s_vals])

    fig_pgf = go.Figure()
    fig_pgf.add_trace(go.Scatter(
        x=s_vals, y=phi_vals,
        mode="lines", name="φ(s)  PGF",
        line=dict(color="#58a6ff", width=2.5),
    ))
    fig_pgf.add_trace(go.Scatter(
        x=s_vals, y=s_vals,
        mode="lines", name="y = s (diagonal)",
        line=dict(color="#3fb950", width=1.5, dash="dash"),
    ))

    if eta is not None and eta < 1.0:
        fig_pgf.add_vline(
            x=eta ** (1 / max(params_z0, 1)),
            line_dash="dot", line_color="#f78166",
            annotation_text=f"η_single ≈ {eta**(1/max(params_z0,1)):.3f}",
            annotation_font_color="#f78166",
        )

    fig_pgf.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_title="s", yaxis_title="φ(s)",
        xaxis_range=[0, 1], yaxis_range=[0, 1],
        title="PGF Fixed-Point: φ(s) = s",
    )
    st.plotly_chart(fig_pgf, width='stretch')

    st.markdown("---")

    # ── Extinction probability as a function of μ ─────────────────────────────
    st.markdown("#### How Does Extinction Risk Change as Growth Rate Increases?")
    st.markdown(
        """
        This chart answers: **"If we improve the average referral rate μ,
        how much safer does the startup become?"**

        *(Like asking: if a patient's resting heart rate drops from 90 to 60,
        how much does their cardiac risk decrease? The answer is not linear —
        the first improvements matter most.)*

        🔍 **What to look for:**
        - At **μ < 1** (left of centre): extinction is almost certain — the
          startup is shrinking on average and will almost surely die.
          *(Like a store losing customers every week.)*
        - At **μ = 1**: still certain extinction, just more slowly.
          *(Breaking even on users still leads to eventual closure — random
          bad weeks accumulate.)*
        - At **μ > 1** (right of centre): extinction risk drops — but notice
          **it never reaches zero even with μ = 3**. This is the paradox:
          even tripling your average growth rate leaves meaningful risk.
          *(A restaurant that adds 3× as many tables as it loses still risks
          a bad review going viral and emptying the place overnight.)*
        - The **blue dashed line** marks your current μ setting. The
          **red dotted line** marks the resulting extinction probability.
        """
    )

    mu_range = np.linspace(0.5, 3.0, 60)
    eta_curve = [
        compute_extinction_probability(params_dist, m, params_sig2, params_z0)
        for m in mu_range
    ]

    fig_eta = go.Figure()
    fig_eta.add_trace(go.Scatter(
        x=mu_range, y=eta_curve,
        mode="lines", name=f"η(μ)  [{params_dist}, σ²={params_sig2:.1f}]",
        line=dict(color="#f78166", width=2.5),
    ))
    fig_eta.add_vline(
        x=params_mu, line_dash="dash", line_color="#58a6ff",
        annotation_text=f"Current μ={params_mu:.2f}",
        annotation_font_color="#58a6ff",
    )
    fig_eta.add_hline(
        y=eta if eta is not None else 0,
        line_dash="dot", line_color="#f78166",
        annotation_text=f"Current η={eta:.3f}" if eta else "",
        annotation_font_color="#f78166",
    )
    fig_eta.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_title="μ  (mean referral rate)",
        yaxis_title="P(extinction)",
        yaxis_range=[0, 1.05],
        title="Extinction Probability vs Mean Referral Rate",
    )
    st.plotly_chart(fig_eta, width='stretch')

    st.markdown("---")

    # ── Risk metrics table ─────────────────────────────────────────────────────
    st.markdown("#### Risk Summary Table")
    st.markdown(
        """
        This table compares what the **mathematics predicts** with what the
        **simulation actually produced** — a reality check on our model.

        *(Like comparing a weather forecast to what actually happened. If the
        forecast said 70% chance of rain and it rained 68% of days over a year,
        the forecast was well-calibrated. Here we do the same for startup survival.)*

        A small **Discrepancy** value (row 3) means the simulation closely matches
        theory — our model is well-calibrated. The closer to 0, the better.
        """
    )

    eta_val = eta if eta is not None else 1.0
    survival_rate_emp = (
        sim_df[sim_df["generation"] == params_n_gen]["alive"].mean()
    )

    risk_data = {
        "Metric": [
            "Theoretical Extinction Probability  η^z₀",
            "Empirical Extinction Rate (simulation)",
            "Discrepancy |theory − empirical|",
            "Expected survivors out of 1 000 startups",
            "P(Unicorn: W_∞ > 2·W_0)",
        ],
        "Value": [
            f"{eta_val:.4f}",
            f"{1 - survival_rate_emp:.4f}",
            f"{abs(eta_val - (1 - survival_rate_emp)):.4f}",
            f"{(1 - eta_val) * 1000:.1f}",
            f"{(survivors > 2 * params_z0).mean():.4f}"
              if len(survivors) > 0 else "N/A",
        ],
    }
    st.dataframe(
        pd.DataFrame(risk_data),
        width='stretch',
        hide_index=True,
    )

    # ── Seed-size sensitivity ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Seed-Size Sensitivity — How Many Seed Users Do You Need?")
    st.markdown(
        """
        This chart answers: **"How many seed users (Z₀) does the startup need
        at launch to meaningfully reduce its bankruptcy risk?"**

        *(Think of it like life rafts on a ship. Each extra liferaft reduces the
        probability that everyone drowns if the ship sinks. But the effect is
        multiplicative — the first few rafts save the most lives; adding the
        1,000th raft barely changes the odds.)*

        - Each extra seed user multiplies the extinction probability by η (< 1),
          so the curve falls **exponentially** *(gets safer very fast at first,
          then slower)*.
        - The **blue vertical line** marks your current Z₀ setting.
        - **Practical implication for VCs:** this curve shows you the minimum
          seed round size needed to get extinction risk below a tolerable threshold —
          *(e.g. 'We need at least 20 seed users to get below 5% bankruptcy risk
          at our current growth rate')*.
        """
    )

    if eta is not None:
        eta_single = eta ** (1.0 / max(params_z0, 1))
        z_range = range(1, 51)
        eta_z = [eta_single ** z for z in z_range]

        fig_seed = go.Figure()
        fig_seed.add_trace(go.Scatter(
            x=list(z_range), y=eta_z,
            mode="lines+markers",
            line=dict(color="#e3b341", width=2),
            marker=dict(size=4),
            name="P(extinction | Z₀ = z)",
        ))
        fig_seed.add_vline(
            x=params_z0, line_dash="dash", line_color="#58a6ff",
            annotation_text=f"Current Z₀={params_z0}",
            annotation_font_color="#58a6ff",
        )
        fig_seed.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            xaxis_title="Seed Users Z₀",
            yaxis_title="P(extinction)",
            yaxis_range=[0, 1.05],
            title="Extinction Probability vs Seed Size",
        )
        st.plotly_chart(fig_seed, width='stretch')

# =============================================================================
# TAB 4 — ML CLASSIFIER
# =============================================================================
with tabs[4]:
    st.markdown("## 🤖 Unicorn vs Zombie Classifier")
    st.markdown(
        """
        ### The VC Diligence Problem

        A Venture Capitalist can only observe a startup's **first few weeks
        of data** before deciding whether to invest. Can a machine learning
        model learn to predict which startups will survive — just from that
        early evidence?

        *(Think of a doctor reading an X-ray from a patient's first checkup
        and predicting whether they'll develop a condition 10 years later.
        The doctor uses early biomarkers — we use early growth metrics.)*

        We frame this as a **binary classification** task:
        - **Label 1 = Unicorn:** startup is still alive at generation n
          *(the patient is healthy long-term)*.
        - **Label 0 = Zombie:** startup went extinct before generation n
          *(the patient developed the condition)*.

        The model only sees data from **the first 5 generations** — mirroring
        what a VC sees at the Seed round stage before committing capital.
        *(We deliberately hide future data to make this a fair test of
        early prediction, not cheating by looking at the answer sheet.)*
        """
    )

    st.markdown("---")

    ml_df = st.session_state["ml_dataset"]
    if ml_df is None:
        st.error("Run the simulation first.")
        st.stop()

    # ── Dataset overview ───────────────────────────────────────────────────────
    st.markdown("#### Feature Dataset Preview")
    st.markdown(
        """
        Each row is one simulated startup. The columns are the **early-stage
        features** extracted from generations 0–5, and the **label** is
        whether it survived to generation n.

        | Feature | What it measures | Plain English |
        |---|---|---|
        | **w_mean_early** | Average W_n in first 5 gens | Was the normalised valuation healthy early on? |
        | **w_std_early** | Variability of W_n | Was growth consistent or erratic? |
        | **w_slope_early** | Trend direction of W_n | Was valuation improving or declining? |
        | **z_max_early** | Peak user count in 5 gens | Did the startup ever get traction? |
        | **z_at_gen5** | User count at generation 5 | Where did they stand at week 5? |
        | **extinct_before5** | Did it die in the first 5 gens? | Already bankrupt = automatic zombie |
        | **log_z0** | Log of starting seed users | Bigger launch = safer start? |
        | **label** | 1 = alive at gen n, 0 = dead | The answer the model must predict |
        """
    )

    col_lb = st.columns(3)
    col_lb[0].metric("Total samples", len(ml_df))
    col_lb[1].metric("Unicorns (label=1)", int(ml_df["label"].sum()))
    col_lb[2].metric("Zombies  (label=0)", int((ml_df["label"] == 0).sum()))

    st.markdown("---")

    # ── Feature correlation heatmap ────────────────────────────────────────────
    st.markdown(
        "#### Feature Correlation Matrix\n\n"
        "High correlation with `label` indicates predictive power.  "
        "Note how `extinct_before5` and `z_at_gen5` are strong "
        "early warning signals."
    )

    feature_cols = [c for c in ml_df.columns if c not in ["sim_id"]]
    corr = ml_df[feature_cols].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    fig_corr.patch.set_facecolor("#0d1117")
    ax_corr.set_facecolor("#161b22")
    sns.heatmap(
        corr, ax=ax_corr,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.3, linecolor="#30363d",
        square=True,
    )
    ax_corr.set_title("Feature Correlation Matrix", color="#e6edf3", fontsize=11)
    ax_corr.tick_params(colors="#8b949e")
    st.pyplot(fig_corr, width='stretch')
    plt.close(fig_corr)

    st.markdown("---")

    # ── Model selection & hyperparameters ─────────────────────────────────────
    st.markdown("#### Model Configuration")
    st.markdown(
        "Select a classifier and tune its hyperparameters.  "
        "We wrap it in a `sklearn.pipeline.Pipeline` with "
        "`StandardScaler` for robustness."
    )

    col_m1, col_m2 = st.columns([1, 2])

    with col_m1:
        model_choice = st.selectbox(
            "Classifier",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        )

    with col_m2:
        if model_choice == "Random Forest":
            n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
            max_depth    = st.slider("max_depth", 2, 20, 6)
            clf_base = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=42, n_jobs=-1,
            )
        elif model_choice == "Gradient Boosting":
            n_estimators = st.slider("n_estimators", 50, 300, 100, step=50)
            lr           = st.select_slider(
                "learning_rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
            clf_base = GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=lr,
                random_state=42,
            )
        else:
            C_val = st.select_slider("C (regularisation)", [0.01, 0.1, 1, 10, 100], value=1)
            clf_base = LogisticRegression(C=C_val, max_iter=1000, random_state=42)

    test_size  = st.slider("Test Set Fraction", 0.10, 0.40, 0.20, step=0.05)

    train_btn = st.button("🏋️ Train Model", type="primary", width='stretch')

    if train_btn:
        feat_cols = [c for c in ml_df.columns
                     if c not in ["sim_id", "label"]]
        X = ml_df[feat_cols].fillna(0).values
        y = ml_df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf_base),
        ])

        with st.spinner("Training classifier…"):
            pipeline.fit(X_train, y_train)

        y_prob   = pipeline.predict_proba(X_test)[:, 1]
        y_pred   = pipeline.predict(X_test)
        auc      = roc_auc_score(y_test, y_prob)
        ap       = average_precision_score(y_test, y_prob)
        cv_auc   = cross_val_score(
            pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

        st.session_state["clf_pipeline"] = pipeline
        st.session_state["model_trained"] = True
        st.session_state["clf_metrics"] = {
            "auc": auc, "ap": ap, "cv_auc": cv_auc,
            "y_test": y_test, "y_prob": y_prob, "y_pred": y_pred,
            "feat_cols": feat_cols,
            "model_name": model_choice,
        }
        st.success(f"Model trained!  AUC = {auc:.3f}", icon="✅")

    if st.session_state["model_trained"]:
        st.markdown("---")
        m = st.session_state["clf_metrics"]

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC",        f"{m['auc']:.3f}")
        c2.metric("Avg Precision",  f"{m['ap']:.3f}")
        c3.metric("5-Fold CV AUC",  f"{m['cv_auc'].mean():.3f} ± {m['cv_auc'].std():.3f}")
        c4.metric("Model",          m["model_name"])

        # ── ROC curve ─────────────────────────────────────────────────────────
        st.markdown("#### ROC Curve")
        st.markdown(
            "The ROC curve plots the trade-off between True Positive Rate "
            "(sensitivity) and False Positive Rate (1-specificity) across all "
            "decision thresholds.  **AUC = 1.0** is perfect; **AUC = 0.5** is "
            "random guessing."
        )

        fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines", name=f"ROC (AUC={m['auc']:.3f})",
            line=dict(color="#58a6ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
        ))
        fig_roc.add_shape(
            type="line", x0=0, x1=1, y0=0, y1=1,
            line=dict(color="#30363d", dash="dash"),
        )
        fig_roc.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis_range=[0, 1], yaxis_range=[0, 1.02],
            title="Receiver Operating Characteristic",
        )
        st.plotly_chart(fig_roc, width='stretch')

        # ── Confusion matrix ───────────────────────────────────────────────────
        st.markdown("#### Confusion Matrix")

        cm = confusion_matrix(m["y_test"], m["y_pred"])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("#0d1117")
        ax_cm.set_facecolor("#161b22")
        sns.heatmap(
            cm, ax=ax_cm,
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=["Zombie", "Unicorn"],
            yticklabels=["Zombie", "Unicorn"],
            linewidths=0.5, linecolor="#30363d",
        )
        ax_cm.set_title("Confusion Matrix", color="#e6edf3", fontsize=11)
        ax_cm.set_xlabel("Predicted", color="#8b949e")
        ax_cm.set_ylabel("Actual", color="#8b949e")
        ax_cm.tick_params(colors="#8b949e")
        st.pyplot(fig_cm, width='stretch')
        plt.close(fig_cm)

        # ── Classification report ──────────────────────────────────────────────
        st.markdown("#### Classification Report")
        report_str = classification_report(
            m["y_test"], m["y_pred"],
            target_names=["Zombie", "Unicorn"],
        )
        st.code(report_str, language="text")

        # ── Feature importances ───────────────────────────────────────────────
        if hasattr(st.session_state["clf_pipeline"]["clf"], "feature_importances_"):
            st.markdown("#### Feature Importances")
            st.markdown(
                "Which early-stage KPIs are most predictive of long-run survival?  "
                "The bar chart below answers this question for tree-based models."
            )
            importances = (
                st.session_state["clf_pipeline"]["clf"].feature_importances_
            )
            fi_df = pd.DataFrame({
                "Feature":    m["feat_cols"],
                "Importance": importances,
            }).sort_values("Importance", ascending=False)

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                title="Feature Importances",
            )
            fig_fi.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font_color="#e6edf3", showlegend=False,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_fi, width='stretch')

        # ── Score distribution ─────────────────────────────────────────────────
        st.markdown("#### Predicted Probability Distribution")
        st.markdown(
            "Ideally, the model pushes Unicorn scores toward 1 and "
            "Zombie scores toward 0 — a **bimodal** distribution indicates "
            "good separation."
        )

        fig_score = go.Figure()
        for lab, name, col in [(0, "Zombie", "#f78166"), (1, "Unicorn", "#58a6ff")]:
            mask = m["y_test"] == lab
            fig_score.add_trace(go.Histogram(
                x=m["y_prob"][mask], nbinsx=30,
                name=name, opacity=0.75,
                marker_color=col,
            ))
        fig_score.update_layout(
            barmode="overlay",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            xaxis_title="Predicted P(Unicorn)",
            yaxis_title="Count",
            title="Score Distribution by True Class",
        )
        st.plotly_chart(fig_score, width='stretch')

# =============================================================================
# TAB 5 — THEORY REFERENCE
# =============================================================================
with tabs[5]:
    st.markdown("## 📖 Theory Reference & Formulary")

    # ── Section 1: Galton-Watson Process ──────────────────────────────────────
    st.markdown("### 1. Galton-Watson Branching Process")
    st.markdown(
        "A **Galton-Watson Branching Process** models a population where each "
        "individual independently produces a random number of offspring. "
        "In our startup context, each user recruits a random number of new users "
        "per generation cycle. The process is defined recursively:"
    )
    st.latex(r"Z_0 = z_0, \qquad Z_{n+1} = \sum_{i=1}^{Z_n} X_i^{(n)}")
    st.markdown(
        "where the $X_i^{(n)}$ are i.i.d. draws from the offspring distribution. "
        "The key parameters are:"
    )
    st.markdown(
        "| Symbol | Definition |\n"
        "|---|---|\n"
        "| Z₀ | Initial seed users |\n"
        "| Zₙ | User count at generation n |\n"
        "| μ = E[X] | Mean referrals per user |\n"
        "| σ² = Var(X) | Referral variance |\n"
        "| φ(s) = E[sˣ] | Probability Generating Function |"
    )

    st.markdown("---")

    # ── Section 2: Martingale Theory ──────────────────────────────────────────
    st.markdown("### 2. Martingale Theory")
    st.markdown(
        "**Definition:** A sequence {Mₙ, Fₙ} is a *martingale* if:\n"
        "1. Mₙ is Fₙ-measurable (adapted to the filtration).\n"
        "2. E[|Mₙ|] < ∞ for all n (finite expectation).\n"
        "3. E[Mₙ₊₁ | Fₙ] = Mₙ a.s. (the 'fair game' property)."
    )
    st.markdown("**The Normalised Valuation Process** is defined as:")
    st.latex(r"W_n = \frac{Z_n}{\mu^n}, \qquad \mu > 0")
    st.markdown(
        "**Theorem:** {Wₙ, Fₙ} is a non-negative martingale.\n\n"
        "**Proof:** Condition on the filtration Fₙ (equivalently, on Zₙ):"
    )
    st.latex(
        r"\mathbb{E}[W_{n+1} \mid \mathcal{F}_n]"
        r"= \frac{1}{\mu^{n+1}} \mathbb{E}\!\left[\sum_{i=1}^{Z_n} X_i \;\Big|\; Z_n\right]"
        r"= \frac{Z_n \cdot \mu}{\mu^{n+1}}"
        r"= \frac{Z_n}{\mu^n} = W_n \qquad \square"
    )
    st.markdown(
        "**Optional Stopping Theorem:** For any bounded stopping time τ "
        "(e.g., the first time Zₙ exceeds a funding threshold):"
    )
    st.latex(r"\mathbb{E}[W_\tau] = \mathbb{E}[W_0] = z_0")
    st.markdown(
        "This means a VC pricing equity at Wₙ · V₀ makes an *unbiased* bet — "
        "the Martingale property is the mathematical statement of 'no free lunch'."
    )

    st.markdown("---")

    # ── Section 3: Extinction Probability ────────────────────────────────────
    st.markdown("### 3. Extinction Probability")
    st.markdown(
        "The extinction probability η satisfies the **PGF fixed-point equation:**"
    )
    st.latex(r"\eta = \phi(\eta)")
    st.markdown(
        "η is the *smallest non-negative* root of φ(s) = s. "
        "The regime determines whether extinction is certain:\n\n"
        "| Regime | Condition | Extinction Probability |\n"
        "|---|---|---|\n"
        "| Sub-critical | μ < 1 | η = 1 (certain) |\n"
        "| Critical | μ = 1 | η = 1 (certain, slower) |\n"
        "| Super-critical | μ > 1 | η < 1 (positive survival) |"
    )
    st.markdown("For Z₀ = z₀ independent seed users, lineages are independent so:")
    st.latex(r"P(\text{extinction}) = \eta^{z_0}")

    st.markdown("---")

    # ── Section 4: Offspring Distributions ───────────────────────────────────
    st.markdown("### 4. Offspring Distributions")

    st.markdown("**Poisson(λ = μ)** — organic word-of-mouth; variance equals mean:")
    st.latex(
        r"p_k = e^{-\mu}\frac{\mu^k}{k!}, \qquad"
        r"\phi(s) = e^{\mu(s-1)}, \qquad"
        r"\sigma^2 = \mu"
    )

    st.markdown(
        "**Geometric(p = 1/(1+μ))** — heavy right tail; "
        "rare influencers drive most growth:"
    )
    st.latex(
        r"p_k = p(1-p)^k, \qquad"
        r"\phi(s) = \frac{p}{1-(1-p)s}, \qquad"
        r"\sigma^2 = \frac{1-p}{p^2}"
    )

    st.markdown(
        "**Negative-Binomial(r, p)** — over-dispersed Poisson; "
        "best fit for cohort-clustered fintech referrals. "
        "Match moments μ, σ² via:"
    )
    st.latex(
        r"r = \frac{\mu^2}{\sigma^2 - \mu}, \qquad"
        r"p = \frac{\mu}{\sigma^2}, \qquad"
        r"\phi(s) = \left(\frac{p}{1-(1-p)s}\right)^r"
    )

    st.markdown("---")

    # ── Section 5: Fair Market Valuation ─────────────────────────────────────
    st.markdown("### 5. Fair Market Valuation")
    st.markdown(
        "Combining the Martingale value with an extinction-risk discount "
        "yields the **Fair Market Value** of the startup at generation n:"
    )
    st.latex(r"\text{FMV}_n = V_0 \cdot W_n \cdot \left(1 - \eta^{Z_n}\right)")
    st.markdown(
        "where:\n"
        "- **V₀** = reference dollar value assigned per unit of W₀ = Z₀\n"
        "- **(1 − η^Zₙ)** = survival-adjusted discount (collapses to 0 at extinction)\n"
        "- **Wₙ** = Martingale value; fundamental strength stripped of exponential noise"
    )

    st.markdown("---")

    # ── Section 6: Kesten-Stigum Theorem ─────────────────────────────────────
    st.markdown("### 6. Kesten-Stigum Theorem")
    st.markdown(
        "The Martingale Convergence Theorem guarantees Wₙ → W∞ almost surely. "
        "The Kesten-Stigum theorem characterises when the limit is non-trivial: "
        "if the **log-moment condition** holds,"
    )
    st.latex(r"\mathbb{E}[X \log X] < \infty")
    st.markdown("then:")
    st.latex(
        r"W_\infty = \lim_{n \to \infty} W_n \quad \text{exists a.s.}, \qquad"
        r"W_\infty > 0 \;\text{ a.s. on the survival event}"
    )
    st.markdown(
        "When E[X log X] = ∞ (extremely heavy tails), W∞ = 0 almost surely even "
        "on surviving paths — the **Schröder case**. This represents startups whose "
        "growth is so fat-tailed that the normalised valuation collapses despite "
        "technical survival."
    )

    st.markdown("---")

    # ── Section 7: References ─────────────────────────────────────────────────
    st.markdown("### 7. References")
    st.markdown(
        "- Athreya, K.B. & Ney, P.E. (1972). *Branching Processes*. Springer.\n"
        "- Williams, D. (1991). *Probability with Martingales*. Cambridge UP.\n"
        "- Karatzas, I. & Shreve, S. (1991). *Brownian Motion and Stochastic Calculus*. Springer.\n"
        "- Durrett, R. (2019). *Probability: Theory and Examples*, 5th ed. Cambridge UP.\n"
        "- Kesten, H. & Stigum, B.P. (1966). A limit theorem for multidimensional "
        "Galton-Watson processes. *Ann. Math. Statist.* 37(5), 1211–1223."
    )

    # ── Interactive PGF explorer ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Interactive PGF Explorer")
    st.markdown(
        "Explore how the PGF shape changes with distribution parameters. "
        "The fixed-point intersection shows the extinction probability."
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        exp_dist = st.selectbox("Distribution", ["Poisson", "Geometric", "NegBinomial"],
                                key="explorer_dist")
    with col_e2:
        exp_mu   = st.slider("μ", 0.5, 3.0, 1.5, 0.1, key="explorer_mu")
    with col_e3:
        exp_sig2 = st.slider("σ²", 0.1, 5.0, 1.0, 0.1, key="explorer_sig2")

    _, exp_pgf = make_offspring_sampler(exp_dist, exp_mu, exp_sig2)
    s_e = np.linspace(0, 0.9999, 400)
    phi_e = np.array([exp_pgf(s) for s in s_e])

    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(
        x=s_e, y=phi_e, mode="lines",
        name="φ(s)", line=dict(color="#e3b341", width=2),
    ))
    fig_exp.add_trace(go.Scatter(
        x=s_e, y=s_e, mode="lines",
        name="y=s", line=dict(color="#8b949e", dash="dash", width=1.5),
    ))
    exp_eta = compute_extinction_probability(exp_dist, exp_mu, exp_sig2, 1)
    if exp_eta < 1.0:
        fig_exp.add_vline(
            x=exp_eta, line_dash="dot", line_color="#f78166",
            annotation_text=f"η={exp_eta:.3f}",
            annotation_font_color="#f78166",
        )
    fig_exp.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_range=[0, 1], yaxis_range=[0, 1],
        xaxis_title="s", yaxis_title="φ(s)",
        title=f"PGF Explorer — {exp_dist}(μ={exp_mu}, σ²={exp_sig2})",
        height=380,
    )
    st.plotly_chart(fig_exp, width='stretch')

# =============================================================================
# 6.  FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:0.82rem;'>"
    "The Valuation-Survival Paradox · Galton-Watson Branching Processes · "
    "Martingale Theory · VC Risk Analytics<br>"
    "Built with Streamlit · NumPy · SciPy · Plotly · Seaborn · Scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)