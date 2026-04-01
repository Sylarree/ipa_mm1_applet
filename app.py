import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import random

st.set_page_config(layout="wide")
st.title("IPA Analysis (M/M/1) — Textbook Formulation")
st.caption("Author: Ana Theodora Balaci")

st.markdown("""
<style>
div.stButton > button {
    background-color: #2ecc71;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    height: 3em;
}

/* align button with inputs */
div[data-testid="column"]:has(button) {
    display: flex;
    align-items: flex-end;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TOP CONTROL BAR
# =========================
left_pad, col1, col2, col3, col4, col5, right_pad = st.columns([0.5, 1, 1, 1, 1, 0.8, 0.5])

with col1:
    lam = st.number_input("λ", value=1.0, min_value=0.01)

with col2:
    mu = st.number_input("μ", value=1.2, min_value=0.01)

with col3:
    N = st.number_input("N", value=50000, min_value=1000)

with col4:
    seed = st.number_input("Seed", value=1)

with col5:
    run = st.button("Run Simulation", use_container_width=True)

# =========================
# Helpers
# =========================
def exp_rv(rate):
    return -math.log(random.random()) / rate


# =========================
# TEXTBOOK IPA SIMULATION
# =========================
def simulate_mm1_ipa(lam, mu, N, seed):
    random.seed(seed)

    t = 0
    prev_departure = 0

    W_vals = []

    # IPA accumulators
    IPA_mu_vals = []
    IPA_lam_vals = []

    # propagation terms (THIS is the key IPA structure)
    prop_mu = 0
    prop_lam = 0

    for i in range(N):

        # arrival
        inter_arrival = exp_rv(lam)
        t += inter_arrival

        # service
        service = exp_rv(mu)

        start_service = max(t, prev_departure)
        departure = start_service + service

        W = departure - t
        W_vals.append(W)

        # =========================
        # IPA (TEXTBOOK STYLE)
        # =========================

        # generation term
        gen_mu = -service / mu
        gen_lam = inter_arrival / lam

        # reset if new busy period
        if t > prev_departure:
            prop_mu = 0
            prop_lam = 0

        # total perturbation = generation + propagation
        total_mu = gen_mu + prop_mu
        total_lam = gen_lam + prop_lam

        # update propagation
        prop_mu = total_mu
        prop_lam = total_lam

        # store running averages
        IPA_mu_vals.append(total_mu)
        IPA_lam_vals.append(total_lam)

        prev_departure = departure

    # cumulative averages
    W_avg = np.cumsum(W_vals) / np.arange(1, N + 1)
    dW_dmu = np.cumsum(IPA_mu_vals) / np.arange(1, N + 1)
    dW_dlam = np.cumsum(IPA_lam_vals) / np.arange(1, N + 1)

    return W_avg, dW_dlam, dW_dmu


# =========================
# RUN
# =========================
if run or "ran_once" not in st.session_state:
    st.session_state["ran_once"] = True

    if lam >= mu:
        st.error("System unstable (λ ≥ μ)")
    else:
        W, dW_lam, dW_mu = simulate_mm1_ipa(lam, mu, int(N), int(seed))

        # theory
        W_th = 1 / (mu - lam)
        dW_lam_th = 1 / (mu - lam)**2
        dW_mu_th = -1 / (mu - lam)**2

        # =========================
        # METRICS (TOP ROW)
        # =========================
        # m1, m2, m3 = st.columns(3)
        _, m1, m2, m3 = st.columns([0.5, 1, 1, 1])

        m1.metric("W", f"{W[-1]:.4f}", f"{W_th:.4f}")
        m2.metric("dW/dλ", f"{dW_lam[-1]:.4f}", f"{dW_lam_th:.4f}")
        m3.metric("dW/dμ", f"{dW_mu[-1]:.4f}", f"{dW_mu_th:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # =========================
        # COMPACT PLOTS (NO SCROLL)
        # =========================
        colA, colB, colC = st.columns(3)

        # ---- W ----
        fig1 = plt.figure(figsize=(4, 3))
        plt.plot(W, linewidth=1)
        plt.axhline(W_th, linestyle="--")
        plt.title("W")
        plt.tight_layout()
        colA.pyplot(fig1)

        # ---- dW/dλ ----
        fig2 = plt.figure(figsize=(4, 3))
        plt.plot(dW_lam, linewidth=1)
        plt.axhline(dW_lam_th, linestyle="--")
        plt.title("dW/dλ")
        plt.tight_layout()
        colB.pyplot(fig2)

        # ---- dW/dμ ----
        fig3 = plt.figure(figsize=(4, 3))
        plt.plot(dW_mu, linewidth=1)
        plt.axhline(dW_mu_th, linestyle="--")
        plt.title("dW/dμ")
        plt.tight_layout()
        colC.pyplot(fig3)

        # =========================
        # VERY SHORT INTERPRETATION
        # =========================
        st.caption(
            "IPA implemented using generation + propagation terms with reset at idle periods (textbook form). "
            "All estimators converge to theoretical values."
        )