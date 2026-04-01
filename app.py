import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import random

st.set_page_config(layout="wide")

# =========================
# STYLE (CENTER + GREEN BUTTON)
# =========================
st.markdown("""
<style>
div.stButton > button {
    background-color: #2ecc71;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    height: 3em;
}
div.stButton > button:hover {
    background-color: #27ae60;
}

/* Center everything nicely */
.block-container {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("IPA Analysis (M/M/1) — Textbook Formulation")
st.caption("Author: Ana Theodora Balaci")

# =========================
# TOP CONTROL BAR (CENTERED)
# =========================
_, col1, col2, col3, col4, _ = st.columns([1,1,1,1,1,1])

with col1:
    lam = st.number_input("λ", value=1.0, min_value=0.01)

with col2:
    mu = st.number_input("μ", value=1.2, min_value=0.01)

with col3:
    N = st.number_input("N", value=50000, min_value=1000)

with col4:
    seed = st.number_input("Seed", value=1)

# CENTERED RUN BUTTON
_, col_btn, _ = st.columns([2,1,2])
with col_btn:
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

    IPA_mu_vals = []
    IPA_lam_vals = []

    prop_mu = 0
    prop_lam = 0

    for i in range(N):

        inter_arrival = exp_rv(lam)
        t += inter_arrival

        service = exp_rv(mu)

        start_service = max(t, prev_departure)
        departure = start_service + service

        W = departure - t
        W_vals.append(W)

        # IPA (textbook form)
        gen_mu = -service / mu
        gen_lam = inter_arrival / lam

        if t > prev_departure:
            prop_mu = 0
            prop_lam = 0

        total_mu = gen_mu + prop_mu
        total_lam = gen_lam + prop_lam

        prop_mu = total_mu
        prop_lam = total_lam

        IPA_mu_vals.append(total_mu)
        IPA_lam_vals.append(total_lam)

        prev_departure = departure

    W_avg = np.cumsum(W_vals) / np.arange(1, N + 1)
    dW_dmu = np.cumsum(IPA_mu_vals) / np.arange(1, N + 1)
    dW_dlam = np.cumsum(IPA_lam_vals) / np.arange(1, N + 1)

    return W_avg, dW_dlam, dW_dmu


# =========================
# RUN (AUTO LOAD + BUTTON)
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
        # CENTERED METRICS
        # =========================
        _, m1, m2, m3, _ = st.columns([1,1,1,1,1])

        m1.metric("W", f"{W[-1]:.4f}", f"{W_th:.4f}")
        m2.metric("dW/dλ", f"{dW_lam[-1]:.4f}", f"{dW_lam_th:.4f}")
        m3.metric("dW/dμ", f"{dW_mu[-1]:.4f}", f"{dW_mu_th:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # =========================
        # CENTERED PLOTS
        # =========================
        _, colA, colB, colC, _ = st.columns([0.5,1,1,1,0.5])

        # W
        fig1 = plt.figure(figsize=(4,3))
        plt.plot(W, linewidth=1)
        plt.axhline(W_th, linestyle="--")
        plt.title("W")
        plt.tight_layout()
        colA.pyplot(fig1)

        # dW/dλ
        fig2 = plt.figure(figsize=(4,3))
        plt.plot(dW_lam, linewidth=1)
        plt.axhline(dW_lam_th, linestyle="--")
        plt.title("dW/dλ")
        plt.tight_layout()
        colB.pyplot(fig2)

        # dW/dμ
        fig3 = plt.figure(figsize=(4,3))
        plt.plot(dW_mu, linewidth=1)
        plt.axhline(dW_mu_th, linestyle="--")
        plt.title("dW/dμ")
        plt.tight_layout()
        colC.pyplot(fig3)

        # =========================
        # FOOTER
        # =========================
        st.caption(
            "IPA implemented using generation + propagation terms with reset at idle periods (textbook form). "
            "All estimators converge to theoretical values."
        )