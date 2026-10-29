"""
Home Page - Comprehensive Project Overview
"""

import streamlit as st

def show():
    """Display comprehensive home page"""

    # Project Overview
    st.markdown("""
    <div class='info-box'>
        <h2 style='margin-top: 0; color: #E8871E;'>🎯 What is This Project?</h2>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
            This is an <b>Integrated Machine Learning Framework</b> for real-time water quality monitoring, combining
            ensemble classification, BiLSTM-Attention forecasting, and multi-detector anomaly detection into one
            fused early-warning pipeline.
        </p>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
            <b>The Problem:</b> Traditional water testing relies on periodic grab-sampling, which can leave a
            distribution network unmonitored for days between laboratory results.
        </p>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
            <b>Our Approach:</b> Three independent ML modules continuously screen water quality (&lt;100ms per sample)
            and fuse their outputs into a single, graduated 4-level alert — offering continuous automated screening
            as a complement to, not a replacement for, laboratory confirmation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # System Capabilities
    st.markdown("## 🚀 What This System Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='alert-normal'>
            <h3>✅ Instant Classification</h3>
            <p><b>Tells you NOW if water is safe</b></p>
            <p>→ Upload 9 measurements</p>
            <p>→ Get result in &lt;100ms</p>
            <p>→ Potable or Not Potable</p>
            <p><b>Ensemble: 61.75% acc · 66.62% AUC</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='alert-low'>
            <h3>📈 Next-Day Prediction</h3>
            <p><b>Predicts tomorrow's pH</b></p>
            <p>→ Forecasts 1 day ahead</p>
            <p>→ From a 30-day lookback window</p>
            <p>→ Early trend warning</p>
            <p><b>R² Score: 0.5251</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='alert-medium'>
            <h3>⚠️ Pattern Detection</h3>
            <p><b>Finds unusual samples</b></p>
            <p>→ 4 AI detectors working</p>
            <p>→ Quorum vote (≥2-of-4)</p>
            <p>→ Flags unusual patterns</p>
            <p><b>Flag rate: 4.21%</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Clear explanation of each component
    st.markdown("## 🔍 Understanding The 3 Core Components")

    st.markdown("""
    <div class='explain-box'>
        <h3>💡 IMPORTANT: What's the Difference?</h3>
        <p>Many people ask: "Why do we need Classification AND Anomaly Detection? Aren't they the same?"</p>
        <p><b>No — they answer different questions. Here's the clear difference:</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Component 1: Classification
    st.markdown("### 1️⃣ Water Classification — \"Is This Water Safe RIGHT NOW?\"")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **What it does:**
        - Takes a single water sample with 9 measurements (pH, hardness, solids, etc.)
        - Runs it through a weighted soft-voting ensemble (Random Forest, XGBoost, SVM)
        - Answers: **"Can you drink this water? YES or NO"**

        **Real-world example:**
        - You collect water from a tap
        - You measure: pH=7.2, Hardness=200mg/L, Solids=20000ppm, etc.
        - System says: **"✅ POTABLE"** or **"❌ NOT POTABLE"**, with a confidence score

        **When to use:**
        - Testing water from a new source
        - Checking if current water is safe
        - Quality control in water treatment plants
        - Verifying after treatment
        """)

    with col2:
        st.success("""
        **Output:**
        - POTABLE or NOT POTABLE
        - Confidence level
        - Alert level (NORMAL/LOW/MEDIUM/CRITICAL)

        **Speed:** <100ms

        **Ensemble Accuracy:** 61.75%
        **Ensemble ROC-AUC:** 66.62%
        """)

    st.markdown("""
    <div class='honest-box'>
        <b>Honest note:</b> 61.75% accuracy is not a high number in absolute terms — but it is the true,
        leakage-free result, verified by independently re-running the pipeline. It is consistent with what
        comparable published work reports on this same dataset (a genuinely hard problem given only 9
        physicochemical features). Recall matters more than raw accuracy here: missing a truly non-potable
        sample (false negative) is far more costly than a false alarm.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Component 2: Forecasting
    st.markdown("### 2️⃣ pH Forecasting — \"What Will Happen Tomorrow?\"")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **What it does:**
        - Takes the last 30 days of pH measurements at a station
        - Uses a BiLSTM with an additive-attention-style temporal pooling layer to learn trends
        - Predicts: **"What will pH be tomorrow (1 day ahead)?"**

        **Real-world example:**
        - Your pH has been: 7.0 → 7.1 → 6.9 → 6.8 → 6.7 (declining trend)
        - System predicts: **"Tomorrow's pH is trending toward 6.4"**
        - You get an **early trend signal** to plan a treatment adjustment

        **When to use:**
        - Continuous water quality monitoring
        - Spotting whether a current trend is worsening
        - Planning treatment adjustments in advance

        *Scope note: this module predicts 1 day ahead, not a 24–48 hour multi-step horizon — extending the
        horizon further is listed as future work, not something currently implemented.*
        """)

    with col2:
        st.info("""
        **Output:**
        - Next-day pH value
        - Trend direction
        - Alert if trending out of safe range

        **Horizon:** 1 day ahead
        **Lookback:** 30 days

        **BiLSTM R²:** 0.5251
        **BiLSTM RMSE:** 0.0524

        *(GRU baseline: R² 0.3572 — BiLSTM improves RMSE by 14.0% and R² by 47.0%)*
        """)

    st.markdown("---")

    # Component 3: Anomaly Detection
    st.markdown("### 3️⃣ Anomaly Detection — \"Is This Sample UNUSUAL or STRANGE?\"")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **What it does:**
        - Compares each water sample against the pattern of ~2,847 historical samples
        - Uses 4 structurally different AI detectors to find unusual combinations
        - Answers: **"Is this sample weird/different, regardless of its potability label?"**

        **Real-world example:**
        - Normal samples cluster around typical pH/hardness/conductivity combinations
        - You get a sample: pH=7.0 (normal alone), but Hardness=500 + Conductivity=1200 (both very high)
        - Even though pH looks fine, the **combination** is unusual
        - System flags: **"⚠️ ANOMALY — quorum of detectors agree, investigate"**

        **When to use:**
        - Finding contamination events (chemical spills, industrial discharge)
        - Detecting sensor malfunctions
        - Catching multivariate problems classification might miss
        """)

    with col2:
        st.warning("""
        **Output:**
        - Normal or Anomaly
        - Number of detectors agreeing (0–4 votes)

        **Rule:** ≥2-of-4 detectors must agree (quorum voting)

        **Methods:** Isolation Forest, LOF, One-Class SVM, Autoencoder

        **Flag Rate:** 4.21% (120 of 2,847)
        """)

    st.markdown("""
    <div class='honest-box'>
        <b>Honest note:</b> there is no ground-truth contamination labeling for this dataset, so the 4.21% flag
        rate is a <i>consistency check</i> against an assumed 5% contamination prior — it is <b>not</b> a validated
        precision, recall, or false-alarm rate. Any number claiming a specific "precision" or "recall" for this
        module elsewhere would be unverified and should not be trusted as more than a demo placeholder.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Differences
    st.markdown("## 🎯 Classification vs Forecasting vs Anomaly Detection")

    st.markdown("""
    | Aspect | Classification | Forecasting | Anomaly Detection |
    |--------|---------------|-------------|-------------------|
    | **Question** | Is water safe NOW? | What will pH be tomorrow? | Is this sample UNUSUAL? |
    | **Dataset** | Dataset 1 (Water Potability) | Dataset 2 (Georgia USGS) | Dataset 1 (Water Potability) |
    | **Input** | Single sample (9 params) | Time series (30 days × 11 features, 37 stations) | Single sample (9 params) |
    | **Output** | POTABLE / NOT POTABLE | Next-day pH value | NORMAL / ANOMALY |
    | **Time** | Present moment | 1 day ahead | Present moment |
    | **Purpose** | Immediate safety check | Near-term trend warning | Pattern/contamination detection |
    | **Model** | RF + XGBoost + SVM (soft-voting) | BiLSTM-Attention | 4 detectors, quorum vote |
    | **Verified metric** | 61.75% acc / 66.62% AUC | R² = 0.5251 | 4.21% flag rate *(no P/R — see note above)* |
    """)

    st.markdown("---")

    # Data Sources
    st.markdown("## 📦 Data Sources & Preparation")

    st.markdown("""
    <div class='explain-box'>
        <p><b>Important:</b> This system uses TWO different real datasets from different sources.</p>
        <p><b>Dataset 1 (Kaggle):</b> Individual water samples → Classification & Anomaly Detection</p>
        <p><b>Dataset 2 (USGS Georgia):</b> Spatio-temporal pH data → Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Dataset 1: Water Potability
        **Source:** Kaggle Water Potability Dataset

        **Used for:** Classification & Anomaly Detection

        **Original Data:**
        - **Size:** 3,276 samples
        - **Features:** 9 physicochemical parameters
        - **Target:** Potable (1) or Not Potable (0)
        - **Balance:** Imbalanced (~61% Not Potable, ~39% Potable)

        **After Preprocessing:**
        - **Size:** 2,847 samples (429 outliers removed via IQR)
        - **Split:** 2,277 train / **570 test** (80/20)
        - **Balancing:** SMOTE applied **only to the training set**
          (860 → 1,417 minority samples, 2,834 total training rows) —
          the 570-sample **test set is never touched by SMOTE** and keeps
          the real ~62/38 class split
        - **Normalization:** StandardScaler (fit on training data only)

        **Why this dataset?**
        - Widely-used public benchmark for potability classification
        - Comparable to several cited related works
        """)

    with col2:
        st.markdown("""
        ### 📈 Dataset 2: Water Quality Prediction (Georgia, USA)
        **Source:** United States Geological Survey (USGS)

        **Citation:** Liang Zhao, Olga Gkountouna, and Dieter Pfoser. 2019.
        "Spatial Auto-regressive Dependency Interpretable Learning Based on
        Spatial Topological Constraints." ACM Trans. Spatial Algorithms Syst.
        5, 3, Article 19. DOI: 10.1145/3339823

        **Structure:**
        - **File:** water_dataset.mat
        - **Locations:** 37 stations across Georgia, USA
        - **Training pool:** 423 daily samples → 8,724 train + 2,908 validation sequences
        - **Test pool:** 282 separate daily samples → **9,324 test sequences**
        - **Features:** 11 water quality indices per station
        - **Target:** Next-day pH (1-day horizon)

        **Preprocessing:**
        - MinMax scaling (normalized to 0–1 range), fit on training data only
        - Sliding-window sequence creation (30-day lookback)

        **Why this dataset?**
        - Real, densely-sampled multi-station time series
        - Published, peer-reviewed source

        *Note: this dataset covers a different water system than Dataset 1 —
        the two modules were validated independently, not on one shared,
        physically-linked monitoring network.*
        """)

    st.markdown("---")

    # System Architecture
    st.markdown("## 🏗️ System Architecture - How It All Works")

    st.markdown("""
    <div class='info-box'>
        <h3>📊 Complete Process Flow:</h3>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    STEP 1: DATA COLLECTION (TWO DATASETS)

    ┌─────────────────────────────────────────────────────────────┐
    │           DATASET 1: Water Potability (Kaggle)              │
    │  • 3,276 samples → 2,847 after preprocessing                │
    │  • 9 physicochemical parameters                             │
    │  • Used for: Classification & Anomaly Detection             │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │   DATASET 2: Water Quality Prediction (Georgia, USA/USGS)   │
    │  • 37 stations, 11 features per station                     │
    │  • Spatio-temporal pH prediction dataset                    │
    │  • Used for: Forecasting (BiLSTM/GRU)                       │
    │  • Citation: Liang Zhao et al. 2019 (ACM TSAS)              │
    └─────────────────────────────────────────────────────────────┘

    ↓

    STEP 2: DATA PREPROCESSING

    For Classification & Anomaly (Dataset 1):
    ├── Handle missing values (KNN Imputation, k=5)
    ├── Remove outliers (IQR method — 429 samples removed)
    ├── Split FIRST: 2,277 train / 570 test (80/20)
    ├── Normalize (StandardScaler, fit on train only)
    └── Balance TRAINING set only (SMOTE: 860→1,417 minority)
        — test set (570) keeps its real ~62/38 split, untouched

    For Forecasting (Dataset 2):
    ├── Normalize features (MinMaxScaler 0-1, fit on train only)
    ├── Create 30-day lookback sequences
    └── 8,724 train / 2,908 val / 9,324 test sequences

    ↓

    STEP 3: PARALLEL AI ANALYSIS (All 3 run independently)

    ┌─────────────────────────────────────────────────────────────┐
    │              CLASSIFICATION MODULE                           │
    │              Uses: Dataset 1 (Water Potability)              │
    ├─────────────────────────────────────────────────────────────┤
    │  → Random Forest (200 trees) — 63.33% acc, 66.36% AUC       │
    │  → XGBoost (150 trees)       — 59.82% acc, 65.01% AUC       │
    │  → SVM (RBF kernel)          — 60.53% acc, 64.12% AUC       │
    │  → Weighted Ensemble (.35/.40/.25) — 61.75% acc, 66.62% AUC │
    │  OUTPUT: POTABLE (1) or NOT POTABLE (0)                     │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │              FORECASTING MODULE                              │
    │              Uses: Dataset 2 (Georgia USGS)                  │
    ├─────────────────────────────────────────────────────────────┤
    │  → BiLSTM with Attention (R² = 0.5251, RMSE = 0.0524)       │
    │  → GRU with Attention, baseline (R² = 0.3572)                │
    │  → Input: 30-day lookback × 11 features                     │
    │  → Predicts next-day pH (1-day horizon)                     │
    │  OUTPUT: Next-day pH value                                  │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │           ANOMALY DETECTION MODULE                           │
    │           Uses: Dataset 1 (Water Potability)                 │
    ├─────────────────────────────────────────────────────────────┤
    │  → Isolation Forest (100 trees, contamination=0.05)          │
    │  → LOF (k=20 neighbors)                                     │
    │  → One-Class SVM (RBF kernel, nu=0.05)                      │
    │  → Autoencoder (9→16→8→16→9, 95th-percentile threshold)     │
    │  → Quorum Voting (≥2-of-4 detectors = anomaly)               │
    │  OUTPUT: NORMAL (0) or ANOMALY (1) — 4.21% flagged           │
    └─────────────────────────────────────────────────────────────┘

    ↓

    STEP 4: ALERT GENERATION (Combines all 3 outputs)
    ├── Classification confidence (C)
    ├── Forecasting deviation (F)
    ├── Anomaly vote ratio (A)
    └── R = 0.40·C + 0.30·F + 0.30·A → Composite Risk Score

    ↓

    STEP 5: 4-LEVEL ALERT SYSTEM
    ├── R < 0.25:        🟢 NORMAL   → Water is safe, continue monitoring
    ├── R 0.25 - 0.50:   🟡 LOW      → Minor concern, increase monitoring
    ├── R 0.50 - 0.75:   🟠 MEDIUM   → Significant issue, investigate now
    └── R ≥ 0.75:        🔴 CRITICAL → Contamination, emergency action!

    ↓

    STEP 6: OPERATOR DASHBOARD
    ├── Display results & visualizations
    ├── Generate reports
    ├── Log alert history
    └── Export data
    """, language="text")

    st.markdown("---")

    # Key Metrics
    st.markdown("## 📊 System Performance Metrics — Verified & Reproducible")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Classification (Ensemble)", "61.75%", "Accuracy")
    with col2:
        st.metric("📈 Forecasting R²", "0.5251", "BiLSTM-Attention")
    with col3:
        st.metric("⚠️ Anomaly Flag Rate", "4.21%", "120 / 2,847 samples")
    with col4:
        st.metric("⚡ Processing Speed", "<100ms", "Per sample")

    st.caption(
        "These figures were corrected after a SMOTE data-leakage bug was found and fixed during peer review "
        "(synthetic training samples had been leaking into the test set, inflating the original numbers). "
        "The values above are the honest, independently-reproduced results."
    )

    st.markdown("---")

    # Quick Start
    st.markdown("## 🚀 How to Use This Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📝 For Testing Individual Samples

        **1. Go to "🔍 Water Classification"**
        - Enter 9 water quality parameters manually
        - OR upload a CSV file with multiple samples
        - Get instant POTABLE/NOT POTABLE results
        - Download results as CSV

        **2. Check for Unusual Patterns**
        - Go to "⚠️ Anomaly Detection"
        - Upload your water quality data
        - System will flag unusual/strange samples
        - Review anomalies for investigation
        """)

    with col2:
        st.markdown("""
        ### 📈 For Continuous Monitoring

        **3. Go to "📈 pH Forecasting"**
        - Upload time-series pH data (hourly/daily)
        - View historical pH trends
        - See next-day forecast
        - Get alerts if pH is trending out of safe range

        **4. Review Historical Alerts**
        - Go to "🚨 Alert History"
        - View all past alerts
        - Analyze patterns over time
        - Export alert logs
        """)

    st.markdown("---")

    # Technical Details
    st.markdown("## 🔬 Technical Implementation")

    with st.expander("📚 Machine Learning Models Used"):
        st.markdown("""
        **Classification Models** *(weighted soft-voting: RF 0.35 · XGBoost 0.40 · SVM 0.25)*
        - Random Forest: 200 trees, max depth 15
        - XGBoost: 150 estimators, learning rate 0.1
        - SVM: RBF kernel, C = 10

        **Forecasting Models**
        - BiLSTM-Attention: 2 stacked Bidirectional LSTM layers (128 → 64 units) + additive-attention-style
          temporal pooling layer
        - GRU-Attention: identical setup with GRU cells, used as the baseline comparison

        **Anomaly Detection** *(quorum vote: ≥2-of-4 must agree)*
        - Isolation Forest: 100 trees, contamination = 0.05
        - LOF: 20 neighbors
        - One-Class SVM: RBF kernel, ν = 0.05
        - Autoencoder: 9 → 16 → 8 → 16 → 9 (ReLU), threshold at 95th percentile of reconstruction error
        """)

    with st.expander("📊 Dataset Information"):
        st.markdown("""
        ### Dataset 1: Water Potability (Kaggle)
        **Used for:** Classification & Anomaly Detection

        **Original Size:** 3,276 samples → **After Preprocessing:** 2,847 samples (429 outliers removed)

        **Features:** 9 physicochemical parameters (pH, Hardness, Solids, Chloramines, Sulfate,
        Conductivity, Organic Carbon, Trihalomethanes, Turbidity)

        **Target:** Potable (1) or Not Potable (0)

        **Split:** 2,277 train / 570 test (80/20) — **split before** SMOTE balancing

        **Class Balance:**
        - Test set (570): real ~62%/38% split, never rebalanced
        - Training set only: 860 → 1,417 minority samples via SMOTE (2,834 total)

        ---

        ### Dataset 2: Water Quality Prediction (Georgia, USA)
        **Used for:** Forecasting (BiLSTM & GRU models)

        **Source:** United States Geological Survey (USGS)

        **Citation:** Liang Zhao, Olga Gkountouna, and Dieter Pfoser. 2019.
        "Spatial Auto-regressive Dependency Interpretable Learning Based on
        Spatial Topological Constraints." ACM Trans. Spatial Algorithms Syst.
        5, 3, Article 19 (August 2019), 28 pages. DOI: https://doi.org/10.1145/3339823

        **File Format:** water_dataset.mat (MATLAB format)

        **Structure:**
        - **Locations:** 37 water stations across Georgia, USA
        - **Training pool:** 423 daily samples → 8,724 train + 2,908 validation sequences
        - **Testing pool:** 282 separate daily samples → 9,324 test sequences
        - **Features per location:** 11 water quality indices
        - **Target:** Next-day pH (1-day horizon)

        **Preprocessing:** MinMax scaling (fit on training data only), 30-day lookback windows
        """)

    with st.expander("⚙️ System Requirements"):
        st.markdown("""
        **Software:**
        - Python 3.8+
        - Streamlit 1.28+
        - scikit-learn 1.3+
        - TensorFlow 2.x (for forecasting)

        **Hardware:**
        - Standard CPU — no GPU required for inference
        - Verified compute: BiLSTM training ≈ 180s (one-time); inference + alert generation < 100ms/sample

        **Deployment:**
        - Runs on laptop, server, or cloud
        - Real-time capable for continuous monitoring at this scale
        """)

    st.markdown("---")

    # Research Context
    st.markdown("## 🎓 Research Context")

    st.markdown("""
    <div class='info-box'>
        <h3>📄 IEEE Paper — ICESC 2026 (Paper ID: ICESC_469)</h3>
        <p><b>Title:</b> Integrated ML Framework for Real-Time Water Quality Monitoring</p>
        <p><b>Objective:</b> Fuse classification, forecasting, and anomaly detection into one operational,
        explainable early-warning pipeline</p>
        <p><b>Contribution:</b> The novelty is architectural integration of all three modules with SHAP
        explainability — not a claim of state-of-the-art accuracy on any single metric</p>
    </div>
    """, unsafe_allow_html=True)

    # Final call to action
    st.markdown("---")
    st.success("""
    ### ✅ Ready to Start?

    **Click on any page in the sidebar to begin:**
    - 🔍 **Water Classification** - Test if water is safe now
    - 📈 **pH Forecasting** - See next-day pH trend
    - ⚠️ **Anomaly Detection** - Find unusual contamination patterns
    """)

if __name__ == "__main__":
    show()
