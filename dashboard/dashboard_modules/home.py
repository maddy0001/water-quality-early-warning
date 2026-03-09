"""
Home Page - Comprehensive Project Overview
"""

import streamlit as st

def show():
    """Display comprehensive home page"""
    
    # Project Overview
    st.markdown("""
    <div class='info-box'>
        <h2 style='margin-top: 0; color: #00d4ff;'>🎯 What is This Project?</h2>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
            This is an <b>AI-Powered Water Contamination Early Warning System</b> that monitors water quality in real-time 
            and predicts contamination events <b>BEFORE they happen</b>.
        </p>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
            <b>The Problem:</b> Traditional water testing is slow (takes hours/days in labs). By the time contamination is detected, 
            people may have already consumed unsafe water.
        </p>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
            <b>Our Solution:</b> Use Machine Learning to analyze water samples instantly (&lt;100ms) and forecast future 
            quality trends, giving advance warning to prevent contamination disasters.
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
            <p><b>Accuracy: 72.5%</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='alert-low'>
            <h3>📈 Future Prediction</h3>
            <p><b>Predicts what WILL happen</b></p>
            <p>→ Forecasts pH 24-48h ahead</p>
            <p>→ Early contamination warning</p>
            <p>→ Time to prevent disaster</p>
            <p><b>R² Score: 0.52</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='alert-medium'>
            <h3>⚠️ Pattern Detection</h3>
            <p><b>Finds unusual samples</b></p>
            <p>→ 4 AI detectors working</p>
            <p>→ Catches strange patterns</p>
            <p>→ Contamination events</p>
            <p><b>Detection: 4.2% rate</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear explanation of each component
    st.markdown("## 🔍 Understanding The 3 Core Components")
    
    st.markdown("""
    <div class='explain-box'>
        <h3>💡 IMPORTANT: What's the Difference?</h3>
        <p>Many people ask: "Why do we need Classification AND Anomaly Detection? Aren't they the same?"</p>
        <p><b>NO! They serve different purposes. Here's the clear difference:</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Component 1: Classification
    st.markdown("### 1️⃣ Water Classification - \"Is This Water Safe RIGHT NOW?\"")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **What it does:**
        - Takes a single water sample with 9 measurements (pH, hardness, solids, etc.)
        - Runs it through 3 AI models (Random Forest, XGBoost, SVM)
        - Answers: **"Can you drink this water? YES or NO"**
        
        **Real-world example:**
        - You collect water from a tap
        - You measure: pH=7.2, Hardness=200mg/L, Solids=20000ppm, etc.
        - System says: **"✅ POTABLE - Safe to drink"** or **"❌ NOT POTABLE - Do not consume"**
        
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
        
        **Accuracy:** 72.5%
        """)
    
    st.markdown("---")
    
    # Component 2: Forecasting
    st.markdown("### 2️⃣ pH Forecasting - \"What WILL Happen in the Next 24-48 Hours?\"")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **What it does:**
        - Takes historical pH measurements over time (e.g., last 30 hours)
        - Uses deep learning (BiLSTM with Attention) to learn trends
        - Predicts: **"What will the pH be in 24-48 hours?"**
        
        **Real-world example:**
        - Your pH has been: 7.0 → 7.1 → 6.9 → 6.8 → 6.7 (declining trend)
        - System predicts: **"In 24 hours, pH will be 6.4 (below safe limit!)"**
        - You get **EARLY WARNING** to adjust treatment NOW, before contamination happens
        
        **When to use:**
        - Continuous water quality monitoring
        - Predicting if current trends will cause problems
        - Planning treatment adjustments in advance
        - Preventing contamination before it occurs
        """)
    
    with col2:
        st.info("""
        **Output:**
        - Future pH values
        - Trend direction
        - Alert if going out of safe range
        
        **Horizon:** 24-48 hours
        
        **R² Score:** 0.52
        """)
    
    st.markdown("---")
    
    # Component 3: Anomaly Detection
    st.markdown("### 3️⃣ Anomaly Detection - \"Is This Sample UNUSUAL or STRANGE?\"")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **What it does:**
        - Compares each water sample to thousands of "normal" samples
        - Uses 4 different AI detectors to find unusual patterns
        - Answers: **"Is this sample weird/different/anomalous?"**
        
        **Real-world example:**
        - Normal samples have: pH=6.5-8.5, Hardness=150-300, etc.
        - You get a sample: pH=7.0 (normal), but Hardness=500 (very high!) + Conductivity=1200 (very high!)
        - Even though pH is OK, the COMBINATION is unusual
        - System flags: **"⚠️ ANOMALY - Unusual pattern detected, investigate!"**
        
        **When to use:**
        - Finding contamination events (chemical spills, industrial discharge)
        - Detecting sensor malfunctions
        - Identifying samples that don't fit normal patterns
        - Catching problems that classification might miss
        """)
    
    with col2:
        st.warning("""
        **Output:**
        - Normal or Anomaly
        - Number of detectors agreeing (0-4 votes)
        - Confidence level
        
        **Methods:** 4-detector ensemble
        
        **Detection Rate:** 4.2%
        """)
    
    st.markdown("---")
    
    # Key Differences
    st.markdown("## 🎯 Classification vs Forecasting vs Anomaly Detection")
    
    st.markdown("""
    | Aspect | Classification | Forecasting | Anomaly Detection |
    |--------|---------------|-------------|-------------------|
    | **Question** | Is water safe NOW? | What will happen LATER? | Is this sample UNUSUAL? |
    | **Dataset** | Dataset 1 (Water Potability) | Dataset 2 (Georgia USGS) | Dataset 1 (Water Potability) |
    | **Input** | Single sample (9 params) | Time series (30 timesteps × 11 features, 37 locations) | Single sample (9 params) |
    | **Output** | POTABLE / NOT POTABLE | Future pH values (next day, all locations) | NORMAL / ANOMALY |
    | **Time** | Present moment | 24-48 hours ahead | Present moment |
    | **Purpose** | Immediate safety check | Early warning prediction | Pattern/contamination detection |
    | **Example** | "Can I drink this?" | "Will pH be safe tomorrow?" | "Is this sample strange?" |
    | **Model** | Random Forest, XGBoost, SVM | BiLSTM-Attention (Deep Learning) | 4 detectors (IF, LOF, OCSVM, AE) |
    | **Accuracy** | 72.5% | R²=0.52 | 83% precision |
    """)
    
    st.markdown("---")
    
    # Data Sources
    st.markdown("## 📦 Data Sources & Preparation")
    
    st.markdown("""
    <div class='explain-box'>
        <p><b>Important:</b> This system uses TWO different REAL datasets from different sources!</p>
        <p><b>Dataset 1 (Kaggle):</b> Individual water samples → Classification & Anomaly</p>
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
        - **Features:** 9 water quality parameters
        - **Target:** Potable (1) or Not Potable (0)
        - **Balance:** Imbalanced (39% Potable, 61% Not Potable)
        
        **After Preprocessing:**
        - **Size:** 2,847 samples (removed 429 outliers using IQR method)
        - **Balance:** 50-50 after SMOTE
        - **Normalization:** StandardScaler (mean=0, std=1)
        - **Ready for:** Classification & Anomaly models
        
        **Why this dataset?**
        - Real-world water quality measurements
        - Covers key contamination indicators (pH, hardness, etc.)
        - Sufficient size for ML training
        - Industry-standard parameters
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Dataset 2: Water Quality Prediction (Georgia, USA)
        **Source:** United States Geological Survey (USGS)
        
        **Citation:** Liang Zhao et al. 2019 (ACM TSAS)
        
        **Used for:** pH Forecasting (BiLSTM & GRU)
        
        **Structure:**
        - **File:** water_dataset.mat
        - **Locations:** 37 water stations in Georgia, USA
        - **Training:** 423 daily samples (from 2016-01-28)
        - **Testing:** 282 daily samples (until 2018-01-01)
        - **Features:** 11 water quality indices per location
        - **Target:** Next-day pH prediction for all 37 locations
        
        **Spatial Characteristics:**
        - 3 connected water systems
        - One centered on Atlanta
        - One on eastern coast of Georgia
        - Spatio-temporal dependencies
        
        **Preprocessing:**
        - MinMax scaling (normalized to 0-1 range)
        - Sequence creation (30 timestep lookback)
        - Train/Test split: 80/20
        
        **Why this dataset?**
        - Real-world spatio-temporal pH data
        - Captures geographic dependencies
        - Multi-location forecasting capability
        - Published in peer-reviewed research
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
    │  • 9 water quality parameters                               │
    │  • Used for: Classification & Anomaly Detection             │
    └─────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────┐
    │   DATASET 2: Water Quality Prediction (Georgia, USA/USGS)   │
    │  • 37 locations × 423 training dates × 11 features          │
    │  • Spatio-temporal pH prediction dataset                    │
    │  • Used for: Forecasting (BiLSTM/GRU)                       │
    │  • Citation: Liang Zhao et al. 2019 (ACM TSAS)              │
    └─────────────────────────────────────────────────────────────┘
    
    ↓
    
    STEP 2: DATA PREPROCESSING
    
    For Classification & Anomaly (Dataset 1):
    ├── Handle missing values (KNN Imputation)
    ├── Remove outliers (IQR method - removed 429 samples)
    ├── Normalize data (StandardScaler)
    └── Balance classes (SMOTE: 39/61 → 50/50)
    
    For Forecasting (Dataset 2):
    ├── Extract sequences from spatio-temporal data
    ├── Create lookback windows (30 timesteps)
    ├── Normalize features (MinMaxScaler 0-1)
    └── Train/Test split (80/20)
    
    ↓
    
    STEP 3: PARALLEL AI ANALYSIS (All 3 run simultaneously)
    
    ┌─────────────────────────────────────────────────────────────┐
    │              CLASSIFICATION MODULE                           │
    │              Uses: Dataset 1 (Water Potability)              │
    ├─────────────────────────────────────────────────────────────┤
    │  → Random Forest (72.5% accuracy)                           │
    │  → XGBoost (70.7% accuracy)                                 │
    │  → SVM (68.4% accuracy)                                     │
    │  → Ensemble Voting                                          │
    │  OUTPUT: POTABLE (1) or NOT POTABLE (0)                     │
    └─────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────┐
    │              FORECASTING MODULE                              │
    │              Uses: Dataset 2 (Georgia USGS)                  │
    ├─────────────────────────────────────────────────────────────┤
    │  → BiLSTM with Attention (R²=0.518)                         │
    │  → GRU with Attention (R²=0.365)                            │
    │  → Input: 30 timesteps × 11 features                        │
    │  → Predicts next 1-2 steps (24-48 hours)                    │
    │  OUTPUT: Future pH values                                   │
    └─────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────┐
    │           ANOMALY DETECTION MODULE                           │
    │           Uses: Dataset 1 (Water Potability)                 │
    ├─────────────────────────────────────────────────────────────┤
    │  → Isolation Forest (finds isolated samples)                │
    │  → LOF (local density comparison)                           │
    │  → One-Class SVM (learns normal boundary)                   │
    │  → Autoencoder (reconstruction error)                       │
    │  → Majority Voting (≥2 detectors = anomaly)                 │
    │  OUTPUT: NORMAL (0) or ANOMALY (1)                          │
    └─────────────────────────────────────────────────────────────┘
    
    ↓
    
    STEP 4: ALERT GENERATION (Combines all 3 outputs)
    ├── Classification result (Potable?)
    ├── Forecasting prediction (Future safe?)
    ├── Anomaly detection (Unusual?)
    └── Weighted combination → Final Risk Score (0-100)
    
    ↓
    
    STEP 5: 4-LEVEL ALERT SYSTEM
    ├── Risk 0-25:   🟢 NORMAL   → Water is safe, continue monitoring
    ├── Risk 25-50:  🟡 LOW      → Minor concern, increase monitoring
    ├── Risk 50-75:  🟠 MEDIUM   → Significant issue, investigate now
    └── Risk 75-100: 🔴 CRITICAL → Contamination, emergency action!
    
    ↓
    
    STEP 6: OPERATOR DASHBOARD
    ├── Display results
    ├── Show visualizations
    ├── Generate reports
    ├── Log alert history
    └── Export data
    """, language="text")
    
    st.markdown("---")
    
    # Key Metrics
    st.markdown("## 📊 System Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Classification Accuracy", "72.5%", "Random Forest")
    
    with col2:
        st.metric("📈 Forecasting R²", "0.52", "BiLSTM")
    
    with col3:
        st.metric("⚠️ Anomaly Precision", "83%", "Ensemble")
    
    with col4:
        st.metric("⚡ Processing Speed", "<100ms", "Per sample")
    
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
        - See forecast for next 24-48 hours
        - Get alerts if pH will leave safe range
        
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
        **Classification Models:**
        - Random Forest: Ensemble of 100 decision trees
        - XGBoost: Gradient boosting with 100 estimators
        - SVM: Radial basis function kernel
        
        **Forecasting Models:**
        - BiLSTM-Attention: 2-layer bidirectional LSTM with attention mechanism
        - GRU-Attention: 2-layer GRU with attention
        
        **Anomaly Detection:**
        - Isolation Forest: 100 trees, contamination=0.05
        - LOF: 20 neighbors
        - One-Class SVM: RBF kernel, nu=0.05
        - Autoencoder: 3-layer neural network
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        ### Dataset 1: Water Potability (Kaggle)
        **Used for:** Classification & Anomaly Detection
        
        **Source:** Kaggle Water Potability Dataset
        
        **Original Size:** 3,276 samples
        
        **After Preprocessing:** 2,847 samples (removed 429 outliers)
        
        **Features:** 9 water quality parameters (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity)
        
        **Target:** Potable (1) or Not Potable (0)
        
        **Class Balance:** 
        - Original: 39% Potable, 61% Not Potable (imbalanced)
        - After SMOTE: 50-50 (balanced)
        
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
        - **Training Data:** 423 contiguous daily samples (from 2016-01-28)
        - **Testing Data:** 282 contiguous daily samples (until 2018-01-01)
        - **Features per location:** 11 water quality indices
        - **Target:** pH (power of hydrogen) - next day prediction
        
        **Spatial Organization:**
        - 3 connected water system groups
        - System 1: Centered on Atlanta
        - System 2: Eastern coast of Georgia
        - System 3: Additional monitoring network
        
        **11 Features:**
        - Volume of dissolved oxygen
        - Temperature
        - Specific conductance
        - (and 8 additional water quality indices)
        
        **Preprocessing:** MinMax scaling (0-1 normalization), 30-timestep lookback windows
        """)
    
    with st.expander("⚙️ System Requirements"):
        st.markdown("""
        **Software:**
        - Python 3.8+
        - Streamlit 1.28+
        - scikit-learn 1.3+
        - TensorFlow 2.x (for forecasting)
        
        **Hardware:**
        - Minimum: 4GB RAM, 2-core CPU
        - Recommended: 8GB RAM, 4-core CPU
        - Processing: Single sample in <100ms
        
        **Deployment:**
        - Can run on laptop, server, or cloud
        - Real-time capable for continuous monitoring
        """)
    
    st.markdown("---")
    
    # Research Context
    st.markdown("## 🎓 Research Context")
    
    st.markdown("""
    <div class='info-box'>
        <h3>📄 IEEE Paper Submission (2026)</h3>
        <p><b>Title:</b> AI-Powered Water Contamination Early Warning System Using Ensemble Machine Learning and Deep Learning Forecasting</p>
        <p><b>Objective:</b> Develop a real-time water quality monitoring system that predicts contamination events before they occur</p>
        <p><b>Innovation:</b> Combines classification, forecasting, and anomaly detection in a unified system</p>
        <p><b>Impact:</b> Prevents waterborne diseases by providing advance warning to water treatment operators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Final call to action
    st.markdown("---")
    st.success("""
    ### ✅ Ready to Start?
    
    **Click on any page in the sidebar to begin:**
    - 🔍 **Water Classification** - Test if water is safe now
    - 📈 **pH Forecasting** - Predict future water quality
    - ⚠️ **Anomaly Detection** - Find unusual contamination patterns
    """)

if __name__ == "__main__":
    show()
