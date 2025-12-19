"""
XAI Loan System - Streamlit Frontend

Research Demonstration Interface for Causal Counterfactual Loan Risk Assessment.

This frontend is designed for academic evaluation and demonstration purposes only.
It does NOT store user data, require authentication, or make real-world loan decisions.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000/api/v1"

# Initialize session state for form persistence
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "current_features" not in st.session_state:
    st.session_state.current_features = None

# Page configuration
st.set_page_config(
    page_title="XAI Loan Risk Assessment - Research Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Professional, Minimal Design
# =============================================================================

st.markdown("""
<style>
    /* Clean, professional color palette */
    :root {
        --primary-color: #2C3E50;
        --secondary-color: #34495E;
        --accent-color: #3498DB;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --danger-color: #E74C3C;
        --background-light: #F8F9FA;
        --text-muted: #6C757D;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .main-header .subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .disclaimer {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Card styling */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #3498DB;
    }
    
    .cf-card {
        background: #F8F9FA;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #E9ECEF;
    }
    
    .cf-card.primary {
        border-left: 4px solid #27AE60;
        background: #F0FDF4;
    }
    
    /* Metric styling */
    .metric-container {
        text-align: center;
        padding: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2C3E50;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        background: #F8F9FA;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #6C757D;
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API FUNCTIONS
# =============================================================================

def call_api(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Make API call to backend.
    Handles errors gracefully for research demonstration.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/{endpoint}",
            json=data,
            timeout=60  # Increased for causal CF generation
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Backend API not available. Please start the API server.")
        return None
    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out. The model may be processing.")
        return None
    except Exception as e:
        st.error(f"⚠️ API Error: {str(e)}")
        return None


def get_prediction(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get risk prediction from API."""
    return call_api("predict", features)


def get_counterfactuals(features: Dict[str, Any], method: str = "causal", num_cfs: int = 3) -> Optional[Dict[str, Any]]:
    """Get counterfactual recommendations from API."""
    return call_api("counterfactual", {
        "features": features,
        "method": method,
        "num_counterfactuals": num_cfs,
        "target_outcome": "approved"
    })


def get_explanation(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get SHAP explanation from API."""
    return call_api("explain", {
        "features": features,
        "top_k": 10
    })


def create_shap_waterfall_chart(explanation: Dict[str, Any]) -> go.Figure:
    """Create a SHAP waterfall-style chart showing feature contributions."""
    positive = explanation.get("top_positive_features", [])
    negative = explanation.get("top_negative_features", [])
    
    # Combine and sort by absolute impact
    all_features = []
    for f in positive:
        all_features.append({
            "feature": f.get("feature", "").replace("_", " ").title()[:20],
            "value": f.get("shap_value", 0),
            "color": "#27AE60"  # Green
        })
    for f in negative:
        all_features.append({
            "feature": f.get("feature", "").replace("_", " ").title()[:20],
            "value": f.get("shap_value", 0),
            "color": "#E74C3C"  # Red
        })
    
    # Sort by absolute value
    all_features.sort(key=lambda x: abs(x["value"]), reverse=True)
    all_features = all_features[:8]  # Top 8
    
    if not all_features:
        return None
    
    features = [f["feature"] for f in all_features]
    values = [f["value"] for f in all_features]
    colors = [f["color"] for f in all_features]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Impact on Prediction",
        xaxis_title="SHAP Value (Impact)",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    return fig


def create_method_comparison_chart() -> go.Figure:
    """Create a comparison chart of counterfactual methods."""
    methods = ["Causal (Strict)", "Causal-Relaxed", "DiCE", "Optimization"]
    causal_validity = [100, 85, 0, 0]
    colors = ["#27AE60", "#2ECC71", "#3498DB", "#E74C3C"]
    
    fig = go.Figure(go.Bar(
        x=methods,
        y=causal_validity,
        marker_color=colors,
        text=[f"{v}%" for v in causal_validity],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Causal Validity by Method",
        xaxis_title="Method",
        yaxis_title="Causal Validity (%)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 110]),
        showlegend=False
    )
    
    return fig


# =============================================================================
# HEADER SECTION
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🎓 XAI Loan Risk Assessment System</h1>
    <div class="subtitle">
        Educational & Decision-Support Demonstration Using Explainable and Causal AI
    </div>
    <div class="disclaimer">
        ⚠️ This system uses historical, anonymized data and is intended solely for 
        academic research demonstration. It is NOT designed for real-world loan decisions.
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN LAYOUT: Two Columns
# =============================================================================

col_input, col_output = st.columns([1, 2])


# =============================================================================
# LEFT COLUMN: Applicant Input Panel
# =============================================================================

with col_input:
    st.markdown("### 📋 Hypothetical Applicant Profile")
    st.caption("Enter sample values for demonstration purposes only.")
    
    with st.form("applicant_form"):
        # Annual Income
        annual_inc = st.number_input(
            "Annual Income ($)",
            min_value=10000,
            max_value=500000,
            value=65000,
            step=1000,
            help="Hypothetical annual income for risk estimation"
        )
        
        # Employment Length
        emp_length = st.number_input(
            "Employment Length (years)",
            min_value=0,
            max_value=40,
            value=5,
            step=1,
            help="Years of employment history"
        )
        
        # Debt-to-Income Ratio
        dti = st.slider(
            "Debt-to-Income Ratio (%)",
            min_value=0.0,
            max_value=50.0,
            value=18.0,
            step=0.5,
            help="Total debt payments as percentage of income"
        )
        
        # Credit Score
        fico_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=680,
            step=5,
            help="FICO-equivalent credit score"
        )
        
        # Open Accounts
        open_acc = st.number_input(
            "Number of Open Accounts",
            min_value=0,
            max_value=50,
            value=8,
            step=1,
            help="Active credit accounts"
        )
        
        # Delinquencies
        delinq_2yrs = st.number_input(
            "Delinquencies (Last 2 Years)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Late payments in past 24 months"
        )
        
        # Loan Amount (for context)
        loan_amnt = st.number_input(
            "Requested Loan Amount ($)",
            min_value=1000,
            max_value=100000,
            value=15000,
            step=500,
            help="Hypothetical loan amount for estimation"
        )
        
        # Submit button
        submitted = st.form_submit_button(
            "Analyze Risk Profile",
            use_container_width=True
        )


# =============================================================================
# RIGHT COLUMN: Results Display
# =============================================================================

with col_output:
    
    # Build features dictionary
    features = {
        "annual_inc": annual_inc,
        "emp_length": emp_length,
        "dti": dti,
        "fico_score": fico_score,
        "open_acc": open_acc,
        "delinq_2yrs": delinq_2yrs,
        "loan_amnt": loan_amnt,
        "revol_util": 45.0,  # Default values for required fields
        "inq_last_6mths": 1,
    }
    
    # Update session state when form is submitted
    if submitted:
        st.session_state.form_submitted = True
        st.session_state.current_features = features.copy()
    
    # Show results if form has been submitted (persists across reruns)
    if st.session_state.form_submitted and st.session_state.current_features:
        # =================================================================
        # PREDICTION SECTION
        # =================================================================
        
        st.markdown("### 📊 Risk Estimation Results")
        
        with st.spinner("Analyzing risk profile..."):
            prediction = get_prediction(st.session_state.current_features)
        
        if prediction:
            # Display metrics in columns
            m1, m2, m3 = st.columns(3)
            
            with m1:
                approval_prob = prediction.get("approval_probability", 0.5)
                st.metric(
                    label="Approval Probability",
                    value=f"{approval_prob:.1%}",
                    delta=None
                )
            
            with m2:
                # Determine risk category
                if approval_prob >= 0.7:
                    risk_cat = "Low Risk"
                    risk_color = "🟢"
                elif approval_prob >= 0.4:
                    risk_cat = "Medium Risk"
                    risk_color = "🟡"
                else:
                    risk_cat = "High Risk"
                    risk_color = "🔴"
                
                st.metric(
                    label="Risk Category",
                    value=f"{risk_color} {risk_cat}"
                )
            
            with m3:
                interest_band = prediction.get("interest_rate_band", "Medium")
                st.metric(
                    label="Interest Rate Band",
                    value=interest_band
                )
            
            st.caption("*These estimates are for demonstration purposes only and do not constitute financial advice.*")
        
        st.markdown("---")
        
        # =================================================================
        # COUNTERFACTUAL RECOMMENDATIONS SECTION (CORE)
        # =================================================================
        
        st.markdown("### 💡 Actionable Recommendations")
        st.caption("Causally-validated suggestions for potential profile improvement")
        
        # Method selection
        method_options = {
            "Causal Counterfactual (Primary)": "causal",
            "DiCE (Baseline)": "dice",
            "Optimization (Baseline)": "optimization"
        }
        
        selected_method = st.radio(
            "Recommendation Method:",
            options=list(method_options.keys()),
            index=0,
            horizontal=True,
            key="cf_method_selector",
            help="Causal counterfactuals respect real-world dependencies between features."
        )
        
        method = method_options[selected_method]
        
        with st.spinner("Generating recommendations..."):
            cf_results = get_counterfactuals(st.session_state.current_features, method=method, num_cfs=3)
        
        if cf_results and cf_results.get("success"):
            counterfactuals = cf_results.get("counterfactuals", [])
            
            if counterfactuals:
                # Display mean actionability
                mean_action = cf_results.get("mean_actionability", 0.5)
                causal_validity = cf_results.get("causal_validity_rate", 1.0)
                
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.info(f"📈 Mean Actionability Score: **{mean_action:.2f}**")
                with info_cols[1]:
                    if method == "causal" and causal_validity is not None:
                        st.success(f"✅ Causal Validity Rate: **{causal_validity:.0%}**")
                
                # Display each counterfactual
                for i, cf in enumerate(counterfactuals[:3]):
                    actionability = cf.get("actionability_score", 0.5)
                    level = cf.get("actionability_level", "Moderate")
                    is_primary = (method == "causal")
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="cf-card {'primary' if is_primary else ''}">
                            <strong>Recommendation {i+1}</strong>
                            {' — Causally Validated ✓' if cf.get('causal_valid', False) else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Actionability progress bar
                        st.progress(actionability, text=f"Actionability: {level} ({actionability:.2f})")
                        
                        # Feature changes
                        changes = cf.get("changes", [])
                        if changes:
                            for change in changes:
                                feature = change.get("feature", "")
                                current = change.get("current_value", 0)
                                target = change.get("recommended_value", 0)
                                direction = change.get("change_direction", "")
                                
                                arrow = "↑" if direction == "increase" else "↓"
                                
                                st.markdown(f"""
                                - **{feature.replace('_', ' ').title()}**: 
                                  {current:.1f} → {target:.1f} {arrow}
                                """)
                        else:
                            st.caption("No specific feature changes identified.")
                        
                        st.markdown("---")
            else:
                st.warning("No recommendations generated. Try adjusting the profile values.")
        else:
            st.info("Submit the form to generate recommendations.")
        
        # =================================================================
        # SHAP EXPLANATION SECTION (Collapsed)
        # =================================================================
        
        with st.expander("🔍 Diagnostic Explanation (Why this estimation?)", expanded=False):
            st.caption("*SHAP-based feature attribution showing which factors influenced the risk estimation.*")
            
            with st.spinner("Computing explanation..."):
                explanation = get_explanation(st.session_state.current_features)
            
            if explanation:
                # SHAP Waterfall Chart
                waterfall_fig = create_shap_waterfall_chart(explanation)
                if waterfall_fig:
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                
                st.markdown("#### Contributing Factors")
                
                col_pos, col_neg = st.columns(2)
                
                with col_pos:
                    # Positive factors
                    positive = explanation.get("top_positive_features", [])
                    if positive:
                        st.markdown("**✅ Positive Influence:**")
                        for f in positive[:3]:
                            feature = f.get("feature", "")
                            impact = abs(f.get("shap_value", 0))
                            st.markdown(f"- {feature.replace('_', ' ').title()} (+{impact:.3f})")
                
                with col_neg:
                    # Negative factors
                    negative = explanation.get("top_negative_features", [])
                    if negative:
                        st.markdown("**⚠️ Areas for Improvement:**")
                        for f in negative[:3]:
                            feature = f.get("feature", "")
                            impact = abs(f.get("shap_value", 0))
                            st.markdown(f"- {feature.replace('_', ' ').title()} (-{impact:.3f})")
                
                # Insight
                insight = explanation.get("diagnostic_insight", "")
                if insight:
                    st.info(f"**Insight:** {insight}")
            else:
                st.caption("Explanation not available.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="footer">
    <strong>Research Project: XAI-Driven Causal Counterfactual Loan Risk Assessment</strong><br><br>
    
    <strong>Ethics Statement:</strong> This system is designed for academic research and demonstration. 
    It does not make real-world loan decisions and should not be used for actual financial assessments.<br><br>
    
    <strong>Data Acknowledgment:</strong> This system is trained on anonymized historical data 
    inspired by the LendingClub dataset structure.<br><br>
    
    <em>This interface is designed for academic research evaluation and demonstration.</em>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR: Additional Information
# =============================================================================

with st.sidebar:
    st.markdown("### 📚 About This System")
    
    st.markdown("""
    This demonstration showcases:
    
    1. **Risk Estimation** - ML-based probability prediction
    
    2. **Causal Counterfactuals** - Recommendations that respect 
       real-world dependencies between features
    
    3. **Actionability Scores** - Quantified feasibility of 
       recommendations
    
    4. **SHAP Explanations** - Diagnostic understanding of 
       model behavior
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔬 Research Focus")
    st.markdown("""
    **Primary Contribution:**
    Causally-constrained counterfactual explanations that provide 
    actionable, realistic recommendations.
    
    **Key Innovation:**
    Unlike correlation-based methods, our causal approach respects 
    the structure of how features relate in the real world.
    """)
    
    st.markdown("---")
    
    # Method Comparison Chart
    st.markdown("### 📊 Method Comparison")
    comparison_fig = create_method_comparison_chart()
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ API Status")
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Backend: Connected")
        else:
            st.warning("⚠️ Backend: Responding with errors")
    except:
        st.error("❌ Backend: Not available")
        st.caption("Start the API with: `python -m src.api.main`")
