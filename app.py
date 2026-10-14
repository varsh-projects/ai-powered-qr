import streamlit as st
from model import evaluate_url

# Page configuration
st.set_page_config(
    page_title="AI Malicious URL Detector", 
    page_icon="🛡️", 
    layout="centered"
)

# App Header
st.title("🛡️ AI-Powered Malicious & Phishing URL Detection System")
st.markdown("Protecting users against online threats, credential theft, and scams in real-time.")

st.markdown("---")

# User Input Section
url_input = st.text_input("🔗 Enter Website URL to Scan:", placeholder="https://example.com/login")

if st.button("Scan URL Now", type="primary"):
    if url_input.strip() == "":
        st.warning("Please enter a valid URL first.")
    else:
        with st.spinner("Analyzing URL features and running AI risk model..."):
            result = evaluate_url(url_input)
            
        st.markdown("### Scan Results")
        
        # Display color-coded verdict banner
        status = result["status"]
        score = result["risk_score"]
        
        if status == "Malicious":
            st.error(f"🚨 **Verdict: MALICIOUS** (Risk Score: {score}%)")
            st.markdown("**Recommended Action:** Do not enter any credentials or personal info. Close this tab immediately.")
        elif status == "Suspicious":
            st.warning(f"⚠️ **Verdict: SUSPICIOUS** (Risk Score: {score}%)")
            st.markdown("**Recommended Action:** Proceed with extreme caution. Verify the sender or official domain.")
        else:
            st.success(f"✅ **Verdict: SAFE** (Risk Score: {score}%)")
            st.markdown("**Recommended Action:** This link appears legitimate based on current lexical heuristics.")
            
        # Display Risk Factor Breakdown
        st.markdown("#### 🔍 Detected Risk Indicators:")
        for reason in result["reasons"]:
            st.write(reason)
            
        # Visual Risk Meter
        st.markdown("#### 📊 Threat Probability Meter")
        st.progress(score / 100)

st.markdown("---")
st.caption("Hackathon Project | Built for Round 2 Live Demonstration")