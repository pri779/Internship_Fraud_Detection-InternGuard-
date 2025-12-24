import gradio as gr
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# Load model
model = joblib.load("internship_fraud_detector.pkl")
scaler = joblib.load("feature_scaler.pkl")

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Preprocessing function
def preprocess_input(stipend, duration, location, registration_fee, email, website, description):
    features_dict = {}
    
    # Registration fee flag
    features_dict["has_registration_fee"] = 1 if registration_fee == "Yes" else 0
    
    # Stipend numeric
    try:
        features_dict["stipend_numeric"] = float(stipend)
    except:
        features_dict["stipend_numeric"] = 0
    
    # Duration in months
    if "week" in str(duration).lower():
        weeks = int("".join(filter(str.isdigit, str(duration))) or 0)
        features_dict["duration_months"] = weeks / 4
    elif "month" in str(duration).lower():
        months = int("".join(filter(str.isdigit, str(duration))) or 0)
        features_dict["duration_months"] = months
    else:
        features_dict["duration_months"] = 0
    
    # Remote flag
    features_dict["is_remote"] = 1 if "remote" in str(location).lower() else 0
    
    # Valid website flag
    features_dict["has_valid_website"] = 0 if str(website).lower() in ["", "not provided", "none"] else 1
    
    # Suspicious keywords count
    desc_lower = str(description).lower()
    keywords = ["pay", "fee", "registration", "deposit", "urgent", 
                "limited seats", "certificate", "guaranteed", "no interview"]
    features_dict["suspicious_keywords"] = sum(1 for keyword in keywords if keyword in desc_lower)
    
    # Email type
    email_lower = str(email).lower()
    if any(domain in email_lower for domain in ["gmail", "yahoo", "hotmail", "outlook", "rediff"]):
        email_type = "free_email"
    elif "@" in email_lower:
        email_type = "professional_email"
    else:
        email_type = "no_email"
    
    # Create DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # Add email type as one-hot encoding
    for email_cat in ["email_type_free_email", "email_type_professional_email"]:
        if email_cat.replace("email_type_", "") == email_type:
            input_df[email_cat] = 1
        else:
            input_df[email_cat] = 0
    
    # Ensure all model features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder columns
    input_df = input_df[feature_names]
    
    return input_df

# Prediction function
def predict_fraud(stipend, duration, location, registration_fee, email, website, description):
    try:
        # Preprocess
        input_df = preprocess_input(stipend, duration, location, registration_fee, email, website, description)
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100
        
        # Generate report
        risk_factors = []
        
        if registration_fee == "Yes":
            risk_factors.append("⚠️ Registration fee required")
        
        if "remote" in str(location).lower():
            risk_factors.append("📍 Remote internship")
        
        if any(domain in str(email).lower() for domain in ["gmail", "yahoo", "hotmail"]):
            risk_factors.append("📧 Free email domain")
        
        if str(website).lower() in ["", "not provided", "none"]:
            risk_factors.append("🌐 No website provided")
        
        suspicious_desc = str(description).lower()
        if any(word in suspicious_desc for word in ["urgent", "limited seats", "hurry"]):
            risk_factors.append("⏰ Urgent hiring mentioned")
        
        if any(word in suspicious_desc for word in ["certificate guaranteed", "no interview"]):
            risk_factors.append("📜 Certificate/no interview mentioned")
        
        if prediction == 1:
            result = "🛑 HIGH RISK: Potentially Fraudulent"
            advice = "❗ RECOMMENDATIONS: Verify company legitimacy, never pay upfront fees, check for proper selection process"
        else:
            result = "✅ LOW RISK: Likely Legitimate"
            advice = "✅ RECOMMENDATIONS: Still verify company details, check reviews, ensure proper selection process"
        
        risk_factors_text = "".join([f"• {factor}\n" for factor in risk_factors]) if risk_factors else "• No major risk factors identified"
        
        report = f"""
## {result}

### Risk Score: {probability:.1f}%

### Identified Risk Factors:
{risk_factors_text}

### {advice}
"""
        
        return report, probability
        
    except Exception as e:
        return f"Error: {str(e)}", 0

# Custom CSS - Blue/Black/Purple Glass Effect
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

.contain {
    background: rgba(15, 12, 41, 0.85) !important;
    backdrop-filter: blur(20px);
    border-radius: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    margin: 20px;
    padding: 25px;
}

.gr-box {
    background: rgba(30, 27, 75, 0.7) !important;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 15px !important;
}

.gr-form {
    background: rgba(40, 37, 95, 0.6) !important;
}

h1, h2, h3 {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
    margin-bottom: 20px !important;
}

button {
    background: linear-gradient(90deg, #4a43b5, #764ba2) !important;
    border: none !important;
    color: white !important;
    font-weight: 600;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

button:hover {
    background: linear-gradient(90deg, #5a53c5, #865bb2) !important;
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(118, 75, 162, 0.5) !important;
}

label {
    color: #a0a0ff !important;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 8px !important;
}

input, textarea, select {
    background: rgba(50, 47, 115, 0.8) !important;
    color: #ffffff !important;
    border: 1px solid #4a43b5 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.gr-markdown {
    background: rgba(20, 17, 55, 0.9) !important;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
}

.gr-slider {
    background: rgba(40, 37, 95, 0.8) !important;
}

.gr-slider .range {
    background: linear-gradient(90deg, #4a43b5, #764ba2) !important;
}
"""

# Create interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    # Header
    gr.Markdown("""
    # 🛡️ Internship Fraud Detection System
    ### *Protecting Students from Fake Internships*
    """)
    
    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Enter Internship Details")
            
            stipend = gr.Number(
                label="Monthly Stipend (₹)",
                value=15000,
                minimum=0,
                maximum=100000,
                step=1000
            )
            
            duration = gr.Dropdown(
                label="Duration",
                choices=["1 week", "2 weeks", "1 month", "2 months", "3 months", 
                        "4 months", "5 months", "6 months", "Full Time"],
                value="3 months"
            )
            
            location = gr.Dropdown(
                label="Location",
                choices=["Remote", "Bangalore", "Delhi", "Mumbai", "Hyderabad", 
                        "Chennai", "Pune", "Gurgaon", "Hybrid"],
                value="Remote"
            )
            
            registration_fee = gr.Radio(
                label="Registration/Enrollment Fee Required?",
                choices=["Yes", "No"],
                value="No"
            )
            
            email = gr.Textbox(
                label="Company Contact Email",
                placeholder="hr@company.com",
                value="hr@company.com"
            )
            
            website = gr.Textbox(
                label="Company Website",
                placeholder="https://www.company.com",
                value="https://www.company.com"
            )
            
            description = gr.Textbox(
                label="Internship Description",
                placeholder="Paste the complete internship description here...",
                lines=5,
                value="Work on real projects with senior developers. Learn cutting-edge technologies. Interview process includes technical assessment."
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze for Fraud Risk",
                variant="primary",
                size="lg"
            )
        
        # Output Column
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Fraud Analysis Results")
            
            result_output = gr.Markdown(
                label="Analysis Report",
                value="**👈 Enter internship details on the left and click 'Analyze'**"
            )
            
            risk_meter = gr.Slider(
                label="Risk Score (%)",
                minimum=0,
                maximum=100,
                value=0,
                interactive=False,
                elem_id="risk_slider"
            )
            
            with gr.Accordion("📊 What do these results mean?", open=False):
                gr.Markdown("""
                **Risk Score Interpretation:**
                - **0-30%**: Low Risk - Likely legitimate
                - **31-60%**: Medium Risk - Exercise caution
                - **61-100%**: High Risk - Potentially fraudulent
                
                **Common Fraud Indicators:**
                1. Registration/enrollment fees
                2. Free email domains (Gmail, Yahoo, Hotmail)
                3. No proper company website
                4. "Urgent hiring" or "Limited seats"
                5. "Certificate guaranteed" or "No interview required"
                6. Unrealistically high stipends for short durations
                """)
            
            with gr.Accordion("🛡️ Safety Checklist", open=True):
                gr.Markdown("""
                **Always Remember:**
                ✅ Research the company on LinkedIn and Google
                ✅ Verify email domain matches company website
                ✅ Never pay money for internships
                ✅ Look for proper selection process (interviews/tests)
                ✅ Check employee reviews on Glassdoor
                ✅ Contact via official company channels
                ✅ Trust your instincts
                """)
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #8888aa; font-size: 12px; padding: 20px;">
    <p>🔒 This tool uses machine learning to detect potential internship fraud. It analyzes patterns from 295+ internship listings.</p>
    <p>⚠️ This is an AI-powered guidance tool only. Always conduct thorough due diligence.</p>
    <p>Made with ❤️ to protect students from scams</p>
    </div>
    """)
    
    # Connect button to function
    analyze_btn.click(
        fn=predict_fraud,
        inputs=[stipend, duration, location, registration_fee, email, website, description],
        outputs=[result_output, risk_meter]
    )

if __name__ == "__main__":
    demo.launch()
