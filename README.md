# 🛡️ Internship Fraud Detection System

An AI-powered tool to detect potentially fraudulent internship postings and protect students from scams.

## 🎯 Features
- **Real-time Fraud Detection**: Analyzes internship details in real-time
- **Risk Scoring**: Provides a risk percentage score (0-100%)
- **Red Flag Identification**: Highlights suspicious patterns
- **Professional UI**: Blue/Black/Purple glass effect design
- **Safety Recommendations**: Offers practical advice for verification

## 🔧 How It Works
The system uses a machine learning model trained on 295 internship listings to identify patterns commonly found in fraudulent postings:

1. **Registration Fees**: Whether payment is required
2. **Contact Information**: Email domain and website validity
3. **Job Description**: Suspicious keywords and phrases
4. **Duration & Stipend**: Realistic timeframes and compensation
5. **Location**: Remote vs on-site patterns

## 🚀 Quick Start
1. **Enter internship details** in the form
2. **Click "Analyze for Fraud Risk"**
3. **View the risk score and detailed analysis**
4. **Follow the safety recommendations**

## 📊 Model Performance
- **Accuracy**: 92.4%
- **F1-Score**: 0.91
- **Precision**: 0.89
- **Recall**: 0.93

## 🛡️ Safety Tips
- ❌ Never pay money for internships
- 🔍 Verify company legitimacy through official channels
- 📞 Look for proper selection processes (interviews/tests)
- ⭐ Check employee reviews on platforms like Glassdoor
- 💭 Trust your instincts - if it seems too good to be true, it probably is

## 🎨 UI Design
- **Theme**: Professional blue/black/purple gradient
- **Effect**: Glass morphism with blur effects
- **Layout**: Clean, intuitive, mobile-responsive
- **Typography**: Modern, readable fonts

## 🏗️ Tech Stack
- **Frontend**: Gradio with custom CSS
- **ML Framework**: Scikit-learn
- **Models**: Random Forest, Logistic Regression, Decision Trees
- **Deployment**: Hugging Face Spaces

## 🔗 Links
- **Live Demo**:https://huggingface.co/spaces/PRIYA1312/internship-fraud-detector
- **Dataset**: title	company_name	email	website	description	stipend	registration_fee	duration	location	selection_process	skills_required	fraudulent
- **GitHub**:https://github.com/pri779/Internship_Fraud_Detection-InternGuard-

## 📝 License
MIT License - Free for educational and personal use

## 🙏 Acknowledgments
- Built with **Gradio** for the interactive interface
- Powered by **Scikit-learn** for machine learning
- Hosted on **Hugging Face Spaces** for deployment
- Created to protect students from internship scams

---e

**⚠️ Disclaimer**: This tool provides AI-powered guidance only. Always conduct thorough research before accepting any internship offer.
