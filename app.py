import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import datetime
import pycountry

# ---- Option mappings: Human-readable for all dropdowns ----
EXPERIENCE_LEVELS = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior-level",
    "EX": "Executive-level"
}
EMPLOYMENT_TYPES = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance"
}
COMPANY_SIZES = {
    "S": "Small (1-50 employees)",
    "M": "Medium (50-250 employees)",
    "L": "Large (250+ employees)"
}
REMOTE_RATIOS = {
    0: "No remote (0%)",
    50: "Hybrid (50% remote)",
    100: "Fully remote (100%)"
}

# Inverse the mapping for lookup
def inv_map(mapping):
    return {v: k for k, v in mapping.items()}

INV_EXPERIENCE_LEVELS = inv_map(EXPERIENCE_LEVELS)
INV_EMPLOYMENT_TYPES = inv_map(EMPLOYMENT_TYPES)
INV_COMPANY_SIZES = inv_map(COMPANY_SIZES)
INV_REMOTE_RATIOS = inv_map(REMOTE_RATIOS)

# --- Country code mapping ---
def country_code_to_name(code):
    """Convert ISO 3166-1 alpha-2 country codes to full country names."""
    if isinstance(code, float) or code is None:
        return "Unknown"
    try:
        return pycountry.countries.get(alpha_2=code).name
    except:
        return code

def get_country_options(encoder_key, label_encoders):
    country_codes = sorted(list(label_encoders[encoder_key].classes_))
    country_names = [country_code_to_name(c) for c in country_codes]
    code_to_name = dict(zip(country_codes, country_names))
    name_to_code = dict(zip(country_names, country_codes))
    return country_names, code_to_name, name_to_code

# ---- Page and style setup ----
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
    <style>
    html, body, [class^="main"], .block-container {
        background: #131a24 !important;
        color: #f1f3f6 !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 390px !important;
        width: 390px !important;
        max-width: 500px !important;
    }
    .metric-container {
        background: #1c2533 !important;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(26, 69, 116, 0.10);
        padding: 2.2rem 2rem 1.3rem 2rem;
        text-align: center;
        margin-top: 1.3rem;
        color: #f2f4f8 !important;
    }
    .stSelectbox > div > div {
        color: #f6fafd !important;
        background: #1a2332 !important;
    }
    .st-bf {
        color: #f6fafd !important;
    }
    label, .stTextInput>div>div>input, .stSelectbox > div>div, .stTextInput>div>input, .stTextArea>div>textarea {
        color: #f6fafd !important;
        background: #232e44 !important;
    }
    .stButton>button {
        background-color: #258dfa;
        color: #f8feff;
        border-radius: 20px;
        font-size: 1.1rem;
        padding: .6rem 1.4rem;
    }
    .st-expanderHeader {
        color: #b7d2ff !important;
    }
    .stAlert, .stSidebar {
        background: #1a2332 !important;
        color: #ececec !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fffeee !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Salary Prediction App")
st.caption("Predict your data science salary anywhere in the world. Powered by AI & 2023 industry data.")

now_ist = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
st.write(f"🕒 **Current date:** {now_ist.strftime('%A, %B %d, %Y, %I:%M %p IST')}")

@st.cache_resource
def load_all_artifacts():
    required = [
        "salary_model.pkl", "preprocessor.pkl", "column_mappings.pkl", "model_info.pkl"
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    model = joblib.load("salary_model.pkl")
    encoders = joblib.load("preprocessor.pkl")
    column_mappings = joblib.load("column_mappings.pkl")
    model_info = joblib.load("model_info.pkl")
    return model, encoders, column_mappings, model_info, None

model, label_encoders, column_mappings, model_info, missing_files = load_all_artifacts()
if missing_files:
    st.error("❌ Required files missing: " + ", ".join(missing_files))
    st.stop()

def get_options(encoder_key, mapping_dict=None, default=None):
    if encoder_key in label_encoders:
        classes = list(label_encoders[encoder_key].classes_)
        return [mapping_dict.get(c, c) for c in sorted(classes)] if mapping_dict else sorted(classes)
    return default if default else []

# ---- Layout: Sidebar and Main Panel ----
# (Sidebar width is set in the custom CSS above)
side_col, main_col = st.columns([1.15, 1.85], gap="large")

with side_col:
    with st.form("job_form"):
        st.header("📝 Enter Job Details")

        job_role = st.selectbox("Job Role", get_options("job_role", None, ["Data Scientist", "ML Engineer"]))

        exp_opts = get_options("experience_level", EXPERIENCE_LEVELS)
        experience_level_display = st.selectbox("Experience Level", exp_opts)
        experience_level = INV_EXPERIENCE_LEVELS.get(experience_level_display, experience_level_display)

        emp_opts = get_options("employment_type", EMPLOYMENT_TYPES)
        employment_type_display = st.selectbox("Employment Type", emp_opts)
        employment_type = INV_EMPLOYMENT_TYPES.get(employment_type_display, employment_type_display)

        size_opts = get_options("company_size", COMPANY_SIZES)
        company_size_display = st.selectbox("Company Size", size_opts)
        company_size = INV_COMPANY_SIZES.get(company_size_display, company_size_display)

        # Countries - full name only
        company_names, code2name, name2code = get_country_options("company_location", label_encoders)
        company_location_display = st.selectbox("Company Location", company_names)
        company_location = name2code.get(company_location_display, company_location_display)

        residence_names, res_code2name, res_name2code = get_country_options("employee_residence", label_encoders)
        employee_residence_display = st.selectbox("Employee Residence", residence_names)
        employee_residence = res_name2code.get(employee_residence_display, employee_residence_display)

        remote_opts = get_options("remote_ratio", REMOTE_RATIOS)
        remote_ratio_display = st.selectbox("Remote Ratio", remote_opts)
        remote_ratio = INV_REMOTE_RATIOS.get(remote_ratio_display, int(str(remote_ratio_display).split()[-1].strip("()%")))

        work_year = st.selectbox("Work Year", get_options("work_year", None, [2023, 2022, 2021]))

        submitted = st.form_submit_button("🔮 Predict Salary")

    st.markdown("ℹ️ _All fields are required. Fields sourced from the latest Kaggle dataset._")

    with st.expander("🧾 Selected Profile (for review)"):
        st.json({
            "Job Role": job_role,
            "Experience Level": experience_level_display,
            "Employment Type": employment_type_display,
            "Company Size": company_size_display,
            "Company Location": company_location_display,
            "Employee Residence": employee_residence_display,
            "Remote Ratio": remote_ratio_display,
            "Work Year": work_year
        })

with main_col:
    st.header("📈 Prediction")
    if submitted:
        try:
            model_features = model_info["feature_names"]
            input_dict = {
                "job_role": job_role,
                "experience_level": experience_level,
                "employment_type": employment_type,
                "company_size": company_size,
                "company_location": company_location,
                "employee_residence": employee_residence,
                "remote_ratio": int(remote_ratio),
                "work_year": int(work_year)
            }

            # Encode categorical variables
            for key, val in input_dict.items():
                if key in label_encoders:
                    enc = label_encoders[key]
                    if val not in enc.classes_:
                        st.error(f"Unknown category '{val}' for {key}. Valid: {list(enc.classes_)}")
                        raise ValueError(f"Unknown value '{val}' for {key}")
                    input_dict[key] = enc.transform([val])[0]
            for f in model_features:
                if f not in input_dict:
                    input_dict[f] = 0

            input_df = pd.DataFrame([[input_dict[f] for f in model_features]], columns=model_features)
            input_df = input_df.astype(float)

            prediction = model.predict(input_df)[0]

            st.markdown(
                f"""
                <div class="metric-container">
                    <h2 style='color:#43bea8; margin-top: 0;'>💰 Predicted Salary</h2>
                    <div style="font-size:2.2rem; font-weight:600; color:#e8f7cd; margin:.1em 0;">
                        ${prediction:,.0f} <span style='font-size:1.15rem; color:#43bea8;'>USD / year</span>
                    </div>
                    <div style="color:#c9d6e2; margin-top:.7em;">Estimation for this exact role and profile.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.success("Prediction complete! Use the sidebar to learn more.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.code(str(e))
    else:
        st.info("Fill in the details and click **Predict Salary** to see your estimate.")

    st.markdown("---")
    st.caption("Built with ❤️ for global data science roles by JobanGrewal. Based on Kaggle 2023 dataset")

    with st.expander("🔍 Debug Info: Model Inputs & Encoders"):
        st.write("Model expects features:", model_info["feature_names"])
        st.write("Encoders available:", list(label_encoders.keys()))
        st.write("Column mappings:", column_mappings.get("mappings", {}))

st.markdown(
    """
    <div style='text-align:center; color:#b7b7b7; font-size:1rem; margin-top:2.2em;'>
    © 2025 Data Science Salary Predictor By JobanGrewal &nbsp;|&nbsp; All rights reserved
    </div>
    """,
    unsafe_allow_html=True,
)
