from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
metadata = load_metadata()


def mapped_selectbox(label, options, help_text=None):
    return st.selectbox(
        label=label,
        options=list(options.keys()),
        format_func=lambda value: options[value],
        help=help_text,
    )


school_options = {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}
sex_options = {"F": "Female", "M": "Male"}
address_options = {"U": "Urban", "R": "Rural"}
family_size_options = {"LE3": "3 or fewer", "GT3": "More than 3"}
parent_status_options = {"T": "Living together", "A": "Living apart"}

education_options = {
    0: "None",
    1: "Primary education",
    2: "5th–9th grade",
    3: "Secondary education",
    4: "Higher education",
}

job_options = {
    "teacher": "Teacher",
    "health": "Healthcare",
    "services": "Civil services",
    "at_home": "At home",
    "other": "Other",
}

reason_options = {
    "home": "Close to home",
    "reputation": "School reputation",
    "course": "Course preference",
    "other": "Other",
}

guardian_options = {"mother": "Mother", "father": "Father", "other": "Other"}
yes_no_options = {"yes": "Yes", "no": "No"}

travel_time_options = {
    1: "< 15 minutes",
    2: "15–30 minutes",
    3: "30–60 minutes",
    4: "> 1 hour",
}

study_time_options = {
    1: "< 2 hours/week",
    2: "2–5 hours/week",
    3: "5–10 hours/week",
    4: "> 10 hours/week",
}

rating_options = {
    1: "1 — Very low",
    2: "2",
    3: "3 — Moderate",
    4: "4",
    5: "5 — Very high",
}


st.title("🎓 Student Performance Predictor")

st.write(
    """
    Predict a student's final Mathematics grade using demographic,
    academic, family, and school-related information.
    """
)

st.caption("The model predicts G3, the final grade on a 0–20 scale.")


with st.form("prediction_form"):

    st.subheader("Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        G1 = st.slider("First-period grade (G1)", 0, 20, 10)

    with col2:
        G2 = st.slider("Second-period grade (G2)", 0, 20, 10)

    with col3:
        failures = st.number_input(
            "Previous class failures",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        studytime = mapped_selectbox("Weekly study time", study_time_options)

    with col2:
        absences = st.number_input(
            "Number of school absences",
            min_value=0,
            max_value=93,
            value=4,
            step=1,
        )

    with col3:
        traveltime = mapped_selectbox(
            "Travel time to school",
            travel_time_options,
        )

    st.subheader("School Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = mapped_selectbox("School", school_options)

    with col2:
        reason = mapped_selectbox(
            "Reason for choosing the school",
            reason_options,
        )

    with col3:
        schoolsup = mapped_selectbox(
            "Extra educational support",
            yes_no_options,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        paid = mapped_selectbox(
            "Extra paid Mathematics classes",
            yes_no_options,
        )

    with col2:
        activities = mapped_selectbox(
            "Extracurricular activities",
            yes_no_options,
        )

    with col3:
        higher = mapped_selectbox(
            "Plans to pursue higher education",
            yes_no_options,
        )

    st.subheader("Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 15, 22, 17)

    with col2:
        sex = mapped_selectbox("Sex", sex_options)

    with col3:
        health = mapped_selectbox("Current health", rating_options)

    col1, col2, col3 = st.columns(3)

    with col1:
        freetime = mapped_selectbox("Free time after school", rating_options)

    with col2:
        goout = mapped_selectbox("Going out with friends", rating_options)

    with col3:
        romantic = mapped_selectbox(
            "Currently in a romantic relationship",
            yes_no_options,
        )

    st.subheader("Lifestyle")

    col1, col2, col3 = st.columns(3)

    with col1:
        Dalc = mapped_selectbox(
            "Workday alcohol consumption",
            rating_options,
        )

    with col2:
        Walc = mapped_selectbox(
            "Weekend alcohol consumption",
            rating_options,
        )

    with col3:
        internet = mapped_selectbox(
            "Internet access at home",
            yes_no_options,
        )

    st.subheader("Family and Home")

    col1, col2, col3 = st.columns(3)

    with col1:
        address = mapped_selectbox("Home area", address_options)

    with col2:
        famsize = mapped_selectbox("Family size", family_size_options)

    with col3:
        Pstatus = mapped_selectbox(
            "Parents' cohabitation status",
            parent_status_options,
        )

    col1, col2 = st.columns(2)

    with col1:
        Medu = mapped_selectbox("Mother's education", education_options)

    with col2:
        Fedu = mapped_selectbox("Father's education", education_options)

    col1, col2, col3 = st.columns(3)

    with col1:
        Mjob = mapped_selectbox("Mother's occupation", job_options)

    with col2:
        Fjob = mapped_selectbox("Father's occupation", job_options)

    with col3:
        guardian = mapped_selectbox("Guardian", guardian_options)

    col1, col2, col3 = st.columns(3)

    with col1:
        famsup = mapped_selectbox(
            "Family educational support",
            yes_no_options,
        )

    with col2:
        nursery = mapped_selectbox(
            "Attended nursery school",
            yes_no_options,
        )

    with col3:
        famrel = mapped_selectbox(
            "Quality of family relationships",
            rating_options,
        )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        type="primary",
        use_container_width=True,
    )


if submitted:

    input_data = pd.DataFrame(
        [
            {
                "school": school,
                "sex": sex,
                "age": age,
                "address": address,
                "famsize": famsize,
                "Pstatus": Pstatus,
                "Medu": Medu,
                "Fedu": Fedu,
                "Mjob": Mjob,
                "Fjob": Fjob,
                "reason": reason,
                "guardian": guardian,
                "traveltime": traveltime,
                "studytime": studytime,
                "failures": failures,
                "schoolsup": schoolsup,
                "famsup": famsup,
                "paid": paid,
                "activities": activities,
                "nursery": nursery,
                "higher": higher,
                "internet": internet,
                "romantic": romantic,
                "famrel": famrel,
                "freetime": freetime,
                "goout": goout,
                "Dalc": Dalc,
                "Walc": Walc,
                "health": health,
                "absences": absences,
                "G1": G1,
                "G2": G2,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))

    st.divider()
    st.subheader("Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Final Grade", f"{prediction:.1f} / 20")

    with col2:
        st.metric("Model Test MAE", f"{metadata['test_mae']:.2f}")

    with col3:
        st.metric("Model Test R²", f"{metadata['test_r2']:.2f}")

    st.info(
        f"""
        The model predicts a final grade of **{prediction:.1f}/20**.

        On the held-out test set, the model's average absolute
        prediction error was approximately
        **{metadata['test_mae']:.2f} grade points**.
        """
    )


with st.expander("About this model"):
    st.write(
        """
        This application uses a Random Forest regression model trained
        on the UCI Student Performance Mathematics dataset.

        The prediction should be treated as an educational demonstration
        of machine-learning deployment, not as a definitive assessment
        of an individual student's academic potential.
        """
    )