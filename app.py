import pickle
import streamlit as st
import pandas as pd
import numpy as np


# Load model and transformer
model = pickle.load(open("placement_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

st.title("🎓 Student Placement Prediction & Skill Gap Analysis")



# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", min_value=18, max_value=35)

degree = st.selectbox(
    "Degree",
    ["B.Tech", "B.Sc", "BCA", "M.Tech", "MCA"]
)

branch = st.selectbox(
    "Branch",
    ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    step=1
)

backlogs = st.number_input("Backlogs", min_value=0)

internships = st.number_input("Internships", min_value=0)

certifications = st.number_input("Certifications", min_value=0)

coding_skills = st.slider("Coding Skills", 0, 100)

communication_skills = st.slider("Communication Skills", 0, 100)

aptitude_score = st.slider("Aptitude Score", 0, 100)

projects = st.number_input("Projects", min_value=0)

# Predict Button
if st.button("Predict Placement"):

    input_df = pd.DataFrame(
        [[gender, age, degree, branch, cgpa,
          backlogs, internships, certifications,
          coding_skills, communication_skills,
          aptitude_score, projects]],
        columns=[
            'gender',
            'age',
            'degree',
            'branch',
            'cgpa',
            'backlogs',
            'internships',
            'certifications',
            'coding_skills',
            'communication_skills',
            'aptitude_score',
            'projects'
        ]
    )

    # Transform input
    input_transformed = transformer.transform(input_df)

    # Prediction
    prediction = model.predict(input_transformed)

    # Probability
    probability = model.predict_proba(input_transformed)

    if prediction[0] == 1:
        st.success("✅ Student is likely to be Placed")
    else:
        st.error("❌ Student is likely to be Not Placed")

    st.write(
        f"Placement Probability: {probability[0][1]*100:.2f}%"
    )

    st.subheader("Skill Gap Analysis")

    if coding_skills < 60:
        st.warning("Improve Coding Skills (Python, DSA, SQL)")

    if communication_skills < 60:
        st.warning("Improve Communication Skills")

    if aptitude_score < 60:
        st.warning("Practice Aptitude and Reasoning")

    if internships == 0:
        st.warning("Complete at least one Internship")

    if certifications < 2:
        st.warning("Earn more Certifications")

    if projects < 2:
        st.warning("Build more Academic/Industry Projects")
