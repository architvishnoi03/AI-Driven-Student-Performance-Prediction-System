import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt



# Path to models folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load trained models
linear_model = joblib.load(os.path.join(MODEL_DIR, "linear_model.pkl"))
logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))

# Load preprocessing objects
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

# Load dataset for visualization

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "dataset", "new.csv"))

data = load_data()

# Create Performance column
def categorize(score):
    if score < 50:
        return "Poor"
    elif score < 70:
        return "Average"
    elif score < 85:
        return "Good"
    else:
        return "Excellent"

data["Performance"] = data["Final_Exam_Score"].apply(categorize)


st.set_page_config(
    page_title="AI Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Predict",
        "Visualization",
        "Model Performance",
        "About",
        "Developer",
        
    ]
)

# ---------------- HOME ----------------

if page == "Home":

    st.title("🎓 AI-Driven Student Performance Prediction System")

    st.markdown("---")

    st.subheader("Project Objective")

    st.write("""
This project predicts:

- 📈 Final Exam Score using Linear Regression
- 🏆 Student Performance Category using Logistic Regression
- 📉 Dimensionality Reduction using PCA

The system is built using Machine Learning techniques and provides
instant predictions based on student academic and personal information.
""")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Size", "500,000")
    col2.metric("Algorithms", "3")
    col3.metric("Features", "46")


# ---------------- PREDICT ----------------


elif page == "Predict":

    st.title("🔮 Predict Student Performance")

    st.write("Enter the student's details below.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=15, max_value=35, value=20)

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        region = st.selectbox(
            "Region Type",
            ["Urban", "Semi-Urban", "Rural"]
        )

        family_size = st.number_input(
            "Family Size",
            min_value=1,
            max_value=15,
            value=4
        )

        major = st.selectbox(
            "Major Subject",
            [
                "Computer Science",
                "Mechanical",
                "Civil",
                "Electronics",
                "Business"
            ]
        )

        parent_education = st.selectbox(
            "Parent Education",
            [
                "High School",
                "Graduate",
                "Post Graduate",
                "PhD"
            ]
        )

        income = st.selectbox(
            "Family Income Level",
            [
                "Low",
                "Lower-Middle",
                "Middle",
                "Upper-Middle",
                "High"
            ]
        )

    with col2:

        internet = st.slider(
            "Internet Quality",
            1,
            10,
            7
        )

        study_space = st.slider(
            "Study Space Quality",
            1,
            10,
            7
        )

        previous_gpa = st.number_input(
            "Previous GPA",
            min_value=0.0,
            max_value=10.0,
            value=7.5
        )

        failed_courses = st.number_input(
            "Failed Courses",
            0,
            20,
            0
        )

        credits = st.number_input(
            "Total Credits Earned",
            0,
            250,
            120
        )

        study_hours = st.slider(
            "Weekly Study Hours",
            0,
            60,
            20
        )

        attendance = st.slider(
            "Attendance Rate (%)",
            0,
            100,
            80
        )

        library = st.slider(
            "Library Visits / Month",
            0,
            30,
            5
        )

        extracurricular = st.slider(
            "Extracurricular Hours",
            0,
            30,
            5
        )

        sleep = st.slider(
            "Sleep Hours",
            3,
            12,
            7
        )

        social_media = st.slider(
            "Social Media Usage (Hours)",
            0,
            12,
            3
        )

        stress = st.slider(
            "Stress Level",
            1,
            10,
            5
        )

        motivation = st.slider(
            "Motivation Score",
            1,
            10,
            7
        )

        efficacy = st.slider(
            "Self-Efficacy Score",
            1,
            10,
            7
        )

        midterm = st.number_input(
            "Midterm Marks",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    
    predict_button = st.button(
        "🚀 Predict Student Performance",
        use_container_width=True
)


    if predict_button:

        # Create input dictionary
        input_data = {
            "Age": age,
            "Gender": gender,
            "Region_Type": region,
            "Family_Size": family_size,
            "Major_Subject": major,
            "Parent_Education": parent_education,
            "Family_Income_Level": income,
            "Internet_Quality": internet,
            "Study_Space_Quality": study_space,
            "Previous_GPA": previous_gpa,
            "Number_of_Failed_Courses": failed_courses,
            "Total_Credits_Earned": credits,
            "Weekly_Study_Hours": study_hours,
            "Attendance_Rate": attendance,
            "Library_Visits_Per_Month": library,
            "Extracurricular_Hours": extracurricular,
            "Sleep_Hours": sleep,
            "Social_Media_Usage_Hours": social_media,
            "Stress_Level": stress,
            "Motivation_Score": motivation,
            "Self_Efficacy_Score": efficacy,
            "Midterm_Mark": midterm
        }

        # Convert dictionary into DataFrame
        input_df = pd.DataFrame([input_data])

        st.subheader("Input Data")

        st.dataframe(input_df)

        # Convert categorical columns into dummy variables
        input_df = pd.get_dummies(input_df)

        # Add missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Arrange columns in the same order as training
        input_df = input_df[feature_columns]

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Apply PCA
        input_pca = pca.transform(input_scaled)

        # Predict Final Exam Score
        predicted_score = linear_model.predict(input_df)[0]

        # Predict Performance Category
        predicted_class = logistic_model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        # Display Results
        st.success("Prediction Completed!")

        st.subheader("Prediction Result")

        st.metric(
            "Predicted Final Exam Score",
            f"{predicted_score:.2f}"
        )

        st.metric(
            "Predicted Performance",
            predicted_label
        )

    
# ---------------- VISUALIZATION ----------------

elif page == "Visualization":

    st.title("📊 Data Visualization")

    chart = st.selectbox(
        "Select Visualization",
        [
            "Performance Distribution",
            "Exam Score Distribution",
            "Correlation Heatmap"
        ]
    )

    if chart == "Performance Distribution":

        fig, ax = plt.subplots(figsize=(8,5))

        data["Performance"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Student Performance Distribution")
        ax.set_xlabel("Performance")
        ax.set_ylabel("Students")

        st.pyplot(fig)

    elif chart == "Exam Score Distribution":

        fig, ax = plt.subplots(figsize=(8,5))

        ax.hist(data["Final_Exam_Score"], bins=20)

        ax.set_title("Final Exam Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Students")

        st.pyplot(fig)

    elif chart == "Correlation Heatmap":

        fig, ax = plt.subplots(figsize=(12,8))

        corr = data.select_dtypes(include="number").corr()

        im = ax.imshow(corr)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)

        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        plt.colorbar(im)

        st.pyplot(fig)


# ---------------- MODEL ----------------

elif page == "Model Performance":

    st.title("📊 Model Performance")

    performance = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Logistic Regression",
                "Linear Regression + PCA"
            ],

            "Metric": [
                "R² Score",
                "Accuracy",
                "R² Score"
            ],

            "Value": [
                "0.5573",
                "74.14%",
                "0.5477"
            ]
        }
    )

    st.dataframe(performance, use_container_width=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Linear Regression R²", "0.5573")

    with col2:
        st.metric("Logistic Accuracy", "74.14%")

    with col3:
        st.metric("PCA R²", "0.5477")

    st.markdown("---")

    st.success("""
    ✔ Linear Regression predicts the Final Exam Score.

    ✔ Logistic Regression predicts the Performance Category.

    ✔ PCA reduces the dimensionality while retaining approximately 95% of the variance.
    """)

# ---------------- ABOUT ----------------

elif page == "About":

    st.title("ℹ About Project")

    st.subheader("Project Name")

    st.write("AI-Driven Student Performance Prediction System")

    st.subheader("Project Objective")

    st.write("""
The objective of this project is to predict a student's Final Exam Score
and Performance Category using Machine Learning algorithms.
""")

    st.subheader("Technologies Used")

    st.write("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
""")

    st.subheader("Machine Learning Algorithms")

    st.write("""
- Linear Regression
- Logistic Regression
- Principal Component Analysis (PCA)
""")

    st.subheader("Dataset")

    st.write("""
Student Academic Performance Dataset
(500,000 records)
""")

    st.subheader("Developer")

    st.write("""
Name: Archit Vishnoi

Branch: Computer Science & Engineering

Project: AI-Driven Student Performance Prediction System
""")

# -----------developer----------------

elif page == "Developer":

    st.title(" Developer")

    st.write("### Name")
    st.write("Archit Vishnoi")

    st.write("### Project")
    st.write("AI-Driven Student Performance Prediction System")

    st.write("### Technologies")
    st.write("""
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
""")
    

   