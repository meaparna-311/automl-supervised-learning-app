import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import joblib
import re

import sweetviz as sv
from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Machine Learning Automation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM THEME


st.markdown("""
<style>

    .stApp {
        background-color: #0B1020;
        color: #F8FAFC;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #26324A;
    }

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #94A3B8;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .info-card {
        background-color: #151D32;
        border: 1px solid #26324A;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 15px;
    }

    .success-card {
        background-color: #132A25;
        border: 1px solid #245C4F;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
    }

    .stButton > button {
        background-color: #6366F1;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
    }

    .stButton > button:hover {
        background-color: #7C3AED;
        color: white;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #26324A;
        border-radius: 10px;
    }
    hr {
        border-color: #26314A;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    .metric-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        min-height: 110px;
    }

    .metric-label {
        font-size: 15px;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #111827;
    }

</style>
""", unsafe_allow_html=True)



# SESSION STATE


defaults = {
    "data": None,
    "file_name": None,
    "problem_type": "Classification",
    "target_column": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "encoder": None,
    "scaler": None,
    "selected_model": None,
    "selected_model_name": None,
    "predictions": None,
    "best_model": None,
    "best_params": None,
    "grid_search": None,
    "cv_scores": None,
    "cv_mean": None,
    "cv_std": None,
    "comparison_results": None,
    "comparison_best_model": None,
    "comparison_best_score": None,
    "profiling_report": None,
    "sweetviz_report": None,
    "best_model_object": None,
    "comparison_models": {},
    "model_package": None,
    "loaded_model_package": None,
    "loaded_model_name": None,
    "numerical_medians": {},
    "categorical_modes": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value



# SIDEBAR — DATASET SETUP


with st.sidebar:

    st.markdown("## 🤖 ML Automation")
    st.markdown("---")

    st.markdown("### 1. Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        key="dataset_uploader"
    )

    if uploaded_file is not None:

        # Only reload when a different file is uploaded.
        if st.session_state.file_name != uploaded_file.name:

            try:

                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)

                elif uploaded_file.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)

                else:
                    st.error("Unsupported file format.")
                    df = None

                if df is not None:
                    st.session_state.data = df
                    st.session_state.file_name = uploaded_file.name

                    
                    st.session_state.target_column = None
                    
                    st.session_state.raw_X_train = None
                    st.session_state.raw_X_test = None
                    st.session_state.raw_y_train = None
                    st.session_state.raw_y_test = None
                    st.session_state.imputed_X_train = None
                    st.session_state.imputed_X_test = None
                    st.session_state.X_train_num = None
                    st.session_state.X_test_num = None
                    st.session_state.X_train_cat = None
                    st.session_state.X_test_cat = None
                    st.session_state.X_train = None
                    st.session_state.X_test = None
                    st.session_state.y_train = None
                    st.session_state.y_test = None
                    st.session_state.encoder = None
                    st.session_state.scaler = None
                    st.session_state.selected_model = None
                    st.session_state.selected_model_name = None
                    st.session_state.predictions = None
                    st.session_state.best_model = None
                    st.session_state.best_params = None
                    st.session_state.grid_search = None
                    st.session_state.cv_scores = None
                    st.session_state.cv_mean = None
                    st.session_state.cv_std = None
                    st.session_state.comparison_results = None
                    st.session_state.comparison_best_model = None
                    st.session_state.comparison_best_score = None
                    st.session_state.profiling_report = None
                    st.session_state.sweetviz_report = None
                    st.session_state.best_model_object = None
                    st.session_state.comparison_models = {}
                    st.session_state.model_package = None
                    st.session_state.loaded_model_package = None
                    st.session_state.loaded_model_name = None
                    st.session_state.numerical_medians = {}
                    st.session_state.categorical_modes = {}

                    st.success("Dataset loaded successfully!")

            except Exception as e:
                            st.error(f"Error reading file: {e}")

    st.markdown("### 2. Problem Type")

    problem_type = st.selectbox(
        "Select problem type",
        ["Classification", "Regression"],
        index=0,
        key="problem_type_select"
    )

if st.session_state.problem_type != problem_type:
    st.session_state.comparison_results = None
    st.session_state.comparison_models = {}
    st.session_state.comparison_best_model = None
    st.session_state.comparison_best_score = None
    st.session_state.best_model_object = None
    st.session_state.model_package = None

st.session_state.problem_type = problem_type

if st.session_state.data is not None:

        st.markdown("---")
        st.markdown("### 3. Target Column")

        columns = st.session_state.data.columns.tolist()

        # If the previous target is no longer present, use the first column.
        previous_target = st.session_state.target_column
        default_index = (
            columns.index(previous_target)
            if previous_target in columns
            else 0
        )

        target_column = st.selectbox(
            "Select target column",
            columns,
            index=default_index,
            key="target_column_select"
        )

        st.session_state.target_column = target_column

    else:
        st.session_state.target_column = None

    st.markdown("---")
    st.caption("Machine Learning Automation Tool")
    st.caption("Dataset Setup")


# MAIN TITLE


st.markdown(
    '<div class="main-title">🤖 Machine Learning Automation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Automated machine learning workflow from dataset upload to model comparison</div>',
    unsafe_allow_html=True
)


# NO DATASET


if st.session_state.data is None:

    st.markdown("""
    <div class="info-card">
        <h3>👋 Welcome to ML Automation</h3>
        <p>
            Upload a CSV or Excel (.xlsx) dataset from the sidebar.
            Then select the problem type and target column.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.stop()



# DATASET LOADED


df = st.session_state.data
target = st.session_state.target_column
problem_type = st.session_state.problem_type

st.markdown(
    f"""
    <div class="success-card">
        <b>✓ Dataset loaded successfully</b><br>
        File: {st.session_state.file_name}
    </div>
    """,
    unsafe_allow_html=True
)



# MAIN WORKFLOW TABS


tabs = st.tabs([
    "📄 Dataset Preview",
    "⚙️ Current Configuration",
    "📊 Dataset Overview",
    "🔍 EDA",
    "🛠️ Data Preparation",
    "✂️ Train-Test Split",
    "🩹 Missing Value Imputation",
    "🔤 Categorical Encoding",
    "📏 Feature Scaling",
    "🤖 Model Selection & Training",
    "🎯 Hyperparameter Tuning",
    "🔄 Cross-Validation",
    "📈 Model Comparison",
    "💾 Save & Predict"
])



# TAB 1 — DATASET PREVIEW


with tabs[0]:

    st.markdown(
        '<div class="section-title">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Showing the first 10 rows of **{st.session_state.file_name}**."
    )

    st.dataframe(
        df.head(10),
        
    )

    st.markdown("### Dataset Shape")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")



# TAB 2 — CURRENT CONFIGURATION


with tabs[1]:

    st.markdown(
        '<div class="section-title">Current Configuration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-card">
            <b>File:</b> {st.session_state.file_name}<br><br>
            <b>Problem Type:</b> {st.session_state.problem_type}<br><br>
            <b>Target Column:</b> {target}<br><br>
            <b>Dataset Size:</b>
            {df.shape[0]:,} rows × {df.shape[1]:,} columns
        </div>
        """,
        unsafe_allow_html=True
    )

    


# TAB 3 — DATASET OVERVIEW


with tabs[2]:

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Rows</div>
            <div class="metric-value">{df.shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📋 Columns</div>
            <div class="metric-value">{df.shape[1]:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Missing Values</div>
            <div class="metric-value">{int(df.isnull().sum().sum()):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔄 Duplicate Rows</div>
            <div class="metric-value">{int(df.duplicated().sum()):,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Target information
    if target is not None:

        st.markdown("### 🎯 Target Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Target Column", target)

        with col2:
            st.metric("Data Type", str(df[target].dtype))

        with col3:
            st.metric("Unique Values", df[target].nunique())

        if problem_type == "Classification":

            st.markdown("#### Target Class Distribution")

            target_counts = (
                df[target]
                .value_counts()
                .reset_index()
            )

            target_counts.columns = [target, "Count"]

            st.dataframe(
                target_counts,
                use_container_width=True
            )

        else:

            st.markdown("#### Target Statistics")

            st.dataframe(
                df[target].describe(),
                use_container_width=True
            )

    st.markdown("### 📋 Column Information")

    overview = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        overview,
        
    )


# TAB 4 — EDA WITH INNER TABS


with tabs[3]:

    st.markdown(
        '<div class="section-title">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    eda_tabs = st.tabs([
        "Overview",
        "Statistics",
        "Missing Values",
        "Correlation",
        "Distributions",
        "Target Analysis",
        "Pandas Profiling",
        "Sweetviz"
    ])

   
    # EDA 1 — OVERVIEW
 

    with eda_tabs[0]:

        st.subheader("Dataset Information")

        overview = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Missing": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            overview,
            use_container_width=True
        )

        st.subheader("First 10 Rows")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    # EDA 2 — STATISTICS
  

    with eda_tabs[1]:

        st.subheader("Descriptive Statistics")

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] > 0:

            st.dataframe(
                numeric_df.describe().T,
                use_container_width=True
            )

        else:

            st.info("No numerical columns found.")

   
    # EDA 3 — MISSING VALUES
 

    with eda_tabs[2]:

        st.subheader("Missing Value Analysis")

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:

            st.success("✓ No missing values found.")

        else:

            missing_table = pd.DataFrame({
                "Column": missing.index,
                "Missing Values": missing.values,
                
            })

            st.dataframe(
                missing_table,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.barplot(
                x=missing.values,
                y=missing.index,
                ax=ax
            )

            ax.set_xlabel("Number of Missing Values")
            ax.set_ylabel("Column")
            ax.set_title("Missing Values by Column")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # EDA 4 — CORRELATION
  

    with eda_tabs[3]:

        st.subheader("Correlation Heatmap")

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:

            st.info(
                "At least two numerical columns are required "
                "for correlation analysis."
            )

        else:

            correlation = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(12, 8))

            sns.heatmap(
                correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )

            ax.set_title("Feature Correlation Matrix")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Strongest Correlations")

            corr_pairs = (
                correlation
                .where(
                    np.triu(
                        np.ones(correlation.shape),
                        k=1
                    ).astype(bool)
                )
                .stack()
                .reset_index()
            )

            corr_pairs.columns = [
                "Feature 1",
                "Feature 2",
                "Correlation"
            ]

            corr_pairs["Absolute Correlation"] = (
                corr_pairs["Correlation"].abs()
            )

            corr_pairs = corr_pairs.sort_values(
                "Absolute Correlation",
                ascending=False
            )

            st.dataframe(
                corr_pairs.head(10),
                use_container_width=True
            )

   
    # EDA 5 — DISTRIBUTIONS
   

    with eda_tabs[4]:

        st.subheader("Numerical Features")

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) == 0:

            st.info("No numerical columns found.")

        else:

            selected_numeric = st.selectbox(
                "Select a numerical feature",
                numeric_columns,
                key="eda_numeric_feature"
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.histplot(
                df[selected_numeric].dropna(),
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {selected_numeric}"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Boxplot")

            fig, ax = plt.subplots(figsize=(10, 4))

            sns.boxplot(
                x=df[selected_numeric],
                ax=ax
            )

            ax.set_title(
                f"Boxplot of {selected_numeric}"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.subheader("Categorical Features")

        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if len(categorical_columns) == 0:

            st.info("No categorical columns found.")

        else:

            selected_categorical = st.selectbox(
                "Select a categorical feature",
                categorical_columns,
                key="eda_categorical_feature"
            )

            counts = (
                df[selected_categorical]
                .value_counts()
                .head(15)
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.barplot(
                x=counts.values,
                y=counts.index,
                ax=ax
            )

            ax.set_xlabel("Count")
            ax.set_ylabel(selected_categorical)
            ax.set_title(
                f"Distribution of {selected_categorical}"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

   
    # EDA 6 — TARGET ANALYSIS
   

    with eda_tabs[5]:

        st.subheader("Target Variable Analysis")

        if target is None:

            st.warning(
                "Please select a target column from the sidebar."
            )

        else:

            st.write(f"Target: **{target}**")

            if problem_type == "Classification":

                target_counts = (
                    df[target]
                    .value_counts()
                    .head(20)
                )

                fig, ax = plt.subplots(figsize=(10, 5))

                sns.barplot(
                    x=target_counts.values,
                    y=target_counts.index,
                    ax=ax,
                    errorbar=None
                )

                ax.set_xlabel("Number of Records")
                ax.set_ylabel("Class")
                ax.set_title("Target Class Distribution")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(
                    target_counts.reset_index(name="Count"),
                    use_container_width=True
                )

            else:

                if pd.api.types.is_numeric_dtype(df[target]):

                    fig, ax = plt.subplots(figsize=(10, 5))

                    sns.histplot(
                        df[target].dropna(),
                        kde=True,
                        ax=ax
                    )

                    ax.set_title(
                        f"Distribution of Target: {target}"
                    )

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(
                        df[target]
                        .describe()
                        .to_frame()
                        .T,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "Regression requires a numerical target."
                    )


    # EDA 7 — PANDAS PROFILING
    with eda_tabs[6]:
        st.subheader("📋 Pandas Profiling Report")

        if st.button("Generate Report", key="profiling_button"):
            with st.spinner("Generating report..."):
                st.session_state.profiling_report = ProfileReport(
                    df, title="Pandas Profiling Report"
                ).to_html()

        if st.session_state.profiling_report:
            components.html(
                st.session_state.profiling_report,
                height=900,
                scrolling=True
            )

    # EDA 8 — SWEETVIZ
    with eda_tabs[7]:
        st.subheader("🍭 Sweetviz Report")

        if st.button("Generate Report", key="sweetviz_button"):
            with st.spinner("Generating report..."):
                report = sv.analyze(df, target_feat=target) if target else sv.analyze(df)
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                temp.close()
                report.show_html(temp.name, open_browser=False)

                with open(temp.name, encoding="utf-8") as f:
                    st.session_state.sweetviz_report = f.read()
                os.unlink(temp.name)

        if st.session_state.sweetviz_report:
            components.html(
                st.session_state.sweetviz_report,
                height=900,
                scrolling=True
            )


#finding input and target features

X = df.drop(columns=[target])
y = df[target]

if y.isnull().sum() > 0:
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

numerical_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()



# TAB 5 — DATA PREPARATION


with tabs[4]:

    st.markdown(
        '<div class="section-title">🛠️ Data Preparation</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**Features:** {X.shape[1]}  |  "
        f"**Target:** {target}"
    )

    if y.isnull().sum() > 0:
        st.warning(
            f"{df[target].isnull().sum()} rows have missing target "
            "values and are excluded from model preparation."
        )

    if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(y):

        st.error(
            "For Regression, the target column must contain numeric values."
        )

    st.subheader("🔍 Feature Identification")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Numerical Features")
        st.write(numerical_columns)
        st.metric("Numerical Feature Count", len(numerical_columns))

    with col2:

        st.write("### Categorical Features")
        st.write(categorical_columns)
        st.metric("Categorical Feature Count", len(categorical_columns))



# TAB 6 — TRAIN-TEST SPLIT


with tabs[5]:

    st.markdown(
        '<div class="section-title">✂️ Train-Test Split</div>',
        unsafe_allow_html=True
    )

    test_size = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="test_size_slider"
    )

    st.write(
        f"Training size: **{100 - test_size}%**  |  "
        f"Testing size: **{test_size}%**"
    )

    if st.button(
        "✂️ Create Train-Test Split",
        key="split_data_button"
    ):

        try:

            if problem_type == "Classification":

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size / 100,
                    random_state=42,
                    stratify=y
                )

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size / 100,
                    random_state=42
                )

        except ValueError:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=42
            )

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        

        st.success("Train-test split created successfully.")

    if (
        st.session_state.X_train is not None
    ):

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Training Rows", X_train.shape[0])

        with col2:
            st.metric("Testing Rows", X_test.shape[0])

        st.write("### Training Features")
        st.dataframe(X_train.head(10), use_container_width=True)

        st.write("### Testing Features")
        st.dataframe(X_test.head(10), use_container_width=True)



if (
    st.session_state.X_train is None
    or list(st.session_state.X_train.columns) != list(X.columns)
):

    try:

        if problem_type == "Classification":

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

        else:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

    except ValueError:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test


X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test


# TAB 7 — MISSING VALUE IMPUTATION


with tabs[6]:

    st.markdown(
        '<div class="section-title">🩹 Missing Value Imputation</div>',
        unsafe_allow_html=True
    )
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    

    numerical_medians = {}
    categorical_modes = {}

    for column in numerical_columns:

        median_value = X_train[column].median()
        numerical_medians[column] = median_value

        X_train.loc[:, column] = (
            X_train[column].fillna(median_value)
        )

        X_test.loc[:, column] = (
            X_test[column].fillna(median_value)
        )

    for column in categorical_columns:

        mode_value = X_train[column].mode()

        if len(mode_value) > 0:

            mode_value = mode_value.iloc[0]
            categorical_modes[column] = mode_value

            X_train.loc[:, column] = (
                X_train[column].fillna(mode_value)
            )

            X_test.loc[:, column] = (
                X_test[column].fillna(mode_value)
            )

    st.session_state.numerical_medians = numerical_medians
    st.session_state.categorical_modes = categorical_modes

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test

    remaining_train_missing = int(X_train.isnull().sum().sum())
    remaining_test_missing = int(X_test.isnull().sum().sum())

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Remaining Training Missing Values",
            remaining_train_missing
        )

    with col2:
        st.metric(
            "Remaining Testing Missing Values",
            remaining_test_missing
        )

    st.success("Missing values handled successfully.")

    st.write("### Imputation Method")
    st.write("• Numerical features → Median")
    st.write("• Categorical features → Mode")

    st.write("### Imputed Training Data")
    st.dataframe(
        X_train.head(10),
        use_container_width=True
    )


# TAB 8 — CATEGORICAL ENCODING


with tabs[7]:

    st.markdown(
        '<div class="section-title">🔤 Categorical Encoding</div>',
        unsafe_allow_html=True
    )

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test

    if len(categorical_columns) > 0:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        X_train_cat = encoder.fit_transform(
            X_train[categorical_columns]
        )

        X_test_cat = encoder.transform(
            X_test[categorical_columns]
        )

        encoded_columns = encoder.get_feature_names_out(
            categorical_columns
        )

        X_train_cat = pd.DataFrame(
            X_train_cat,
            columns=encoded_columns,
            index=X_train.index
        )

        X_test_cat = pd.DataFrame(
            X_test_cat,
            columns=encoded_columns,
            index=X_test.index
        )

        st.session_state.encoder = encoder
        st.session_state.X_train_cat = X_train_cat
        st.session_state.X_test_cat = X_test_cat

        st.success("Categorical features encoded successfully.")

        st.metric(
            "Encoded Categorical Features",
            X_train_cat.shape[1]
        )

        st.dataframe(
            X_train_cat.head(10),
            use_container_width=True
        )

    else:

        st.session_state.encoder = None

        st.session_state.X_train_cat = pd.DataFrame(
            index=X_train.index
        )

        st.session_state.X_test_cat = pd.DataFrame(
            index=X_test.index
        )

        st.info("No categorical features found. Encoding was not required.")



# TAB 9 — FEATURE SCALING


with tabs[8]:

    st.markdown(
        '<div class="section-title">📏 Feature Scaling</div>',
        unsafe_allow_html=True
    )

    scaling_method = st.selectbox(
        "Choose Scaling Method",
        [
            "Standardization (StandardScaler)",
            "Normalization (MinMaxScaler)",
            "No Scaling"
        ],
        key="scaling_method_select"
    )

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test

    if len(numerical_columns) > 0:

        if scaling_method == "Standardization (StandardScaler)":

            scaler = StandardScaler()

            X_train_num = pd.DataFrame(
                scaler.fit_transform(
                    X_train[numerical_columns]
                ),
                columns=numerical_columns,
                index=X_train.index
            )

            X_test_num = pd.DataFrame(
                scaler.transform(
                    X_test[numerical_columns]
                ),
                columns=numerical_columns,
                index=X_test.index
            )

        elif scaling_method == "Normalization (MinMaxScaler)":

            scaler = MinMaxScaler()

            X_train_num = pd.DataFrame(
                scaler.fit_transform(
                    X_train[numerical_columns]
                ),
                columns=numerical_columns,
                index=X_train.index
            )

            X_test_num = pd.DataFrame(
                scaler.transform(
                    X_test[numerical_columns]
                ),
                columns=numerical_columns,
                index=X_test.index
            )

        else:

            scaler = None

            X_train_num = X_train[numerical_columns].copy()
            X_test_num = X_test[numerical_columns].copy()

    else:

        scaler = None

        X_train_num = pd.DataFrame(index=X_train.index)
        X_test_num = pd.DataFrame(index=X_test.index)

    st.session_state.scaler = scaler
    st.session_state.X_train_num = X_train_num
    st.session_state.X_test_num = X_test_num

    st.success(f"Scaling option applied: {scaling_method}")

    st.write("### Scaled Numerical Features")

    st.dataframe(
        X_train_num.head(10),
        use_container_width=True
    )



# CREATE FINAL PROCESSED DATA


X_train_num = st.session_state.X_train_num
X_test_num = st.session_state.X_test_num
X_train_cat = st.session_state.X_train_cat
X_test_cat = st.session_state.X_test_cat

X_train_processed = pd.concat(
    [
        X_train_num.reset_index(drop=True),
        X_train_cat.reset_index(drop=True)
    ],
    axis=1
)

X_test_processed = pd.concat(
    [
        X_test_num.reset_index(drop=True),
        X_test_cat.reset_index(drop=True)
    ],
    axis=1
)

st.session_state.X_train = X_train_processed
st.session_state.X_test = X_test_processed
st.session_state.y_train = st.session_state.y_train.reset_index(drop=True)
st.session_state.y_test = st.session_state.y_test.reset_index(drop=True)



# TAB 10 — MODEL SELECTION & TRAINING


with tabs[9]:

    st.markdown(
        '<div class="section-title">🤖 Model Selection & Training</div>',
        unsafe_allow_html=True
    )

    if problem_type == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier()
        }

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor()
        }

    st.write(f"**Problem Type:** {problem_type}")

    st.write("### Available Models")

    selected_model_name = st.selectbox(
        "Select a model",
        list(models.keys()),
        key="model_selection"
    )

    selected_model = models[selected_model_name]

    st.info(
        f"Selected Model: **{selected_model_name}**"
    )

    if st.button(
        "🚀 Train Model",
        key="train_model_button"
    ):

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        with st.spinner("Training model..."):

            selected_model.fit(
                X_train_processed,
                y_train
            )

            predictions = selected_model.predict(
                X_test_processed
            )

        st.session_state.selected_model = selected_model
        st.session_state.selected_model_name = selected_model_name
        st.session_state.predictions = predictions

        st.success(
            f"{selected_model_name} trained successfully!"
        )

        st.write("### Prediction Preview")

        preview = pd.DataFrame({
            "Actual": y_test.head(10),
            "Predicted": predictions[:10]
        })

        st.dataframe(
            preview,
            use_container_width=True
        )



# TAB 11 — HYPERPARAMETER TUNING


with tabs[10]:

    st.markdown(
        '<div class="section-title">🎯 Hyperparameter Tuning</div>',
        unsafe_allow_html=True
    )

    if st.session_state.selected_model is None:

        st.info(
            "Please train/select a model in the Model Selection & Training tab first."
        )

    else:

        model_name = st.session_state.selected_model_name
        model = st.session_state.selected_model

        st.write(f"**Selected Model:** {model_name}")

        if model_name == "Logistic Regression":

            param_grid = {
                "C": [0.1, 1, 10],
                "max_iter": [1000, 2000]
            }

        elif model_name == "Decision Tree":

            param_grid = {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }

        elif model_name == "Random Forest":

            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }

        elif model_name == "SVM" or model_name == "SVR":

            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }

        elif model_name == "KNN":

            param_grid = {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            }

        elif model_name == "Linear Regression":

            param_grid = {
                "fit_intercept": [True, False]
            }

        else:
            param_grid = {}

        st.write("### Parameters to Test")
        st.json(param_grid)

        if st.button(
            "🔍 Find Best Parameters",
            key="hyperparameter_tuning_button"
        ):

            X_train = st.session_state.X_train
            y_train = st.session_state.y_train

            with st.spinner(
                "Testing different parameter combinations..."
            ):

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring=(
                        "accuracy"
                        if problem_type == "Classification"
                        else "r2"
                    ),
                    n_jobs=-1
                )

                grid_search.fit(
                    X_train_processed,
                    y_train
                )

            st.session_state.best_model = grid_search.best_estimator_
            st.session_state.best_params = grid_search.best_params_
            st.session_state.grid_search = grid_search

            st.success("Hyperparameter tuning completed!")

            st.write("### 🏆 Best Parameters")

            st.json(
                grid_search.best_params_
            )

            st.metric(
                "Best Cross-Validation Score",
                f"{grid_search.best_score_:.4f}"
            )



# TAB 12 — 5-FOLD CROSS-VALIDATION


with tabs[11]:

    st.markdown(
        '<div class="section-title">🔄 5-Fold Cross-Validation</div>',
        unsafe_allow_html=True
    )

    if st.session_state.best_model is None:

        st.info(
            "Please complete hyperparameter tuning first."
        )

    else:

        best_model = st.session_state.best_model
        model_name = st.session_state.selected_model_name

        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        st.write(f"**Model:** {model_name}")

        if problem_type == "Classification":

            scoring_metric = "accuracy"
            metric_name = "Accuracy"

        else:

            scoring_metric = "r2"
            metric_name = "R² Score"

        if st.button(
            "🔄 Run 5-Fold Cross-Validation",
            key="cross_validation_button"
        ):

            with st.spinner(
                "Running 5-fold cross-validation..."
            ):

                cv_scores = cross_val_score(
                    best_model,
                    X_train_processed,
                    y_train,
                    cv=5,
                    scoring=scoring_metric
                )

            st.session_state.cv_scores = cv_scores
            st.session_state.cv_mean = cv_scores.mean()
            st.session_state.cv_std = cv_scores.std()

            st.success(
                "5-fold cross-validation completed!"
            )

            results = pd.DataFrame({
                "Fold": [
                    "Fold 1",
                    "Fold 2",
                    "Fold 3",
                    "Fold 4",
                    "Fold 5"
                ],
                metric_name: cv_scores
            })

            results[metric_name] = results[metric_name].round(4)

            st.dataframe(
                results,
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    f"Average {metric_name}",
                    f"{cv_scores.mean():.4f}"
                )

            with col2:

                st.metric(
                    "Standard Deviation",
                    f"{cv_scores.std():.4f}"
                )

            st.info(
                f"Average {metric_name}: {cv_scores.mean():.4f}"
            )

            st.write(
                "A lower standard deviation means the model's "
                "performance is more consistent across the 5 folds."
            )



# TAB 13 — MODEL COMPARISON


with tabs[12]:

    st.markdown(
        '<div class="section-title">📊 Model Comparison</div>',
        unsafe_allow_html=True
    )

    X_train = X_train_processed
    X_test = X_test_processed
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    if problem_type == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier()
        }

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor()
        }

    if st.button(
        "📊 Compare All Models",
        key="compare_all_models_button"
    ):

        results = []

        with st.spinner(
            "Training and comparing all models..."
        ):

            for name, model in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                # Store the fitted object. Do not recreate a new unfitted model
                # when the Streamlit script reruns.
                st.session_state.comparison_models[name] = model

                train_predictions = model.predict(
                    X_train
                )

                test_predictions = model.predict(
                    X_test
                )

                if problem_type == "Classification":

                    train_accuracy = accuracy_score(
                        y_train,
                        train_predictions
                    )

                    train_precision = precision_score(
                        y_train,
                        train_predictions,
                        average="weighted",
                        zero_division=0
                    )

                    train_recall = recall_score(
                        y_train,
                        train_predictions,
                        average="weighted",
                        zero_division=0
                    )

                    train_f1 = f1_score(
                        y_train,
                        train_predictions,
                        average="weighted",
                        zero_division=0
                    )

                    test_accuracy = accuracy_score(
                        y_test,
                        test_predictions
                    )

                    test_precision = precision_score(
                        y_test,
                        test_predictions,
                        average="weighted",
                        zero_division=0
                    )

                    test_recall = recall_score(
                        y_test,
                        test_predictions,
                        average="weighted",
                        zero_division=0
                    )

                    test_f1 = f1_score(
                        y_test,
                        test_predictions,
                        average="weighted",
                        zero_division=0
                    )

                    results.append({
                        "Model": name,
                        "Train Accuracy": train_accuracy,
                        "Test Accuracy": test_accuracy,
                        "Train Precision": train_precision,
                        "Test Precision": test_precision,
                        "Train Recall": train_recall,
                        "Test Recall": test_recall,
                        "Train F1": train_f1,
                        "Test F1": test_f1
                    })

                else:

                    train_r2 = r2_score(
                        y_train,
                        train_predictions
                    )

                    train_mae = mean_absolute_error(
                        y_train,
                        train_predictions
                    )

                    train_rmse = np.sqrt(
                        mean_squared_error(
                            y_train,
                            train_predictions
                        )
                    )

                    test_r2 = r2_score(
                        y_test,
                        test_predictions
                    )

                    test_mae = mean_absolute_error(
                        y_test,
                        test_predictions
                    )

                    test_rmse = np.sqrt(
                        mean_squared_error(
                            y_test,
                            test_predictions
                        )
                    )

                    results.append({
                        "Model": name,
                        "Train R²": train_r2,
                        "Test R²": test_r2,
                        "Train MAE": train_mae,
                        "Test MAE": test_mae,
                        "Train RMSE": train_rmse,
                        "Test RMSE": test_rmse
                    })

        comparison_df = pd.DataFrame(results)

        st.session_state.comparison_results = comparison_df

        st.success(
            "All models have been trained and compared!"
        )


   
    # DISPLAY STORED RESULTS
    

    if st.session_state.comparison_results is not None:

        comparison_df = st.session_state.comparison_results

        st.subheader("📋 Model Performance")

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True
        )

        # BEST MODEL
        

        if problem_type == "Classification":

            best_index = comparison_df[
                "Test F1"
            ].idxmax()

            best_model_name = comparison_df.loc[
                best_index,
                "Model"
            ]

            best_score = comparison_df.loc[
                best_index,
                "Test F1"
            ]

            st.session_state.comparison_best_model = best_model_name
            st.session_state.comparison_best_score = best_score

            # Store the actual trained model object for saving and prediction.
            st.session_state.best_model_object = st.session_state.comparison_models[best_model_name]

            st.success(
                f"🏆 Best Model: {best_model_name}"
            )

            st.metric(
                "Best Test F1 Score",
                f"{best_score:.4f}"
            )

        else:

            best_index = comparison_df[
                "Test R²"
            ].idxmax()

            best_model_name = comparison_df.loc[
                best_index,
                "Model"
            ]

            best_score = comparison_df.loc[
                best_index,
                "Test R²"
            ]

            st.session_state.comparison_best_model = best_model_name
            st.session_state.comparison_best_score = best_score

            # Store the actual trained model object for saving and prediction.
            st.session_state.best_model_object = st.session_state.comparison_models[best_model_name]

            st.success(
                f"🏆 Best Model: {best_model_name}"
            )

            st.metric(
                "Best Test R²",
                f"{best_score:.4f}"
            )

       
        # OVERFITTING INFORMATION
       

        st.subheader("🔎 Training vs Testing Performance")

        if problem_type == "Classification":

            comparison_df["Accuracy Gap"] = (
                comparison_df["Train Accuracy"]
                - comparison_df["Test Accuracy"]
            )

            st.dataframe(
                comparison_df[
                    [
                        "Model",
                        "Train Accuracy",
                        "Test Accuracy",
                        "Accuracy Gap"
                    ]
                ].round(4),
                use_container_width=True
            )

            st.info(
                "A large difference between training and testing "
                "performance may indicate overfitting."
            )

        else:

            comparison_df["R² Gap"] = (
                comparison_df["Train R²"]
                - comparison_df["Test R²"]
            )

            st.dataframe(
                comparison_df[
                    [
                        "Model",
                        "Train R²",
                        "Test R²",
                        "R² Gap"
                    ]
                ].round(4),
                use_container_width=True
            )

            st.info(
                "A large difference between training and testing "
                "R² may indicate overfitting."
            )


# =========================================================
# TAB 14 — SAVE BEST MODEL & MAKE PREDICTIONS
# =========================================================

with tabs[13]:

    st.markdown(
        '<div class="section-title">💾 Save Best Model & Make Prediction</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # CREATE LOCAL STORAGE FOLDERS
    # -----------------------------------------------------
    models_folder = "saved_models"
    predictions_folder = "prediction_history"

    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(predictions_folder, exist_ok=True)

    st.info(
        "The saved Joblib file contains the model, encoding, scaling, "
        "feature information, target information, and missing-value "
        "imputation information needed for future predictions."
    )

    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------
    st.subheader("💾 Save Best Model")

    if st.session_state.best_model_object is not None:

        best_model_object = st.session_state.best_model_object
        best_model_name = st.session_state.comparison_best_model

        # Create a safe filename from the uploaded dataset name.
        base_dataset_name = os.path.splitext(
            st.session_state.file_name
        )[0]

        safe_dataset_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            base_dataset_name
        ).strip("_")

        if not safe_dataset_name:
            safe_dataset_name = "dataset"

        model_filename = (
            f"{safe_dataset_name}_model.joblib"
        )

        model_path = os.path.join(
            models_folder,
            model_filename
        )

        # Save category options so the same choices can be shown
        # when the model is loaded later.
        categorical_options = {}

        for column in categorical_columns:

            categorical_options[column] = (
                df[column]
                .dropna()
                .unique()
                .tolist()
            )

        model_package = {
            "model": best_model_object,
            "model_name": best_model_name,
            "encoder": st.session_state.encoder,
            "scaler": st.session_state.scaler,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "target_column": target,
            "problem_type": problem_type,
            "numerical_medians": st.session_state.numerical_medians,
            "categorical_modes": st.session_state.categorical_modes,
            "categorical_options": categorical_options,
            "feature_columns": X.columns.tolist(),
            "processed_feature_columns": (
                X_train_processed.columns.tolist()
            ),
            "dataset_name": st.session_state.file_name
        }

        st.session_state.model_package = model_package

        if st.button(
            "💾 Save Best Model as Joblib",
            key="save_best_model_button"
        ):

            try:

                joblib.dump(
                    model_package,
                    model_path
                )

                st.success(
                    f"Best model saved successfully: "
                    f"{model_filename}"
                )

                with open(model_path, "rb") as model_file:

                    st.download_button(
                        "⬇️ Download Joblib File",
                        data=model_file.read(),
                        file_name=model_filename,
                        mime="application/octet-stream",
                        key="download_joblib_button"
                    )

            except Exception as e:

                st.error(
                    f"Could not save the model: {e}"
                )

        st.write(
            f"**Best Model:** {best_model_name}"
        )

        st.write(
            f"**Target:** {target}"
        )

        st.write(
            f"**Problem Type:** {problem_type}"
        )

    else:

        st.warning(
            "Please run Model Comparison first. "
            "The best model will then be available to save."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # LOAD A PREVIOUSLY SAVED MODEL
    # -----------------------------------------------------
    st.subheader("📂 Load Saved Model")

    saved_model_files = sorted(
        [
            file
            for file in os.listdir(models_folder)
            if file.lower().endswith(".joblib")
        ]
    )

    if len(saved_model_files) == 0:

        st.info(
            "No saved Joblib models found yet."
        )

    else:

        selected_saved_model = st.selectbox(
            "Select a saved model",
            saved_model_files,
            key="saved_model_selector"
        )

        if st.button(
            "📂 Load Selected Model",
            key="load_saved_model_button"
        ):

            selected_model_path = os.path.join(
                models_folder,
                selected_saved_model
            )

            try:

                loaded_package = joblib.load(
                    selected_model_path
                )

                st.session_state.loaded_model_package = (
                    loaded_package
                )

                st.session_state.loaded_model_name = (
                    selected_saved_model
                )

                st.success(
                    f"Loaded successfully: "
                    f"{selected_saved_model}"
                )

            except Exception as e:

                st.error(
                    f"Could not load the saved model: {e}"
                )

    # -----------------------------------------------------
    # SELECT WHICH MODEL TO USE FOR PREDICTION
    # -----------------------------------------------------
    active_package = None

    if st.session_state.loaded_model_package is not None:

        active_package = (
            st.session_state.loaded_model_package
        )

        st.success(
            f"Using loaded model: "
            f"{st.session_state.loaded_model_name}"
        )

    elif st.session_state.model_package is not None:

        active_package = st.session_state.model_package

        st.info(
            "Using the best model from the current dataset. "
            "Save it as Joblib if you want to use it later."
        )

    # -----------------------------------------------------
    # MAKE NEW PREDICTION
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 Make a New Prediction")

    if active_package is None:

        st.info(
            "First compare models and save the best model, "
            "or load a previously saved Joblib model."
        )

    else:

        prediction_model = active_package["model"]
        prediction_encoder = active_package["encoder"]
        prediction_scaler = active_package["scaler"]

        prediction_numerical_columns = (
            active_package["numerical_columns"]
        )

        prediction_categorical_columns = (
            active_package["categorical_columns"]
        )

        prediction_target = (
            active_package["target_column"]
        )

        prediction_problem_type = (
            active_package["problem_type"]
        )

        prediction_numerical_medians = (
            active_package.get(
                "numerical_medians",
                {}
            )
        )

        prediction_categorical_modes = (
            active_package.get(
                "categorical_modes",
                {}
            )
        )

        prediction_categorical_options = (
            active_package.get(
                "categorical_options",
                {}
            )
        )

        st.write(
            f"**Problem Type:** {prediction_problem_type}"
        )

        st.write(
            f"**Target:** {prediction_target}"
        )

        st.write(
            "Enter values for the features below."
        )

        prediction_input = {}

        # Create input fields automatically for numerical columns.
        if len(prediction_numerical_columns) > 0:

            st.write("### 🔢 Numerical Features")

            numerical_input_columns = st.columns(2)

            for i, column in enumerate(
                prediction_numerical_columns
            ):

                default_value = (
                    prediction_numerical_medians.get(
                        column,
                        0
                    )
                )

                try:
                    default_value = float(default_value)
                except Exception:
                    default_value = 0.0

                with numerical_input_columns[
                    i % 2
                ]:

                    prediction_input[column] = (
                        st.number_input(
                            column,
                            value=default_value,
                            key=(
                                "prediction_num_"
                                + re.sub(
                                    r"[^A-Za-z0-9_]+",
                                    "_",
                                    str(column)
                                )
                            )
                        )
                    )

        # Create dropdowns automatically for categorical columns.
        if len(prediction_categorical_columns) > 0:

            st.write("### 🔤 Categorical Features")

            categorical_input_columns = st.columns(2)

            for i, column in enumerate(
                prediction_categorical_columns
            ):

                options = prediction_categorical_options.get(
                    column,
                    []
                )

                if len(options) == 0:

                    st.warning(
                        f"No categorical values were saved "
                        f"for {column}."
                    )

                    continue

                default_mode = (
                    prediction_categorical_modes.get(
                        column,
                        options[0]
                    )
                )

                if default_mode not in options:

                    default_mode = options[0]

                default_index = options.index(
                    default_mode
                )

                with categorical_input_columns[
                    i % 2
                ]:

                    prediction_input[column] = (
                        st.selectbox(
                            column,
                            options,
                            index=default_index,
                            key=(
                                "prediction_cat_"
                                + re.sub(
                                    r"[^A-Za-z0-9_]+",
                                    "_",
                                    str(column)
                                )
                            )
                        )
                    )

        if st.button(
            "🔮 Predict",
            key="make_prediction_button"
        ):

            try:

                # -----------------------------------------
                # CREATE NEW DATAFRAME
                # -----------------------------------------
                new_data = pd.DataFrame(
                    [prediction_input]
                )

                # -----------------------------------------
                # HANDLE NUMERICAL MISSING VALUES
                # -----------------------------------------
                for column in prediction_numerical_columns:

                    if pd.isnull(
                        new_data.loc[0, column]
                    ):

                        new_data.loc[0, column] = (
                            prediction_numerical_medians.get(
                                column,
                                0
                            )
                        )

                # -----------------------------------------
                # HANDLE CATEGORICAL MISSING VALUES
                # -----------------------------------------
                for column in prediction_categorical_columns:

                    if pd.isnull(
                        new_data.loc[0, column]
                    ):

                        new_data.loc[0, column] = (
                            prediction_categorical_modes.get(
                                column,
                                ""
                            )
                        )

                # -----------------------------------------
                # SCALE NUMERICAL FEATURES
                # -----------------------------------------
                new_num = pd.DataFrame(
                    index=[0]
                )

                if len(
                    prediction_numerical_columns
                ) > 0:

                    if prediction_scaler is not None:

                        new_num = pd.DataFrame(
                            prediction_scaler.transform(
                                new_data[
                                    prediction_numerical_columns
                                ]
                            ),
                            columns=(
                                prediction_numerical_columns
                            )
                        )

                    else:

                        new_num = new_data[
                            prediction_numerical_columns
                        ].copy()

                # -----------------------------------------
                # ENCODE CATEGORICAL FEATURES
                # -----------------------------------------
                new_cat = pd.DataFrame(
                    index=[0]
                )

                if (
                    len(prediction_categorical_columns) > 0
                    and prediction_encoder is not None
                ):

                    encoded_values = (
                        prediction_encoder.transform(
                            new_data[
                                prediction_categorical_columns
                            ]
                        )
                    )

                    encoded_columns = (
                        prediction_encoder
                        .get_feature_names_out(
                            prediction_categorical_columns
                        )
                    )

                    new_cat = pd.DataFrame(
                        encoded_values,
                        columns=encoded_columns
                    )

                # -----------------------------------------
                # COMBINE PROCESSED FEATURES
                # -----------------------------------------
                new_processed = pd.concat(
                    [
                        new_num.reset_index(drop=True),
                        new_cat.reset_index(drop=True)
                    ],
                    axis=1
                )

                # Make sure columns are in exactly the same
                # order used during model training.
                processed_feature_columns = (
                    active_package.get(
                        "processed_feature_columns"
                    )
                )

                if (
                    processed_feature_columns
                    is not None
                ):

                    for column in processed_feature_columns:

                        if column not in new_processed.columns:

                            new_processed[column] = 0

                    new_processed = new_processed[
                        processed_feature_columns
                    ]

                # -----------------------------------------
                # PREDICT
                # -----------------------------------------
                prediction = prediction_model.predict(
                    new_processed
                )

                prediction_value = prediction[0]

                st.success(
                    "Prediction completed successfully!"
                )

                if prediction_problem_type == "Regression":

                    st.metric(
                        "Predicted Value",
                        f"{prediction_value:,.2f}"
                    )

                else:

                    st.metric(
                        "Predicted Class",
                        str(prediction_value)
                    )

                # -----------------------------------------
                # SAVE PREDICTION HISTORY
                # -----------------------------------------
                prediction_dataset_name = (
                    active_package.get(
                        "dataset_name",
                        "dataset"
                    )
                )

                prediction_base_name = os.path.splitext(
                    prediction_dataset_name
                )[0]

                safe_prediction_name = re.sub(
                    r"[^A-Za-z0-9_-]+",
                    "_",
                    prediction_base_name
                ).strip("_")

                if not safe_prediction_name:

                    safe_prediction_name = "dataset"

                history_filename = (
                    f"{safe_prediction_name}"
                    "_predictions.csv"
                )

                history_path = os.path.join(
                    predictions_folder,
                    history_filename
                )

                prediction_record = prediction_input.copy()

                prediction_record[
                    "Predicted Value"
                ] = prediction_value

                prediction_record[
                    "Model"
                ] = active_package.get(
                    "model_name",
                    "Saved Model"
                )

                prediction_record[
                    "Problem Type"
                ] = prediction_problem_type

                prediction_record[
                    "Target Column"
                ] = prediction_target

                prediction_record[
                    "Prediction Time"
                ] = pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                prediction_record_df = pd.DataFrame(
                    [prediction_record]
                )

                if os.path.exists(
                    history_path
                ):

                    old_history = pd.read_csv(
                        history_path
                    )

                    updated_history = pd.concat(
                        [
                            old_history,
                            prediction_record_df
                        ],
                        ignore_index=True
                    )

                else:

                    updated_history = (
                        prediction_record_df
                    )

                updated_history.to_csv(
                    history_path,
                    index=False
                )

                st.success(
                    f"Prediction saved to "
                    f"{history_filename}"
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

    # -----------------------------------------------------
    # VIEW PREDICTION HISTORY
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("📜 Prediction History")

    history_files = sorted(
        [
            file
            for file in os.listdir(
                predictions_folder
            )
            if file.lower().endswith(".csv")
        ]
    )

    if len(history_files) == 0:

        st.info(
            "No prediction history is available yet."
        )

    else:

        selected_history_file = st.selectbox(
            "Select prediction history",
            history_files,
            key="history_file_selector"
        )

        selected_history_path = os.path.join(
            predictions_folder,
            selected_history_file
        )

        try:

            history_df = pd.read_csv(
                selected_history_path
            )

            st.dataframe(
                history_df,
                use_container_width=True
            )

            with open(
                selected_history_path,
                "rb"
            ) as history_file:

                st.download_button(
                    "⬇️ Download Prediction History",
                    data=history_file.read(),
                    file_name=selected_history_file,
                    mime="text/csv",
                    key="download_history_button"
                )

        except Exception as e:

            st.error(
                f"Could not read prediction history: {e}"
            )
