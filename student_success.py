# ============================================================
# STUDENT SUCCESS INTELLIGENCE PROJECT
# BharatCares Internship Project
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

def load_data():

    from ucimlrepo import fetch_ucirepo

    print("Loading UCI Student Performance dataset...")

    dataset = fetch_ucirepo(id=320)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    df = pd.concat([features, targets], axis=1)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------

def clean_data(df):

    df = df.copy()

    print("\nChecking missing values...")
    print("Total missing values:", df.isnull().sum().sum())

    duplicates = df.duplicated().sum()

    if duplicates > 0:
        print("Duplicate rows:", duplicates)
        df = df.drop_duplicates()

    numeric_columns = [
        "age",
        "Medu",
        "Fedu",
        "traveltime",
        "studytime",
        "failures",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
        "G1",
        "G2",
        "G3"
    ]

    for column in numeric_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ------------------------------------------------------------
# 4. CALCULATE KPIs
# ------------------------------------------------------------

def calculate_kpis(df):

    total_students = len(df)

    average_grade = df["G3"].mean()

    pass_rate = (
        (df["G3"] >= 10).mean()
        * 100
    )

    average_absences = df["absences"].mean()

    students_failed = (
        df["G3"] < 10
    ).sum()

    kpis = {
        "Total Students": total_students,
        "Average Final Grade": round(average_grade, 2),
        "Pass Rate (%)": round(pass_rate, 2),
        "Average Absences": round(average_absences, 2),
        "Students Below Pass Mark": int(students_failed)
    }

    print("\n================ KPIs ================")

    for key, value in kpis.items():
        print(f"{key}: {value}")

    return kpis


# ------------------------------------------------------------
# 5. CREATE RISK INDICATOR
# ------------------------------------------------------------

def create_risk_indicator(df):

    df = df.copy()

    absence_threshold = df["absences"].quantile(0.75)

    df["risk_score"] = (

        (df["G3"] < 10).astype(int)

        +

        (df["failures"] >= 1).astype(int)

        +

        (df["absences"] >= absence_threshold).astype(int)

        +

        (df["studytime"] == 1).astype(int)

    )

    df["risk_level"] = np.select(

        [
            df["risk_score"] >= 3,
            df["risk_score"] == 2,
            df["risk_score"] == 1
        ],

        [
            "High",
            "Medium",
            "Low"
        ],

        default="Low"
    )

    return df


# ------------------------------------------------------------
# 6. CREATE VISUALIZATIONS
# ------------------------------------------------------------

def create_visualizations(df):

    print("\nCreating visualizations...")

    # --------------------------------------------------------
    # Chart 1: Final Grade Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["G3"],
        bins=range(0, 22),
        edgecolor="black"
    )

    plt.title("Distribution of Final Grades")

    plt.xlabel("Final Grade")

    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "01_final_grade_distribution.png"
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 2: Study Time vs Grade
    # --------------------------------------------------------

    study_data = (
        df.groupby("studytime")["G3"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        study_data.index.astype(str),
        study_data.values
    )

    plt.title(
        "Average Final Grade by Study Time"
    )

    plt.xlabel(
        "Study Time Category"
    )

    plt.ylabel(
        "Average Final Grade"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "02_studytime_vs_grade.png"
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 3: Absences vs Grade
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["absences"],
        df["G3"],
        alpha=0.6
    )

    plt.title(
        "Absences vs Final Grade"
    )

    plt.xlabel(
        "Number of Absences"
    )

    plt.ylabel(
        "Final Grade"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "03_absences_vs_grade.png"
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 4: Previous Grade vs Final Grade
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["G2"],
        df["G3"],
        alpha=0.6
    )

    plt.title(
        "Previous Grade vs Final Grade"
    )

    plt.xlabel(
        "Second Period Grade (G2)"
    )

    plt.ylabel(
        "Final Grade (G3)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "04_previous_grade_vs_final_grade.png"
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 5: Previous Failures vs Grade
    # --------------------------------------------------------

    failure_data = (
        df.groupby("failures")["G3"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        failure_data.index.astype(str),
        failure_data.values
    )

    plt.title(
        "Average Final Grade by Previous Failures"
    )

    plt.xlabel(
        "Number of Previous Failures"
    )

    plt.ylabel(
        "Average Final Grade"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "05_failures_vs_grade.png"
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 6: Risk Distribution
    # --------------------------------------------------------

    risk_data = (
        df["risk_level"]
        .value_counts()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        risk_data.index,
        risk_data.values
    )

    plt.title(
        "Student Risk-Level Distribution"
    )

    plt.xlabel(
        "Risk Level"
    )

    plt.ylabel(
        "Number of Students"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "06_risk_distribution.png"
    )

    plt.close()

    print("Visualizations created successfully.")


# ------------------------------------------------------------
# 7. MACHINE LEARNING
# ------------------------------------------------------------

def train_models(df):

    print("\nTraining machine-learning models...")

    data = df.copy()

    target = "G3"

    X = data.drop(columns=[target])

    y = data[target]

    # Do not use our own risk indicator as a model input
    for column in [
        "risk_score",
        "risk_level"
    ]:

        if column in X.columns:
            X = X.drop(columns=[column])

    categorical_columns = (
        X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    )

    numerical_columns = (
        X.select_dtypes(
            exclude=["object", "category"]
        ).columns.tolist()
    )

    preprocessing = ColumnTransformer(

        transformers=[

            (
                "numeric",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    )
                ]),

                numerical_columns
            ),

            (
                "categorical",

                Pipeline([

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )

                ]),

                categorical_columns
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=250,
                random_state=42,
                max_depth=8
            )
    }

    results = []

    for model_name, model in models.items():

        pipeline = Pipeline([

            (
                "preprocessing",
                preprocessing
            ),

            (
                "model",
                model
            )

        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({

            "Model": model_name,

            "MAE": round(
                mae,
                3
            ),

            "RMSE": round(
                rmse,
                3
            ),

            "R2": round(
                r2,
                3
            )
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n================ MODEL RESULTS ================")

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# ------------------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------------------

def save_results(
    df,
    kpis,
    model_results
):

    print("\nSaving project results...")

    # KPIs

    pd.DataFrame(
        [kpis]
    ).to_csv(

        OUTPUT_DIR /
        "kpis.csv",

        index=False
    )

    # Risk analysis

    df[
        [
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2",
            "G3",
            "risk_score",
            "risk_level"
        ]
    ].to_csv(

        OUTPUT_DIR /
        "student_risk_analysis.csv",

        index=False
    )

    # Model results

    model_results.to_csv(

        OUTPUT_DIR /
        "model_results.csv",

        index=False
    )

    # Study-time summary

    (
        df.groupby("studytime")["G3"]
        .agg(
            ["count", "mean"]
        )
        .round(2)
        .to_csv(
            OUTPUT_DIR /
            "studytime_summary.csv"
        )
    )

    # Failure summary

    (
        df.groupby("failures")["G3"]
        .agg(
            ["count", "mean"]
        )
        .round(2)
        .to_csv(
            OUTPUT_DIR /
            "failures_summary.csv"
        )
    )

    print(
        "Results saved in the outputs folder."
    )


# ------------------------------------------------------------
# 9. MAIN PROGRAM
# ------------------------------------------------------------

def main():

    print(
        "\n=================================================="
    )

    print(
        "      STUDENT SUCCESS INTELLIGENCE PROJECT"
    )

    print(
        "==================================================\n"
    )

    # Load data

    df = load_data()

    # Clean data

    df = clean_data(df)

    # KPIs

    kpis = calculate_kpis(df)

    # Risk analysis

    df = create_risk_indicator(df)

    # Visualizations

    create_visualizations(df)

    # Machine learning

    model_results = train_models(df)

    # Save everything

    save_results(
        df,
        kpis,
        model_results
    )

    print(
        "\n=================================================="
    )

    print(
        "PROJECT COMPLETED SUCCESSFULLY"
    )

    print(
        "Check the 'outputs' folder for results."
    )

    print(
        "=================================================="
    )


# ------------------------------------------------------------
# RUN PROJECT
# ------------------------------------------------------------

if __name__ == "__main__":

    main()
