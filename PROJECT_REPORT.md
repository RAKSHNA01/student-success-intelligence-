# Student Success Intelligence Dashboard

## Analyzing Factors Associated with Student Academic Performance

## 1. Executive Summary

This project analyzes student academic-performance data to identify patterns associated with academic performance and academic-risk indicators.

The project follows the Business Intelligence workflow:

**Data → KPI → Trend → Driver → Risk → Action**

The objective is to transform raw student data into meaningful insights that can support academic monitoring and intervention planning.

## 2. Problem Statement

Educational institutions collect information about students' study habits, previous grades, attendance, family background, school-related factors and other characteristics.

This project analyzes these variables to understand:

- Overall student academic performance
- Relationship between study time and final grades
- Relationship between absences and academic performance
- Relationship between previous failures and final grades
- Factors associated with academic risk
- Performance of machine-learning models in predicting final grades

## 3. Dataset

The project uses the Student Performance Dataset from the UCI Machine Learning Repository.

**Dataset:** Student Performance  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 320

The dataset contains demographic, social, school-related and academic information about students from two Portuguese schools.

The Portuguese-language student-performance dataset is used for this project.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- UCI ML Repository
- GitHub

## 5. Project Workflow

### Data Collection

The dataset is retrieved from the UCI Machine Learning Repository using the `ucimlrepo` Python package.

### Data Cleaning

The project checks:

- Missing values
- Duplicate records
- Data types
- Numeric variables

### KPI Analysis

The following KPIs are calculated:

- Total students
- Average final grade
- Pass rate
- Average absences
- Students below the pass threshold

### Exploratory Data Analysis

The project analyzes:

- Final grade distribution
- Study time vs final grade
- Absences vs final grade
- Previous grade vs final grade
- Previous failures vs final grade
- Student risk distribution

### Risk Analysis

An analytical risk indicator is created using observable signals such as:

- Final grade below the project pass threshold
- Previous failures
- High absence count
- Very low study-time category

### Machine Learning

Two regression models are compared:

- Linear Regression
- Random Forest Regressor

The models are evaluated using:

- MAE
- RMSE
- R²

## 6. Key Performance Indicators

| KPI | Result |
|---|---:|
| Total Students | 649 |
| Average Final Grade | 11.91 / 20 |
| Pass Rate | 82.1% |
| Average Absences | 4.9 |
| Students Below Pass Mark | 116 |

## 7. Exploratory Data Analysis

### Final Grade Distribution

The analysis examines the distribution of students' final grades and identifies the proportion of students performing below the project pass threshold.

### Study Time vs Final Grade

The analysis examines the relationship between reported weekly study time and final academic performance.

Higher study-time categories are associated with higher average grades in the analyzed dataset.

### Absences vs Final Grade

The analysis examines whether absence levels are associated with academic performance.

Higher absence levels are associated with weaker academic outcomes in the analyzed dataset.

### Previous Failures vs Final Grade

The analysis compares academic performance between students with and without previous failures.

Students with previous failures show lower average final grades in the analyzed dataset.

### Previous Grades vs Final Grade

Previous-period grades are analyzed against final grades to identify their relationship with overall academic performance.

## 8. Risk Analysis

The project uses an analytical risk indicator based on observable student characteristics.

| Risk Level | Students |
|---|---:|
| Low | 438 |
| Medium | 151 |
| High | 60 |

The risk indicator is an analytical classification created for this project and is not an official academic-risk assessment.

## 9. Machine Learning Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.42 | 2.05 | 0.78 |
| Random Forest | 1.21 | 1.79 | 0.83 |

## 10. Key Insights and Actions

| Finding | Insight | Possible Action |
|---|---|---|
| Study time | Study habits are associated with academic performance | Introduce structured study-planning and mentoring |
| Absences | Higher absence levels are associated with weaker outcomes | Implement attendance monitoring and early support |
| Previous failures | Previous academic difficulty is associated with later performance | Provide tutoring and academic mentoring |
| Previous grades | Previous grades are strongly associated with final performance | Use academic progress monitoring for early support |

## 11. Business Value

The project demonstrates how an institution can transform:

**Raw Data → KPIs → Patterns → Risk Signals → Actionable Insights**

The objective is to move beyond visualization and connect data analysis with practical academic-support decisions.

## 12. Limitations

- The dataset represents students from two Portuguese schools and should not automatically be generalized to all student populations.
- Observed relationships do not establish causation.
- The project-defined risk indicator is not an official academic-risk classification.
- Machine-learning performance depends on the selected variables, model configuration and train/test split.
- Previous-period grades are strongly related to the final grade, which can make prediction easier but may reduce usefulness for very early intervention.

## 13. Future Improvements

- Interactive Streamlit dashboard
- Advanced machine-learning models
- Hyperparameter tuning
- Feature-importance analysis
- Early-intervention prediction
- Interactive student-risk filtering
- Dashboard deployment
- Automated academic-support recommendations

## 14. Conclusion

The Student Success Intelligence Dashboard demonstrates how educational data can be transformed into structured business-intelligence insights.

The project combines data cleaning, KPI analysis, exploratory data analysis, risk analysis, machine learning and visualization to identify patterns associated with student academic performance.

The overall approach follows:

**Data → KPI → Trend → Driver → Risk → Action**

This provides a structured way to transform student data into insights that can support academic monitoring and institutional decision-making.

## 15. How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
