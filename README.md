# Student Success Intelligence Dashboard

## BharatCares Internship Project

### Project Overview

The Student Success Intelligence Dashboard analyzes student academic-performance data to identify patterns associated with academic performance and observable risk signals.

The project follows the workflow:

**Data → KPI → Trend → Driver → Risk → Action**

The objective is to convert raw student data into meaningful insights that can support academic monitoring and intervention planning.

---

## Problem Statement

Educational institutions collect information about students' study habits, previous grades, attendance, family background, school-related factors and other characteristics.

This project analyzes these variables to answer:

- What is the overall academic performance of students?
- How is study time associated with final grades?
- How are absences associated with academic performance?
- How are previous failures associated with final grades?
- What observable signals can indicate academic risk?
- How well can machine-learning models predict final grades?

---

## Dataset

The project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

Dataset source:

https://archive.ics.uci.edu/dataset/320/student+performance

The dataset contains student demographic, social, school-related and academic variables. UCI provides separate datasets for Mathematics and Portuguese language performance. This project uses the Portuguese-language student-performance data retrieved using the `ucimlrepo` package. 0

Dataset citation:

> Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- UCI ML Repository
- GitHub

---

## Project Workflow

### 1. Data Collection

The dataset is automatically retrieved from the UCI Machine Learning Repository using the `ucimlrepo` Python package.

### 2. Data Cleaning

The project:

- Checks missing values
- Checks duplicate records
- Converts relevant columns to numeric format
- Prepares the dataset for analysis and machine learning

### 3. KPI Analysis

The following KPIs are calculated:

- Total number of students
- Average final grade
- Pass rate
- Average absences
- Number of students below the project pass threshold

### 4. Exploratory Data Analysis

The project analyzes:

- Final grade distribution
- Study time vs final grade
- Absences vs final grade
- Previous grade vs final grade
- Previous failures vs final grade
- Student risk-level distribution

### 5. Risk Analysis

A project-defined analytical risk indicator is created using observable signals such as:

- Final grade below the project pass threshold
- Previous class failures
- High absence count
- Very low study-time category

This indicator is intended for analytical exploration and is not a diagnosis or causal classification of an individual student.

### 6. Machine Learning

Two regression models are compared:

- Linear Regression
- Random Forest Regressor

The models are evaluated using:

- MAE
- RMSE
