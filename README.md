# ⚡ Energy Consumption Prediction

**A Machine Learning Project for Household Energy Consumption Forecasting**

![Python](https://img.shields.io/badge/Python-0b1730?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-0b1730?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0b1730?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-0b1730?style=for-the-badge&logo=streamlit&logoColor=white)

[🇸🇦 Read in Arabic](README_AR.md)

---

## 🎯 Project Overview

**Energy Consumption Prediction** is a machine learning project designed to estimate household appliance energy consumption through data analysis, feature engineering, and regression modeling.

The project uses the UCI Appliances Energy Prediction dataset, collected from a single residential building, to investigate consumption patterns and predict energy usage in watt-hours per 10-minute interval.

It integrates exploratory data analysis, model comparison, predictive modeling, and an interactive application into a structured machine learning workflow.

## ✨ Key Features

- **Exploratory Data Analysis** — examining energy consumption patterns and relationships between variables.
- **Feature Engineering** — transforming temporal information into cyclical representations.
- **Regression Model Comparison** — evaluating five regression algorithms.
- **Predictive Modeling** — training and exporting an Extra Trees regression model.
- **Model Evaluation** — comparing random-split and chronological evaluation results.
- **Interactive Application** — providing a Streamlit interface for entering inputs and generating predictions.

## 🧠 Machine Learning Methodology

The development process follows a structured workflow, beginning with data exploration and preprocessing, followed by feature engineering, regression model comparison, and model deployment.

The final exported model uses the **Extra Trees Regressor** algorithm.

Cyclical transformations are applied to temporal features to represent recurring patterns more effectively.

The application uses the same feature engineering pipeline as the training process, ensuring consistency between model development and inference.

## 📊 Evaluation & Findings

The project includes two evaluation approaches:

**Random-Split Evaluation**

Assesses predictive performance using randomly distributed training and testing observations.

**Chronological Evaluation**

Examines the model's ability to generalize to later time periods.

The chronological evaluation revealed limitations in forecasting future observations, highlighting the difference between performance under random splitting and real-world temporal generalization.

These findings emphasize the importance of selecting evaluation strategies that reflect the intended deployment environment.

Detailed evaluation results and recommendations are available in the [Final Project Report](reports/FINAL_REPORT_AR.md).

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and model development |
| Scikit-learn | Regression modeling and evaluation |
| Extra Trees Regressor | Exported prediction model |
| Streamlit | Interactive application |
| Jupyter Notebook | Exploratory analysis and model comparison |

## 📁 Project Structure

```text
Energy-Consumption-Prediction/
├── app/              # Interactive Streamlit application
├── data/             # Original dataset
├── models/           # Trained and exported model
├── notebooks/        # Data analysis and model comparison
├── reports/          # Evaluation reports and findings
├── src/              # Feature engineering and training
├── requirements.txt  # Python dependencies
├── README.md         # English documentation
└── README_AR.md      # Arabic documentation
```

## 🚀 Getting Started

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and export the prediction model:

```bash
python src/train_and_export.py
```

Launch the interactive application:

```bash
python -m streamlit run app/app.py
```

## 🔬 Scope & Limitations

The dataset represents observations from a single household. Consequently, the model's predictions should not be assumed to generalize across different residential environments.

Default sensor values in the application are intended for demonstration and do not represent live measurements.

The project demonstrates an end-to-end machine learning workflow while emphasizing transparent evaluation, reproducibility, and the practical limitations of predictive modeling.

## 👩‍💻 Developed By

**Fay Naif**

Programmer | Data Analyst

[GitHub](https://github.com/fay-naif)

---

**© 2026 Fay Naif — Energy Consumption Prediction**
