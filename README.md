# Cardio Health Risk Assessment
**Autor: Aaron Elizondo**

**Used libraries**
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns

- from sklearn.ensemble import GradientBoostingClassifier
- from sklearn.ensemble import AdaBoostClassifier
- from sklearn.naive_bayes import GaussianNB
- from sklearn.neural_network import MLPClassifier
- from xgboost import XGBClassifier
- from sklearn.linear_model import LogisticRegression
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.svm import SVC
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.metrics import accuracy_score

**Data**

- Age: The patient’s age.
- Sex: The patient’s gender.
- Chest pain type: The classification of chest pain experienced (e.g., typical angina, atypical, non-anginal).
- BP (Blood Pressure): The patient’s blood pressure.
- Cholesterol: The level of cholesterol in the patient’s blood.
- FBS over 120 (Fasting Blood Sugar over 120 mg/dl): Indicates if the patient’s fasting blood sugar level is over 120 mg/dl.
- EKG results: The findings from the patient’s electrocardiogram.
- Max HR (Maximum Heart Rate): The maximum heart rate achieved during exercise.
- Exercise angina: Whether the patient experienced angina (chest pain) during exercise.
- ST depression: The amount of ST segment depression observed on the EKG after exercise, an indicator of coronary artery disease.
- Slope of ST: The slope of the ST segment during exercise, which can indicate the severity of coronary artery disease.
- Number of vessels fluro (fluoroscopy): The number of major blood vessels visible on a fluoroscopy, a type of medical imaging.
- Thallium: Refers to a thallium stress test, a radioactive isotope used in myocardial perfusion imaging to detect poor blood flow areas to the heart.
- Heart Disease: Indicates whether or not the patient has been diagnosed with heart disease.

*Training models for scoring tests**
- "logistic": LogisticRegression()
- "decision_tree": DecisionTreeClassifier()
- "svm": SVC()
- "knn": KNeighborsClassifier()
- "random_forest": RandomForestClassifier()
- "gradient_boosting": GradientBoostingClassifier()
- "adaboost": AdaBoostClassifier()
- "naive_bayes": GaussianNB()
- "mlp": MLPClassifier()
- "xgboost": XGBClassifier()

**Used models**
- "naive_bayes": GaussianNB(),
- "logistic": LogisticRegression()

# Result

**Logistic Regression:**

- The ROC curve remained at 0.95, which is still a good score.
- The results for F1, recall, precision, and accuracy have improved from:
F1 = 0.9275, Recall = 0.9697, Precision = 0.8889, Accuracy = 0.9074
to:
F1 = 0.9429, Recall = 1.0, Precision = 0.8919, Accuracy = 0.9259.
- This improvement made the recall score perfect.
- The comparison between real vs predicted labels is almost perfect; out of 20 predictions, there was only one error.

**Naive Bayes:**

- When predicting against real values, it has a higher but still low error rate; out of 20 predictions, it only fails 3 times.
- The ROC curve could not be improved, and it was not possible to improve the recall, accuracy, F1, and precision with the applied hyperparameters.
- The logistic model performs better than the naive_bayes model.
It seems that Logistic Regression is outperforming Naive Bayes in this scenario based on the metrics provided. Would you like any further explanation or assistance with these models?

