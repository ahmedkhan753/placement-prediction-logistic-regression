# placement-prediction-logistic-regression

![WhatsApp Image 2025-12-07 at 13 21 56_2eba6d03](https://github.com/user-attachments/assets/db6581f2-5755-46bf-a5a5-6ca356302d18)
![WhatsApp Image 2025-12-07 at 13 24 27_c524998f](https://github.com/user-attachments/assets/e1dc80e6-822c-488c-861c-2cecb4cece3c)
![WhatsApp Image 2025-12-07 at 13 24 39_e450aa97](https://github.com/user-attachments/assets/5c1cca68-187b-43af-8e68-f9f30d9cbf33)




# 📊 Student Placement Prediction using Logistic Regression

This project is a **supervised machine learning classification model** that predicts whether a student is likely to get **placed or not** based on their **CGPA** and **IQ**.

The goal of this project is to understand and implement the **complete machine learning workflow**, from data preprocessing to model evaluation and visualization.

---

## 🔍 Problem Statement

Given a dataset containing students’ academic information:

* **CGPA**
* **IQ**

the model predicts:

* **Placement Status** → *Placed (1) or Not Placed (0)*

---

## ⚙️ Machine Learning Workflow

The project follows a structured ML lifecycle:

1. **Data Loading & Cleaning**

   * Loaded dataset using **Pandas**
   * Removed unnecessary index column for clean analysis

2. **Exploratory Data Analysis (EDA)**

   * Visualized the relationship between CGPA, IQ, and placement using **Matplotlib**
   * Helped in understanding class separation before modeling

3. **Feature Selection**

   * Independent variables: `CGPA`, `IQ`
   * Target variable: `Placement`

4. **Train-Test Split**

   * Split the dataset into training and testing sets using **scikit-learn**
   * Test size: 10%

5. **Feature Scaling**

   * Applied **StandardScaler** to normalize feature values
   * Ensures better model convergence and performance

6. **Model Training**

   * Trained a **Logistic Regression** classifier
   * Suitable for binary classification problems

7. **Model Evaluation**

   * Evaluated accuracy on test data
   * Compared predicted values with actual outcomes

8. **Decision Boundary Visualization**

   * Plotted decision regions using **mlxtend**
   * Provides clear insight into how the model separates classes

---

## 🧰 Technologies & Libraries Used

* **Python**
* **Pandas** – data manipulation
* **NumPy** – numerical operations
* **Matplotlib** – data visualization
* **Scikit-learn** – preprocessing, model training, and evaluation
* **Mlxtend** – decision region visualization

---

## ✅ Results

* The model successfully classifies whether a student is likely to get placed based on CGPA and IQ with 80% accuracy.
* Visualization of decision boundaries helps interpret model behavior beyond just accuracy scores.

---







