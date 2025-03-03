# 🏡 **California House Price Prediction: An End-to-End Machine Learning Project**  
**Author:** Shrunali Salian  
**Skills:** Data Preprocessing, Feature Engineering, Regression Models, Scikit-Learn  

---

## 🚀 **Project Overview**  
This project is inspired by **Chapter 2** of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron. The goal is to **build a machine learning model that predicts house prices in California** based on real-world housing data.  

This project covers the **entire machine learning pipeline**, including:  
✅ **Data Collection & Preprocessing** – Handling missing values, feature scaling  
✅ **Exploratory Data Analysis (EDA)** – Visualizing key housing trends  
✅ **Feature Engineering** – Transforming raw data into meaningful predictors  
✅ **Model Selection & Training** – Comparing multiple regression models  
✅ **Hyperparameter Tuning** – Optimizing model performance  

📌 **Reference Repository:** [GitHub - Aurélien Géron](https://github.com/ageron/handson-ml2)  

---

## 🎯 **Key Objectives**  
✔ **Develop a predictive model for house prices**  
✔ **Perform in-depth data analysis & feature selection**  
✔ **Compare different machine learning algorithms**  
✔ **Optimize model performance using cross-validation & fine-tuning**  

---

## 📊 **Dataset Overview**  
The **California Housing Dataset** contains housing market information, including:  
- 📍 **Geographical Features** – Latitude, Longitude  
- 🏠 **Housing Characteristics** – Number of rooms, bedrooms, population  
- 💰 **Median Income & House Value** – Key economic indicators  

✅ **Example: Loading the Dataset**  
```python
import pandas as pd

housing = pd.read_csv("housing.csv")
housing.head()
```

✅ **Example: Checking for Missing Values**  
```python
housing.info()
```
💡 **Observation:** Some columns, like `total_bedrooms`, have missing values.  

---

## 📈 **Exploratory Data Analysis (EDA)**  
Before training the model, we analyze the data distribution and key insights.  

✅ **Example: Visualizing Geographical Data**  
```python
import matplotlib.pyplot as plt

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="Population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()
```
💡 **Insight:** Coastal regions tend to have **higher house prices**.  

✅ **Example: Correlation Analysis**  
```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```
💡 **Key Observations:**  
- **Median Income is highly correlated** with house prices.  
- **Number of Rooms also has a positive correlation** with house value.  

---

## 🏗 **Feature Engineering & Data Preprocessing**  
We transform the data to improve model performance.  

✅ **Handling Missing Values**  
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
housing_num_imputed = imputer.transform(housing_num)
```
💡 **Why?** – Missing values in `total_bedrooms` are replaced with the median.  

✅ **Creating New Features**  
```python
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
```
💡 **Why?** – Feature engineering **captures meaningful relationships** between variables.  

---

## 🤖 **Model Training & Evaluation**  
We compare different **regression models** for price prediction.  

### **Baseline Models:**  
✅ **Linear Regression**  
```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```
✅ **Decision Tree Regression**  
```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
```
✅ **Random Forest Regression**  
```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100)
forest_reg.fit(housing_prepared, housing_labels)
```
💡 **Why Multiple Models?** – Each regression model **has strengths & weaknesses**; comparing them **helps find the best one**.  

---

## 📊 **Model Evaluation & Performance Metrics**  
We evaluate models using:  
✔ **Root Mean Squared Error (RMSE)** – Measures prediction accuracy  
✔ **Cross-Validation (K-Fold CV)** – Ensures generalization  
✔ **Hyperparameter Tuning (GridSearchCV)** – Finds the best model settings  

✅ **Example: RMSE Calculation**  
```python
from sklearn.metrics import mean_squared_error

predictions = lin_reg.predict(housing_prepared)
rmse = mean_squared_error(housing_labels, predictions, squared=False)
print(f"Linear Regression RMSE: {rmse}")
```
✅ **Example: Cross-Validation for Model Selection**  
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(f"Decision Tree RMSE: {rmse_scores.mean()}")
```
💡 **Finding:** **Random Forest performed the best**, reducing RMSE significantly.  

✅ **Example: Hyperparameter Tuning with GridSearchCV**  
```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators": [50, 100, 150], "max_features": [8, 10, 12]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
```
💡 **Why?** – Hyperparameter tuning **optimizes model performance**.  

---

## 🔮 **Future Enhancements**  
🔹 **Deploying the model** as a web app for real-time predictions  
🔹 **Using Deep Learning (Neural Networks)** for better accuracy  
🔹 **Feature Selection Optimization** using Recursive Feature Elimination (RFE)  

---

## 🎯 **Why This Project Stands Out for ML & Data Science Roles**  
✔ **End-to-End Machine Learning Pipeline** – Covers all steps from data to model tuning  
✔ **Compares Multiple Regression Models** – Linear Regression, Decision Trees, Random Forests  
✔ **Strong Feature Engineering & EDA** – Key insights for better predictions  
✔ **Demonstrates Model Optimization** – Hyperparameter tuning improves accuracy  

---

## 🛠 **How to Run This Project**  
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/california-house-price-prediction.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook "California House Price Prediction.ipynb"
   ```

---

## 📌 **Connect with Me**  
- **LinkedIn:** [Shrunali Salian](https://www.linkedin.com/in/shrunali-salian/)  
- **Portfolio:** [Your Portfolio Link](#)  
- **Email:** [Your Email](#)  
