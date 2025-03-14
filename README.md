# 🏗️ Bulk Target Weight Prediction using Polynomial Regression 🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Polynomial%20Regression-orange)

This project demonstrates the use of **Polynomial Regression** to predict the `Bulk_Target_Weight` based on various input features such as `NMAS_mm`, `PG_Low`, `RAP_Percent`, `Gyrations`, `Target_Air_Voids_Percent`, and `RBR_Percent`. The dataset is preprocessed, missing values are handled, and the model is trained and evaluated using metrics like Mean Squared Error (MSE) and R-squared.

![Actual vs Predicted](https://github.com/riyagpt0251/bulk-target-weight-prediction/raw/main/assets/actual_vs_predicted.png)  
*(Visualization of Actual vs Predicted Bulk_Target_Weight)*

---

## 🚀 Features

- 🧠 **Polynomial Regression**: Captures non-linear relationships between input features and target variable.
- 📊 **Data Preprocessing**: Handles missing values using mean imputation.
- 📈 **Model Evaluation**: Provides metrics like Mean Squared Error (MSE) and R-squared score.
- 🎯 **Prediction Visualization**: Compares actual vs. predicted values for `Bulk_Target_Weight`.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bulk-target-weight-prediction.git
   cd bulk-target-weight-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Dependencies:**
- `numpy==1.24.3`
- `pandas==1.5.3`
- `scikit-learn==1.2.2`
- `matplotlib==3.7.1`

---

## 🛠️ Usage

### Load the Dataset
The dataset is loaded from an Excel file (`INFOMATERIALS_DATA_2025-1-24_000465.xlsx`). Replace the URL with your actual dataset path.

```python
url = 'INFOMATERIALS_DATA_2025-1-24_000465.xlsx'  # Replace with your actual dataset URL
df = pd.read_excel(url)
```

### Define Input Features and Target Variable
```python
X = df[['NMAS_mm', 'PG_Low', 'RAP_Percent', 'Gyrations', 'Target_Air_Voids_Percent', 'RBR_Percent']]  # Input features
y = df['Bulk_Target_Weight']
```

### Handle Missing Values
Missing values in the input features (`X`) and target variable (`y`) are imputed using the mean strategy.

```python
imputer_X = SimpleImputer(strategy='mean')  # Impute missing values in X with mean
X_imputed = imputer_X.fit_transform(X)
imputer_y = SimpleImputer(strategy='mean')  # Impute missing values in y with mean
y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()
```

### Split the Data into Training and Test Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.3, random_state=42)
```

### Apply Polynomial Features
Polynomial features of degree 2 are applied to capture non-linear relationships.

```python
poly = PolynomialFeatures(degree=2)  # You can adjust the degree
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
```

### Train the Linear Regression Model
```python
model = LinearRegression()
model.fit(X_poly_train, y_train)
```

### Predict Using the Test Set
```python
y_pred = model.predict(X_poly_test)
```

### Evaluate the Model
```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

### Visualize Actual vs Predicted Values
```python
plt.scatter(y_test, y_pred, label='Predicted vs Actual', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual Bulk_Target_Weight')
plt.ylabel('Predicted Bulk_Target_Weight')
plt.title('Polynomial Regression: Actual vs Predicted Bulk_Target_Weight')
plt.show()
```

---

## ⚙️ Hyperparameters

| Parameter            | Value   | Description                          |
|----------------------|---------|--------------------------------------|
| Polynomial Degree    | 2       | Degree of polynomial features        |
| Test Size            | 0.3     | Proportion of the dataset to include in the test split |
| Random State         | 42      | Seed for reproducibility             |

---

## 📊 Evaluation Results

| Metric           | Value               |
|------------------|---------------------|
| Mean Squared Error | 16263542.435842892 |
| R-squared        | -4.962396591319853  |

---

## 🖼️ Prediction Visualization

![Actual vs Predicted](https://github.com/yourusername/bulk-target-weight-prediction/raw/main/assets/actual_vs_predicted.png)  
*(Visualization of Actual vs Predicted Bulk_Target_Weight)*

---

## 🏗️ Project Structure

```
bulk-target-weight-prediction/
├── src/
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── assets/               # Visual assets (images, graphs)
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── LICENSE               # MIT License
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <img src="https://github.com/yourusername/bulk-target-weight-prediction/raw/main/assets/ml_icon.png" width="100">
  <br>
  <em>Predicting Bulk Target Weight with Polynomial Regression!</em>
</p>
```

---

### **How to Use This README**
1. Replace `yourusername` with your GitHub username.
2. Add the following files to the `assets/` folder:
   - `actual_vs_predicted.png`: Visualization of actual vs predicted values.
   - `ml_icon.png`: A machine learning-themed icon.
3. Add the implementation files (`train.py`, `evaluate.py`) to the `src/` folder.

This README combines **professional styling**, **visual elements**, and **detailed explanations** to make your GitHub repository stand out! 🚀
```
