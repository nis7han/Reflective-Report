# from google.colab import drive
# drive.mount('/content/drive/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

# file_path = '/content/drive/MyDrive/Housing.csv'
file_path = '\DataScience_Final\Housing.csv'
df = pd.read_csv(file_path)

# Data Preprocessing
print(df.isnull().sum())

# Filling missing values for numeric columns with median
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Filling missing values for categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)



# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Checking the cleaned DataFrame structure
print(df.info())

# Defining features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Keeping X as a DataFrame for feature selection later
X_df = X.copy()

# Converting to float32 for TensorFlow compatibility
X = X.values.astype(np.float32)
y = y.values.astype(np.float32)

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluating Linear Regression Model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f'Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, R^2: {r2_linear}')

# 2. Polynomial Regression
poly_features = PolynomialFeatures(degree=2)  
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)

# Evaluating Polynomial Regression Model
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'Polynomial Regression - MAE: {mae_poly}, MSE: {mse_poly}, R^2: {r2_poly}')



# 3. Feature Selection using RFE
selector = RFE(linear_model, n_features_to_select=5)  # Selecting top 5 features
selector.fit(X_train, y_train)
selected_features = X_df.columns[selector.support_]  # Use X_df to get column names

print("Selected Features:", selected_features)

# 4. K-Means Clustering
features_for_clustering = df[['area', 'bedrooms', 'bathrooms']]
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_for_clustering)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_for_clustering)

print(df.head())

# Baseline Model - KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=3) 
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluating KNN Model Performance
mse_knn = mean_squared_error(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f'KNN Regressor - MAE: {mae_knn}, MSE: {mse_knn}, R^2: {r2_knn}')

# Parameter Analysis for KNN (finding best k)
neighbors_range = range(1, 21)  # Testing k from 1 to 20
scores = []
for k in neighbors_range:
    knn_model_k = KNeighborsRegressor(n_neighbors=k)
    knn_model_k.fit(X_train, y_train)
    scores.append(knn_model_k.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(neighbors_range, scores)
plt.title('KNN Accuracy vs Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Neural Network Implementation
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model on training data
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Predicting on test set using ANN model
y_pred_ann = model.predict(X_test).flatten()

# Evaluating ANN model performance
mse_ann = mean_squared_error(y_test, y_pred_ann)
mae_ann = mean_absolute_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)

print(f'ANN - Mean Absolute Error: {mae_ann}')
print(f'ANN - Mean Squared Error: {mse_ann}')
print(f'ANN - R-squared: {r2_ann}')

# Visualizing ANN results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ann)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Identityfying the  line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('House Price Predictions vs Actual Prices (ANN)')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 4. K-Means Clustering
features_for_clustering = df[['area', 'bedrooms', 'bathrooms']]
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_for_clustering)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_for_clustering)

# Classification Example
# Prepare data for classification
df['Price_Category'] = np.where(df['price'] > df['price'].median(), 1, 0)  # 1: Expensive, 0: Cheap

X_class = df.drop(['price', 'Price_Category', 'Cluster'], axis=1)  # Exclude target and other non-feature columns
y_class = df['Price_Category']

# Split data into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_class = scaler.fit_transform(X_train_class)
X_test_class = scaler.transform(X_test_class)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_class, y_train_class)

# Predict and evaluate
y_pred_class = logistic_model.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f'Logistic Regression Accuracy: {accuracy}')
print(classification_report(y_test_class, y_pred_class))  # Adding closing parenthesis here

# Predict and evaluate
from sklearn.metrics import accuracy_score, classification_report, f1_score
y_pred_class = logistic_model.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)  # Calculating F1-score

print(f'Logistic Regression Accuracy: {accuracy}')
print(f'Logistic Regression F1-Score: {f1}')
print(classification_report(y_test_class, y_pred_class))

# Introduction
# Defining the problem or questions to analyze
questions = [
    "1. What factors most significantly impact house prices?",
    "2. Can we improve predictions using polynomial regression or neural networks?",
    "3. How do different features correlate with house prices?"
]

print("Analysis Questions:")
for question in questions:
    print(question)

# Additional EDA
# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['area'], bins=30, kde=True)
plt.title('Distribution of House Areas')
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='area', y='price')
plt.title('House Price vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()









