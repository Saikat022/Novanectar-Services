import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load data from Excel file
data = pd.read_excel(r"D:\HousePricePrediction.xlsx")

# Drop rows with missing values
data.dropna(inplace=True)

# Split data into features and target variable
X = data.drop('SalePrice', axis=1)  # Features
y = data['SalePrice']  # Target variable

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline with imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_cols),  # Impute missing values for numerical columns
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Define the model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predictions on the testing set
y_pred = pipeline.predict(X_test)

# Model evaluation
print("Model Evaluation:")
print("R^2 Score:", pipeline.score(X_test, y_test))

# Plotting actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Combine actual prices, predicted prices, and features into a DataFrame
predictions_df = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': y_pred})
predictions_df = pd.concat([predictions_df, X_test.reset_index(drop=True)], axis=1)

# Save DataFrame to Excel file
predictions_df.to_excel("predictions.xlsx", index=False)
