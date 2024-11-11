import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor # gradient boosting
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from scipy.stats import zscore
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from google.colab import drive
drive.mount('/content/drive')

# Load in all datasets
lotwize_path = 'https://drive.google.com/uc?export=download&id=1eHW3VlKRt5TdIgqrmBAOQUtkQxxsNjid'
lotwize = pd.read_excel(lotwize_path)

count_path = 'https://drive.google.com/uc?export=download&id=1et52HmJrxFYXOovwhCE-6ADTVvjgksYv'
count = pd.read_csv(count_path)

distance_path = 'https://drive.google.com/uc?export=download&id=1nssLniEwBDMIOjOEKQ2JNmL6lNYJuSji'
distance = pd.read_csv(distance_path)

median_income_path = 'https://drive.google.com/uc?export=download&id=1QZw4EWpmwQdEQ453hmUyvx1tyFgauVqk'
median_income = pd.read_excel(median_income_path)

bar_ratings_path = 'https://drive.google.com/uc?export=download&id=1OXwCT8e9KgBo1h2Cb1JJuKZx1zGHOiCF'
bar_ratings = pd.read_excel(bar_ratings_path)

best_school_rating_path = 'https://drive.google.com/uc?export=download&id=1Ck7xn1nF-zOB3Af9iu3_osS2gTc7j2Wd'
best_school_rating = pd.read_excel(best_school_rating_path)

coast_distance_path = 'https://drive.google.com/uc?export=download&id=166wDKsS4Ua5jLBCdp5rI5MuIZK0q13jI'
coast_distance = pd.read_excel(coast_distance_path)

# Standardize the city names so that they can be joined upon
lotwize['city'] = (lotwize['city']
                      .str.strip()  # Remove leading/trailing spaces
                      .str.title()  # Convert to title case
                      .str.replace(r'\s+', ' ', regex=True))  # Replace multiple spaces with a single space

# Round longitude and lattitude to 6 to keep consistency so they can be joined upon
lotwize['longitude'] = lotwize['longitude'].round(6)
lotwize['latitude'] = lotwize['latitude'].round(6)
count['longitude'] = count['longitude'].round(6)
count['latitude'] = count['latitude'].round(6)
distance['longitude'] = distance['longitude'].round(6)
distance['latitude'] = distance['latitude'].round(6)
bar_ratings['longitude'] = distance['longitude'].round(6)
bar_ratings['latitude'] = distance['latitude'].round(6)
best_school_rating['longitude'] = distance['longitude'].round(6)
best_school_rating['latitude'] = distance['latitude'].round(6)
coast_distance['longitude'] = distance['longitude'].round(6)
coast_distance['latitude'] = distance['latitude'].round(6)

# Drop any redundant coordinates and city names
median_income = median_income.drop_duplicates(subset=['zipcode'])

count = count.drop_duplicates(subset=['longitude', 'latitude'])
distance = distance.drop_duplicates(subset=['longitude', 'latitude'])
bar_ratings = bar_ratings.drop_duplicates(subset=['longitude', 'latitude'])
best_school_rating = best_school_rating.drop_duplicates(subset=['longitude', 'latitude'])
coast_distance = coast_distance.drop_duplicates(subset=['longitude', 'latitude'])

# Step 1: Check Shape of Lotwize
print(lotwize.shape)

# Step 2: Merge lotwize with median household income dataset on 'zipcode' (LEFT join to keep 9142 rows)
merged_data = pd.merge(lotwize, median_income, how='left', on='zipcode')

# Step 3: Merge with count dataset using both 'longitude' and 'latitude' (LEFT join)
merged_data = pd.merge(merged_data, count, how='left', on=['longitude', 'latitude'])

# Step 4: Merge with distance dataset using both 'longitude' and 'latitude' (LEFT join)
merged_data = pd.merge(merged_data, distance, how='left', on=['longitude', 'latitude'])

# Step 5: Merge with average bar ratings dataset using both 'longitude' and 'latitude' (LEFT join)
merged_data = pd.merge(merged_data, bar_ratings, how='left', on=['longitude', 'latitude'])

# Step 6: Merge with best school ratings dataset using both 'longitude' and 'latitude' (LEFT join)
merged_data = pd.merge(merged_data, best_school_rating, how='left', on=['longitude', 'latitude'])

# Step 7: Merge with coast distance dataset using both 'longitude' and 'latitude' (LEFT join)
all_data = pd.merge(merged_data, coast_distance, how='left', on=['longitude', 'latitude'])

# Step 8: Ensure no rows have been added and the left joins were successful
print(all_data.shape)

# Handle na values for distance from coast
all_data['coast_distance'] = all_data['coast_distance'].fillna(100) #we did max 100 mile search in ArcGIS, so will fill with 100

# Drop rest of rows with na values in any of these columns
all_data.dropna(subset=['yearBuilt', "livingArea", 'average_bar_rating', 'best_school_rating',
          'GasStation_Count', 'GolfCourse_Count', 'household_income', 'TraderJoes_distance']
, inplace=True)

# Print the ending shape to compare to the original and make sure there are no na values left
print(all_data.shape)
selected_data = ['yearBuilt', "livingArea", 'average_bar_rating', 'household_income', 'best_school_rating',
          'GasStation_Count', 'GolfCourse_Count', 'TraderJoes_distance']
all_data[selected_data].isna().any()

# Feature engineering

# Years since built
all_data['years_since_built'] = 2024 - all_data['yearBuilt']

# Replace zeros with -1 (assuming zeros mean that there are no bars in the area and we will distinguish this with -1)
all_data['average_bar_rating'] = all_data['average_bar_rating'].replace(0, -1)

# Replace zeros with -1 (assuming zeros mean that there are no schools in the area and we will distinguish this with -1)
all_data['best_school_rating'] = all_data['best_school_rating'].replace("No rating", -1)

selected_columns = [
    "livingArea", "years_since_built", "GasStation_Count", "GolfCourse_Count",
    "average_bar_rating", "household_income", "TraderJoes_distance", "best_school_rating", "coast_distance", "price"
]

# Select the specified columns from all_data
all_data_selected = all_data[selected_columns]

all_data_selected.head()

all_data_selected.to_csv('all_data.csv', index=False)

# Trigger the download
files.download('all_data.csv')

# Function to plot histograms for each feature
def plot_feature_distributions(data, selected_data):
    # Set up the figure size based on the number of features
    num_features = len(selected_data)
    plt.figure(figsize=(15, num_features * 3))

    for i, feature in enumerate(selected_data, 1):
        plt.subplot(num_features, 1, i)
        plt.hist(data[feature].dropna(), bins=30, color='brown', alpha=0.7)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

plot_feature_distributions(all_data, selected_data)

# Log_transform function
def log_transform(data, column):
    """Apply log transformation to a specific column and return the transformed column."""
    return np.log1p(data[column])  # log1p to handle zeros and positive values

# Cap_outliers function
def cap_outliers(data, column, lower_percentile=0.01, upper_percentile=0.99):
    """Cap outliers at the given percentiles for a specific column."""
    lower_bound = data[column].quantile(lower_percentile)
    upper_bound = data[column].quantile(upper_percentile)
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data[column]  # Return only the modified column

# Step 1: Apply log transformations for skewed features
all_data['livingArea'] = log_transform(all_data, 'livingArea')
all_data['household_income'] = log_transform(all_data, 'household_income')
all_data['TraderJoes_distance'] = log_transform(all_data, 'TraderJoes_distance')

# Step 2: Cap outliers for count-based features and `yearBuilt`
all_data['GasStation_Count'] = cap_outliers(all_data, 'GasStation_Count', upper_percentile=0.99)
all_data['GolfCourse_Count'] = cap_outliers(all_data, 'GolfCourse_Count', upper_percentile=0.95)
all_data['yearBuilt'] = cap_outliers(all_data, 'yearBuilt', upper_percentile=0.99)

# Step 3: Chose to leave ratings data and not creat binary indicator for -1 values because Gradient Boosting Regressor can handle them.

# Step 4: Plot histograms for each feature (after transformations)
plot_feature_distributions(all_data, selected_data)

#defining the features
features = ["livingArea", 'years_since_built', 'GasStation_Count', 'GolfCourse_Count', 'average_bar_rating',
            'household_income', 'TraderJoes_distance', 'best_school_rating', 'coast_distance']

#correlation heat map to check for multicollinearity
plt.figure(figsize=(12, 8))
sb.heatmap(all_data[features + ['price']].corr(), annot=True, cmap='coolwarm')
plt.show()

# Gradient Boosting Regressor

# Define X and y variables
X = all_data[features]
y = all_data['price']

# Preprocessing
pre = make_column_transformer((StandardScaler(), features), remainder="passthrough")

# Create the Gradient Boosting Pipeline
pipe_gbr = Pipeline([("pre", pre),  # Preprocessing steps
                     ("gradientboostingregressor", GradientBoostingRegressor())])

# Define the hyperparameter grid
param_distributions = {
    'gradientboostingregressor__n_estimators': [100, 300],
    'gradientboostingregressor__learning_rate': [0.05, 0.1],
    'gradientboostingregressor__max_depth': [2, 3],
    'gradientboostingregressor__min_samples_split': [10],
    'gradientboostingregressor__min_samples_leaf': [5],
    'gradientboostingregressor__subsample': [0.8]
}

# Define a KFold object for cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Setup RandomizedSearchCV using KFold
random_search = RandomizedSearchCV(
    pipe_gbr, param_distributions, n_iter=8,
    cv=kfold,  # Use KFold cross-validation for hyperparameter tuning
    scoring='neg_mean_squared_error',  # Base search on negative Mean Squared Error
    n_jobs=-1, random_state=42, verbose=1
)

# Perform hyperparameter tuning with KFold cross-validation
random_search.fit(X, y)

# Get the best model after tuning
best_model = random_search.best_estimator_

# Set up storage for training and testing results across folds
train_mse, test_mse = [], []
train_mae, test_mae = [], []
train_r2, test_r2 = [], []

# Perform manual KFold cross-validation on the final tuned model
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the best model on the training data
    best_model.fit(X_train, y_train)

    # Predictions on the training and testing data
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Training metrics
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    train_mae.append(mean_absolute_error(y_train, y_train_pred))
    train_r2.append(r2_score(y_train, y_train_pred))

    # Testing metrics
    test_mse.append(mean_squared_error(y_test, y_test_pred))
    test_mae.append(mean_absolute_error(y_test, y_test_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

# Convert lists to numpy arrays for averaging
train_mse, test_mse = np.array(train_mse), np.array(test_mse)
train_mae, test_mae = np.array(train_mae), np.array(test_mae)
train_r2, test_r2 = np.array(train_r2), np.array(test_r2)

# Output training vs. testing performance (averaged across folds)
print("Training Set Performance (Averaged across folds):")
print(f"  - MSE: {train_mse.mean():.2f}")
print(f"  - MAE: {train_mae.mean():.2f}")
print(f"  - R2 : {train_r2.mean():.2f}")

print("\nTesting Set Performance (Averaged across folds):")
print(f"  - MSE: {test_mse.mean():.2f}")
print(f"  - MAE: {test_mae.mean():.2f}")
print(f"  - R2 : {test_r2.mean():.2f}")

# Check for potential overfitting
if train_r2.mean() > test_r2.mean():
    print("\nThe model may be overfitting, as the training R2 is higher than the testing R2.")
else:
    print("\nThe model does not seem to be overfitting, as training and testing R2 scores are close.")

# Linear Regression w/ Polynomial Features

# Add Polynomial Features (degree 2 for now, can adjust later)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Create Pipeline with Polynomial Features
pipe_poly_lr = Pipeline([
    ("pre", pre),  # Preprocessing steps
    ("poly", poly),  # Add polynomial features
    ("linearregression", LinearRegression())  # Linear Regression
])

# Define a KFold object for cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up storage for training and testing results across folds
train_mse, test_mse = [], []
train_mae, test_mae = [], []
train_r2, test_r2 = [], []

# Perform KFold cross-validation on the Polynomial Linear Regression model
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    pipe_poly_lr.fit(X_train, y_train)

    # Predictions on the training and testing data
    y_train_pred = pipe_poly_lr.predict(X_train)
    y_test_pred = pipe_poly_lr.predict(X_test)

    # Training metrics
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    train_mae.append(mean_absolute_error(y_train, y_train_pred))
    train_r2.append(r2_score(y_train, y_train_pred))

    # Testing metrics
    test_mse.append(mean_squared_error(y_test, y_test_pred))
    test_mae.append(mean_absolute_error(y_test, y_test_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

# Convert lists to numpy arrays for averaging
train_mse, test_mse = np.array(train_mse), np.array(test_mse)
train_mae, test_mae = np.array(train_mae), np.array(test_mae)
train_r2, test_r2 = np.array(train_r2), np.array(test_r2)

# Output training vs. testing performance (averaged across folds)
print("Training Set Performance (Averaged across folds):")
print(f"  - MSE: {train_mse.mean():.2f}")
print(f"  - MAE: {train_mae.mean():.2f}")
print(f"  - R2 : {train_r2.mean():.2f}")

print("\nTesting Set Performance (Averaged across folds):")
print(f"  - MSE: {test_mse.mean():.2f}")
print(f"  - MAE: {test_mae.mean():.2f}")
print(f"  - R2 : {test_r2.mean():.2f}")

# Check for potential overfitting
if train_r2.mean() > test_r2.mean():
    print("\nThe model may be overfitting, as the training R2 is higher than the testing R2.")
else:
    print("\nThe model does not seem to be overfitting, as training and testing R2 scores are close.")

# Feature Importance

# Get the Gradient Boosting Regressor from the pipeline
gbr = best_model.named_steps['gradientboostingregressor']

# Get feature importances
importances = gbr.feature_importances_

# Get the feature names from the preprocessing step
feature_names = best_model.named_steps['pre'].get_feature_names_out()

# Combine feature names and importances into a DataFrame
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Display the feature importances DataFrame
print(importances_df)

# Customizing the plot to fit the brown and cream theme
plt.figure(figsize=(12, 8))

# Setting the theme colors
sns.set(style="whitegrid")
custom_palette = sns.color_palette(["#8B4513", "#D2B48C"])  # Brown and Cream colors

# Create the barplot
sns.barplot(x='Importance', y='Feature', data=importances_df, palette=custom_palette)

# Customize the titles and labels to fit the theme
plt.title('Feature Importance from Gradient Boosting Regressor', fontsize=16, color='#8B4513')  # Brown title
plt.xlabel('Importance', fontsize=14, color='#8B4513')
plt.ylabel('Feature', fontsize=14, color='#8B4513')

# Customize color
plt.xticks(color='#8B4513')
plt.yticks(color='#8B4513')

# Show the plot
plt.tight_layout()
plt.show()

# Exporting final cleaned dataset

final_dataset = all_data[features + ['price']]

# Specify the path where the CSV file will be saved in your Google Drive
output_path_drive = '/content/drive/MyDrive/Project1_Extension/final_cleaned_dataset.csv'

# Export the dataset to a CSV file
final_dataset.to_csv(output_path_drive, index=False)








