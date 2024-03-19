# INSY 662 Individual Project
# Ethan Pirso
# 260863065

# 1. Importing the Dataset

import pandas as pd

# Load the dataset directly from the Excel file
kickstarter_data = pd.read_excel('Kickstarter.xlsx')

# 2. Data Preprocessing


def preprocess(df):
    # Identifying predictors known only after the project starts and unique identifiers
    invalid_predictors = [
        'pledged', 'backers_count', 'usd_pledged',
        'spotlight', 'state_changed_at', 'state_changed_at_weekday',
        'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr',
        'state_changed_at_hr', 'launch_to_state_change_days',
        'id', 'name'
    ]

    # Drop invalid predictors
    df = df.drop(columns=invalid_predictors)

    # Drop rows with missing values
    df = df.dropna()

    # Drop rows where target 'state' was not 'successful' or 'failed'
    df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]

    # Identify datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns

    # Convert datetime columns to numerical format (e.g., timestamp)
    for col in datetime_cols:
        df[col] = df[col].apply(lambda x: x.timestamp())

    # Normalize numeric features
    from sklearn.preprocessing import StandardScaler

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Dummify target variable 'state'
    df['state'] = df['state'].apply(lambda x: 1 if x == 'successful' else 0)

    # Dummify the remaining categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df


kickstarter_data = preprocess(kickstarter_data)

# 3. Outlier Detection and Removal

from sklearn.ensemble import IsolationForest

# Detecting outliers
iso_forest = IsolationForest(contamination=0.1, bootstrap=True)
outliers = iso_forest.fit_predict(kickstarter_data)
print('Number of outliers to remove:', len(kickstarter_data[outliers == -1]))

# Removing outliers
kickstarter_data = kickstarter_data[outliers != -1]

# 4. Feature Engineering

kickstarter_data['goal_usd_rate'] = kickstarter_data['goal'] * kickstarter_data['static_usd_rate']

# 5. Correlation and Multicollinearity Analysis

import numpy as np

# Removing highly correlated predictors
correlation_matrix = kickstarter_data.corr().abs()

# Identify pairs of highly correlated features
high_corr_var = np.where(correlation_matrix > 0.7)
high_corr_pairs = [(correlation_matrix.columns[x], correlation_matrix.columns[y]) for x, y in zip(*high_corr_var) if
                   x != y and x < y]

col_to_drop = []

# Remove one of each pair, prioritizing removal of features highly correlated with the target
for col1, col2 in high_corr_pairs:
    if 'state_successful' in [col1, col2]:
        # Drop the more correlated one
        col_to_drop = col1 if correlation_matrix['state'][col1] > correlation_matrix['state'][col2] else col2
    else:
        col_to_drop = col1

    # Check if the column exists before dropping
    if col_to_drop in kickstarter_data.columns:
        kickstarter_data.drop(col_to_drop, axis=1, inplace=True)

print('Highly correlated features to remove:', col_to_drop)

# Analyzing VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Isolate numeric features
kickstarter_data_numeric = kickstarter_data.select_dtypes(include=['int64', 'float64'])

# Replace inf/-inf with NaN and drop NaN values
kickstarter_data_numeric = kickstarter_data_numeric.replace([np.inf, -np.inf], np.nan).dropna()


# Function to calculate VIF and return high VIF feature names
def calculate_vif_(df, thresh=5.0):
    variables = list(df.columns)
    high_vif_features = []
    for var in variables:
        vif = variance_inflation_factor(df[variables].values, df.columns.get_loc(var))
        if vif > thresh:
            print(f"{var} with VIF: {vif}")
            high_vif_features.append(var)
    return high_vif_features


# Calculate VIF for numeric features and get the list of features to remove
high_vif_features_to_remove = calculate_vif_(kickstarter_data_numeric)

print('High VIF features to remove:', high_vif_features_to_remove)

# Remove high VIF features from the original DataFrame
kickstarter_data = kickstarter_data.drop(columns=high_vif_features_to_remove)

# 6. Classification Model Training

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# Split the data into features and target
X = kickstarter_data.drop(columns=['state'], axis=1)
y = kickstarter_data['state']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define hyperparameter grid for RandomForest
param_grid_rf = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [3, 5, 7, 10],
    'min_samples_leaf': [3, 5, 7, 10],
}

# Define hyperparameter grid for GradientBoosting
param_grid_gbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize the models
rf = RandomForestClassifier(class_weight='balanced')
gbm = GradientBoostingClassifier()

# Grid search
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, scoring='recall', cv=5)
grid_search_gbm = GridSearchCV(estimator=gbm, param_grid=param_grid_gbm, scoring='recall', cv=5)

# Fit models to training data
grid_search_rf.fit(X_train, y_train)
grid_search_gbm.fit(X_train, y_train)

# Best estimators
best_rf = grid_search_rf.best_estimator_
best_gbm = grid_search_gbm.best_estimator_
print('Best RF parameters:', grid_search_rf.best_params_)
# Best RF parameters: {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 100}
print('Best GBM parameters', grid_search_gbm.best_params_)

# Train other models
log_reg = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)

# Re-declaring best_rf, sometimes grid_search does not actually find the best parameters
best_rf = RandomForestClassifier(class_weight='balanced', max_depth=3, min_samples_leaf=3, min_samples_split=10, n_estimators=100)

models = [log_reg, best_rf, best_gbm, svm]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(
        f'{model.__class__.__name__} (Kickstarter Training) - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, ROC-AUC: {roc_auc}')

# Ensemble Model
from sklearn.ensemble import VotingClassifier

# Create an ensemble of the previously trained models
ensemble = VotingClassifier(estimators=[('lr', log_reg), ('rf', best_rf), ('gbm', best_gbm), ('svm', svm)],
                            voting='soft')
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred)
ensemble_recall = recall_score(y_test, y_pred)
ensemble_precision = precision_score(y_test, y_pred)
ensemble_roc_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])

print(
    f'Ensemble Model (Kickstarter Training) - Accuracy: {ensemble_accuracy}, Recall: {ensemble_recall}, Precision: {ensemble_precision}, ROC-AUC: {ensemble_roc_auc}')

import matplotlib.pyplot as plt

# Get feature importances
feature_importances = best_rf.feature_importances_

# Create a DataFrame for feature importances
features = pd.DataFrame({
    'Feature': kickstarter_data.drop(columns=['state'], axis=1).columns,
    'Importance': feature_importances
})

# Sort features by importance
top_features = features.sort_values(by='Importance', ascending=False).head(10)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.title('Top 10 Features by Importance')
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 7. Classification Model Testing on Best Model - RandomForestClassifier

# Import the grading data
kickstarter_grading_data = pd.read_excel('Kickstarter-Grading.xlsx')

# Apply pre-processing function
kickstarter_grading_data = preprocess(kickstarter_grading_data)

# Feature Engineering
kickstarter_grading_data['goal_usd_rate'] = kickstarter_grading_data['goal'] * kickstarter_grading_data['static_usd_rate']

# Remove highly correlated feature
if col_to_drop in kickstarter_grading_data.columns:
    kickstarter_grading_data.drop(col_to_drop, axis=1, inplace=True)

train_features = list(kickstarter_data.columns)

# Add missing dummy columns as zeros
for feature in train_features:
    if feature not in list(kickstarter_grading_data.columns):
        kickstarter_grading_data[feature] = 0

# Remove new features unseen at fit
for feature in list(kickstarter_grading_data.columns):
    if feature not in train_features:
        kickstarter_grading_data.drop(feature, axis=1, inplace=True)

# Ensure the order of columns matches the training set
kickstarter_grading_data = kickstarter_grading_data[train_features]

# Split the data into features and target
X = kickstarter_grading_data.drop(columns=['state'], axis=1)
y = kickstarter_grading_data['state']

# Make predictions and generate scores
y_pred = best_rf.predict(X)
accuracy = accuracy_score(y, y_pred)
recall = recall_score(y, y_pred)
precision = precision_score(y, y_pred)
roc_auc = roc_auc_score(y, best_rf.predict_proba(X)[:, 1])

print(f'{best_rf.__class__.__name__} (Kickstarter Grading) - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, ROC-AUC: {roc_auc}')

# 8. Clustering Model

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Normalize data for the autoencoder
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(kickstarter_data)

# Train and Save the Model (Do this once)

# Autoencoder Model
autoencoder = Sequential()
autoencoder.add(Dense(50, input_dim=data_scaled.shape[1], activation='relu'))
autoencoder.add(Dense(3, activation='relu'))  # Encoded Layer
autoencoder.add(Dense(20, activation='relu'))
autoencoder.add(Dense(data_scaled.shape[1], activation='sigmoid'))
autoencoder.compile(optimizer='adam', loss='mse')

# Fit the model
autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=512, verbose=0)

# Save the entire model (architecture + weights)
autoencoder.save('autoencoder_model.h5')

# Load the Model and Encode Data (Do this in subsequent runs)

# Load the saved model
autoencoder = load_model('autoencoder_model.h5')

# Extract the encoder part
encoder = Sequential()
encoder.add(autoencoder.layers[0])
encoder.add(autoencoder.layers[1])

# Encode data
encoded_data = encoder.predict(data_scaled)

# Determining the optimal number of clusters
best_score = -1
optimal_clusters = 0

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    clusters = kmeans.fit_predict(encoded_data)
    score = silhouette_score(encoded_data, clusters)
    print(f'Number of clusters: {k}, Silhouette Score: {score}')
    if score > best_score:
        best_score = score
        optimal_clusters = k

print(f'Optimal number of clusters: {optimal_clusters}, Best Silhouette Score: {best_score}')

# K-Means Clustering with Optimal Number of Clusters
kmeans_optimal = KMeans(n_clusters=optimal_clusters, n_init='auto', random_state=42)
clusters_optimal = kmeans_optimal.fit_predict(encoded_data)

# Add cluster assignments to original data
kickstarter_data['Cluster'] = clusters_optimal

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
pca_components = pca.fit_transform(encoded_data)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters_optimal, cmap='viridis', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of K-Means Clusters')
plt.show()

# Numerical Feature Analysis
numerical_features = kickstarter_data.select_dtypes(include=['int64', 'float64']).columns
cluster_means = kickstarter_data.groupby('Cluster')[numerical_features].mean()

# Plotting the means for comparison
cluster_means.plot(kind='bar', figsize=(30, 7))
plt.title('Mean of Numerical Features by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.legend(title='Feature', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()

# Categorical Feature Analysis for 'category'

# Identifying all dummified 'category' columns
category_columns = [col for col in kickstarter_data.columns if col.startswith('category_')]

# Calculate proportions for each category within each cluster
category_proportions = kickstarter_data.groupby('Cluster')[category_columns].mean()

# Transpose for easier plotting
category_proportions = category_proportions.T

# Plotting
plt.figure(figsize=(15, 10))
category_proportions.plot(kind='bar', figsize=(15, 10))
plt.title('Proportion of Project Categories by Cluster')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

from sklearn.cluster import DBSCAN

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.62, min_samples=20)
clusters_dbscan = dbscan.fit_predict(encoded_data)

# Filter out noisy samples (-1 cluster label)
filtered_data = encoded_data[clusters_dbscan != -1]
filtered_labels = clusters_dbscan[clusters_dbscan != -1]

# Only calculate silhouette score if clusters are present
if len(set(filtered_labels)) > 1:
    score_dbscan = silhouette_score(filtered_data, filtered_labels)
    print(f'DBSCAN Silhouette Score (excluding noise): {score_dbscan}')
else:
    print("Not enough clusters for silhouette scoring")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters_dbscan, cmap='viridis', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of DBSCAN Clusters')
plt.show()

# Add cluster assignments to original data
kickstarter_data['Cluster'] = clusters_dbscan

# Remove noisy samples (-1 cluster label)
kickstarter_data = kickstarter_data[kickstarter_data['Cluster'] != -1]

# Numerical Feature Analysis
numerical_features = kickstarter_data.select_dtypes(include=['int64', 'float64']).columns
cluster_means = kickstarter_data.groupby('Cluster')[numerical_features].mean()
cluster_means.drop('Cluster', axis=1, inplace=True)

# Plotting the means for comparison
cluster_means.plot(kind='bar', figsize=(30, 7))
plt.title('Mean of Numerical Features by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.legend(title='Feature', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()

# Categorical Feature Analysis for 'category'

# Identifying all dummified 'category' columns
category_columns = [col for col in kickstarter_data.columns if col.startswith('category_')]

# Calculate proportions for each category within each cluster
category_proportions = kickstarter_data.groupby('Cluster')[category_columns].mean()

# Transpose for easier plotting
category_proportions = category_proportions.T

# Plotting
plt.figure(figsize=(15, 10))
category_proportions.plot(kind='bar', figsize=(15, 10))
plt.title('Proportion of Project Categories by Cluster')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
