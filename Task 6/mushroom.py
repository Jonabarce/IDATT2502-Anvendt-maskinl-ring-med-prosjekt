import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Load the UCI mushroom dataset
def load_data():
    # Define dataset URL and column names
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                   "gill-spacing",
                   "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                   "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                   "veil-color",
                   "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

    # Load dataset from the URL
    return pd.read_csv(url, names=column_names)

# Convert categorical columns to numeric labels
def encode_data(data):
    label_encoders = {}
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    return data, label_encoders

# Determine and print the importance of each feature using RandomForest
def feature_importance_analysis(X, y):
    print("Feature Importance Analysis:")
    print("-----------------------------")
    clf = RandomForestClassifier()
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    features = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)
    for feature, importance in features:
        print(f"{feature}: {importance}")

# Perform PCA and print the explained variance of each component
def pca_analysis(X):
    print("\nPCA Analysis:")
    print("-------------")
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    for i, var in enumerate(explained_variance):
        print(f"Principal Component {i + 1}: {var * 100:.2f}%")
    return pca

# Print how much each feature contributes to the principal components
def pca_loadings_analysis(X, pca):
    print("\nPCA Loadings Analysis:")
    print("----------------------")
    components = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC-{i+1}' for i in range(pca.n_components_)]).T
    for i in range(pca.n_components_):
        print(f"\nLoadings for Principal Component {i+1}:")
        sorted_loadings = components[f'PC-{i+1}'].abs().sort_values(ascending=False)
        print(sorted_loadings)

def main():
    # Load and encode the dataset
    data = load_data()
    data, encoders = encode_data(data)

    X = data.drop('class', axis=1)
    y = data['class']

    # Feature importance analysis
    feature_importance_analysis(X, y)

    # PCA analysis
    pca = pca_analysis(X)

    # PCA loadings analysis
    pca_loadings_analysis(X, pca)


if __name__ == "__main__":
    main()
