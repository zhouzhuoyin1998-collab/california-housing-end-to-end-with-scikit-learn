"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
import pandas as pd
import tempfile
import tarfile
import urllib.request
from pathlib import Path

def load_housing():

    # Step 1: Define the download URL
    url = "https://github.com/ageron/data/raw/main/housing.tgz"

    # Step 2: Define the local file path
    path = Path(tempfile.gettempdir()) / "housing.tgz"

    # Step 3: Download only if the file doesn't exist
    if not path.exists():
        urllib.request.urlretrieve(url, path)

    # Step 4: Open the compressed archive
    with tarfile.open(path, "r:gz") as tar:

        # Step 5: Read CSV directly from the archive
        with tar.extractfile("housing/housing.csv") as file:
            housing = pd.read_csv(file)

    return housing

# Step 2 - income_categories
def income_categories(df):
    # TODO: pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    return pd.cut(
        df["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split

def stratified_split(df, test_size=0.2, random_state=42):

    train_set, test_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=income_categories(df)
    )

    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    # TODO: Pearson correlation of every numeric column with median_house_value, sorted descending, target excluded.
    corr_matrix = df.corr(numeric_only=True)

    correlations = corr_matrix["median_house_value"]

    correlations = correlations.drop("median_house_value")

    return correlations.sort_values(ascending=False)

# Step 5 - add_ratio_features
def add_ratio_features(df):
    # TODO: Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    result = df.copy()

    result["rooms_per_house"] = (
        result["total_rooms"] / result["households"]
    )

    result["bedrooms_ratio"] = (
        result["total_bedrooms"] / result["total_rooms"]
    )

    result["people_per_house"] = (
        result["population"] / result["households"]
    )

    return result

# Step 6 - split_features_labels
def split_features_labels(df):
    # TODO: Return (X without median_house_value, y = median_house_value Series).
    X = df.drop(columns=["median_house_value"])

    y = df["median_house_value"].copy()

    return X, y

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # TODO: store the parameters
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )

        self.kmeans_.fit(X, sample_weight=sample_weight)

        return self

    def transform(self, X):

        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):

        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]

# Step 8 - numeric_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

# Step 9 - categorical_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(
            missing_values=None,
            strategy="most_frequent"
        ),
        OneHotEncoder(handle_unknown="ignore")
    )

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

