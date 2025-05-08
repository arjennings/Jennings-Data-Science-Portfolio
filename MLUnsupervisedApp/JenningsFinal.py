
#installing kagglehub so I can import the data to my notebook
#the below line is specific to Jupyter notebooks, use it in terminal if you are not in this environment
#%pip install kagglehub
import kagglehub #importing kagglehub to use its functions

#downloading latests version of the dataset from kagglehub
path = kagglehub.dataset_download("subhamjain/loan-prediction-based-on-customer-behavior") #setting it equal to path to recall more easily later

#displaying the path to the downloaded dataset
print("Path to dataset: ", path) #printing the path to the dataset so that my output is easily readable

import streamlit as st #importing streamlit to create a web app later
import pandas as pd #importing pandas to work with the dataset
import numpy as np #importing numpy to work with arrays and numerical data
import matplotlib.pyplot as plt #importing matplotlib to create visualizations
import seaborn as sns #importing seaborn to create more advanced visualizations
import os #importing os to work with the file system

from sklearn.preprocessing import StandardScaler #this will be used later for centering and scaling. PCA is sensitive to the scale, so this step is essential
    #Centering ensures that each feature has a mean of zero, and scaling ensures that each feature has unit variance.
    #This prevents features with larger numerical ranges from dominating the PCA results.
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#title and description for the app to introduce the user to the functionality. Because this is unsupervised machine learning, it is inherently more exploratory. 
st.title("Interactive Unsupervised ML App: Exploring Clusters in Customer Behavior")
st.write("Upload or select a dataset and explore clustering using PCA, KMeans, and Hierarchical Clustering.")

#laying out step one for the user to interact with the model
st.sidebar.header("1. Upload or Select Dataset")

#creating a dataset from the csv
data_path = os.path.join(path, "Test Data.csv")
sample_dataset = {
    "Test Dataset": pd.read_csv(https://github.com/arjennings/Jennings-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/Test%20Data.csv)
}
#easy select feature for my dataset or their own upload
dataset_option = st.sidebar.selectbox("Choose my dataset", options=["Upload your own"] + list(sample_dataset.keys()))

#if else loop to check if the user has uploaded a file or selected a sample dataset and proceed accordingly
if dataset_option == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) #these chunk of the loop runs through the same steps as I did above for my dataset
    else:
        st.warning("Please upload a CSV to continue.")
        st.stop()
else:
    df = sample_dataset[dataset_option] #if the user is not uploading a file, they can select the sample dataset

st.write("Here is an overview of your selected data:") #prefacing the table with a description to let the user know their choice is being shown
st.dataframe(df.head()) #displaying the first few rows of the DataFrame to see if the upload worked

st.sidebar.header("2. Select Features for Clustering") #now we move to step two, continuing to let the user choose their features no matter what dataset they uploaded

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() #numeric columns are selected for PCA and clustering
st.sidebar.markdown("### Choose numeric features for PCA and KMeans clustering:")
#this is a sidebar header that will be displayed on the left side of the app
#this is a multiselect box that allows the user to select multiple features at once (this will also be displayed to the side)
selected_features = st.sidebar.multiselect(
    "Choose numeric features to include",
    options=numeric_cols,
    default=numeric_cols[:3]  #preselect first 3
)

st.subheader("Preview of Selected Features") #so that the user can see what's happening behind the scenes, the app will now display a preview of what the user selected
st.write(df[selected_features].head())

#PCA and clustering really only works with two or more than two features, otherwise it is redundant, so I will add a warning if the user selects less than two features
if len(selected_features) < 2:
    st.warning("Please select at least 2 features for meaningful PCA and clustering.")
    st.stop()

X = df[selected_features].dropna() #dropping any rows with missing values in the selected features to clean the data
#this is important because PCA and clustering algorithms require complete data to function properly


#In this next step, I will use PCA to reduce the dimensionality of the dataset and make it easier to use

#ask users for input on choosing the number of PCA components and K-Means clusters
st.sidebar.markdown("### Choose the number of dimensions (PCA) and clusters (KMeans):")
#this is a sidebar header that will be displayed on the left side of the app
st.sidebar.header("3. Set Model Parameters")

# PCA Components
n_components = st.sidebar.slider("Number of PCA components", 2, min(2, 10), 10)
#by having a slider the user is able to easily visualize the number of components they are selecting

# K-Means Clusters
n_clusters = st.sidebar.slider("Number of K-Means clusters", 2, 10, 3)
#the slider is effective in the same way for selecting the number of clusters

if X.shape[0] < n_clusters:
    st.error(f"Not enough data points ({X.shape[0]}) for {n_clusters} clusters. Reduce the number of clusters or use a different dataset.")
    st.stop()

if X.shape[1] < n_components:
    st.error(f"Dataset only has {X.shape[1]} numeric features—reduce the number of PCA components.")
    st.stop()

#actually scaling the data
scaler = StandardScaler() #creating an instance of the StandardScaler
X_scaled = scaler.fit_transform(X) #fitting the scaler to the data and transforming it
#the fit_transform method computes the mean and standard deviation for each feature and then scales the data accordingly.

#reducing the data to 2 components for visualization and further analysis.
pca = PCA(n_components=n_components) #creating an instance of the PCA class with the number of components specified by the user, rather than manually selecting it ahead of time
X_pca = pca.fit_transform(X_scaled)

#displaying the Explained Variance Ratio. This tells us the proportion of the variance that is explained by each of the selected components.
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
print("Cumulative Explained Variance:", np.cumsum(explained_variance))

#in this next step I will offer options for clustering
st.sidebar.header("Step 3: Choose Clustering Technique")
model_option = st.sidebar.radio("Clustering Method", ["KMeans", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

if model_option == "KMeans":
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca) #fitting the KMeans model to the PCA-transformed data and predicting the cluster labels
#the fit_predict method combines the fit and predict steps into one, making it more efficient for large datasets.   
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    st.write(f"Silhouette Score for KMeans: {silhouette:.2f}")
    st.markdown("""The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
A score close to 1 indicates that the object is well clustered, while a score close to -1 indicates that the object may have been assigned to the wrong cluster.
A score of 0 indicates that the object is on or very close to the decision boundary between two neighboring clusters.
""")

else:
    # Hierarchical Clustering
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    Z = linkage(X_scaled, method=linkage_method)
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

    # Show dendrogram
    st.subheader("Dendrogram (Hierarchical Clustering)")
    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 4))
    dendrogram(Z, ax=ax_dendro)
    st.pyplot(fig_dendro)
    st.markdown("""
### What Does This Dendrogram Show?

The dendrogram visualizes the results of hierarchical clustering. Each merge in the plot represents a cluster combination, with the vertical axis showing the distance between clusters when they were merged.
- The x-axis shows individual data points (or small clusters).
- The height of the U-shaped lines indicates how dissimilar the clusters being merged are.
- You can interpret the number of clusters by drawing a horizontal line—each intersection represents a distinct cluster.
""")

df_clustered = pd.DataFrame(X_pca[:, :2], columns=["PCA1", "PCA2"])
df_clustered['Cluster'] = [f"Cluster {label}" for label in clusters]

st.subheader("Cluster Visualization (First Two PCA Components)")
fig_pca, ax_pca = plt.subplots()
sns.scatterplot(
    data=df_clustered,
    x="PCA1", 
    y="PCA2",
    hue="Cluster",
    palette="tab10",
    ax=ax_pca
)
ax_pca.set_title("PCA Cluster Scatterplot")
ax_pca.legend(title="Cluster Group", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig_pca)

# Explanation of the plot
st.markdown("""
### What Does This Graph Show?

This scatterplot shows how your data points are grouped after applying PCA (Principal Component Analysis) and clustering.
- The x- and y-axes represent the first two principal components, which capture the most significant variation in the data.
- Each point represents a customer (or row in your dataset), and the color shows which cluster it was assigned to.
- Clusters that are tightly grouped and clearly separated from others indicate good clustering performance.
""")

#the elbow plot is a good visualization to discern the optimal number of clusters, so I wanted to include that option
st.sidebar.header("Optional: Elbow Plot (for KMeans)")
if st.sidebar.button("Generate Elbow Plot"):
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_pca)
        distortions.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K_range, distortions, marker='o')
    ax_elbow.set_title("Elbow Method")
    ax_elbow.set_xlabel("Number of clusters")
    ax_elbow.set_ylabel("Inertia (Within-cluster sum of squares)")
    st.pyplot(fig_elbow)
    st.markdown("""
### What is the Elbow Plot?

The Elbow Plot helps determine the optimal number of clusters for K-Means clustering. 
It shows how the model's inertia (sum of squared distances to the nearest cluster center) decreases as the number of clusters increases.

Initially, adding more clusters significantly improves the model (reduces inertia), but after a certain point the benefit drops off—this "elbow" (point of diminishing returns) is a good indication of the best number of clusters to use. If you go beyond the elbow and choose too many clusters, the model may overfit the data and become less generalizable.
""")
#the elbow plot also will inform the reader of ideal number of clusters to use and then they can go back and re-adjust the parameters to get a better model




