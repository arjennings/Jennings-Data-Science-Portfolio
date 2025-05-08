### Project Overview ###

This interactive Streamlit app enables users to explore patterns in customer data through unsupervised machine learning techniques. Because it is unsupervised, this app emphasizes exploration and trial and error. Users can upload their own datasets or use a sample dataset and then choose numeric features, perform dimensionality reduction with PCA, and apply clustering methods (KMeans or Hierarchical) to identify meaningful groups.

### Instructions for Running the App ###
## 1. Clone the Repository: git clone https://github.com/arjennings/unsupervised-ml-streamlit.git
## 2. Set up Environment
## 3. Install requirements listed in the requirements.txt
## 4. Run the App
## 5. Access the app locally after it launches to see the visualizations


### Summary of App Features

### Dataset Handling
- Upload your own .csv file or use a built-in test dataset.
- Automatically detects numeric features for analysis.
- Preview dataset and selected features.

### PCA (Principal Component Analysis)
- Reduce dimensionality to make high-dimensional data interpretable.
- Adjustable number of components via slider.

### Clustering Methods
- KMeans Clustering:
  - Choose the number of clusters (2–10).
  - View silhouette score to evaluate model performance.
  - Optional Elbow Plot to determine optimal cluster count.
- Hierarchical Clustering:
  - Choose linkage method (ward, complete, average, single).
  - Visual dendrogram of the clustering process.

### Visualization
- Cluster scatterplot (colored by group).
- Dendrogram for hierarchical clustering.
- Elbow plot for KMeans.
- Informative descriptions below each chart to aid interpretation.



