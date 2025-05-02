The goal of this project given in class was to build an interactive machine learning experience for the user. The app should allow users to upload a dataset, experiment with hyperparameters, and observe how these affect model training and performance. The final result should be an intuitive interface that invites exploration and communicates results clearly.

This project meets these requirements; my streamlit app allows users to: upload their own dataset in a csv format, select a target column to investigate, choose a supervised learning model between the options of a logistic regression, a decision tree, and K nearest neighbor, tune hyperparameters, and finally to view performance metrics alongside contextual descriptions of what those numbers mean. The app includes a classification report, accuracy score, and ROC curve (for binary classifications). 

This interactive app supports training and evaluation of three popular supervised machine learning models:

### 1. Logistic Regression
- **Use case:** binary classification (ex: yes/no, success/failure).
- **Hyperparameter tunable:**  
  - `C` (inverse of regularization strength) - a lower value increases regularization to prevent overfitting.
- Feature scaling recommended for this model.

### 2. Decision Tree Classifier
- **Use case:** flexible for both classification and regression.
- **Hyperparameter tunable:**  
  - `max_depth` — limits the tree’s depth to prevent overfitting.
- No scaling needed; it handles both numeric and categorical features well.

### 3. K-Nearest Neighbors (KNN)
- **Use case:** Classification based on similarity to nearby points.
- **Hyperparameter tunable:**  
  - `n_neighbors` (k) — number of neighbors used to determine the class.
- Feature scaling highly recommended since KNN relies on distance calculations.

**Deployment link:** https://jennings-data-science-portfolio-gc98vvzgmmwkeoc86bpdtr.streamlit.app/ 
