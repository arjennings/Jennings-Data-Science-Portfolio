import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#title app
st.title("Interactive ML App: Supervised Learning")

#file upload
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"]) #this allows the user to be interactive from the beginning, uploading their chosen csv file

if uploaded_file:
    df = pd.read_csv(uploaded_file) #now we read the uploaded csv into the rest of the program
    st.write("Data Preview:", df.head()) #getting an overview of the csv file to situate the user in the data
    
    target_col = st.selectbox("Select Target Column", df.columns) #this is also an interactivity step, allowing the user to choose what aspect of the data to focus on

#going through some data cleaning steps -- I ran the entire code before going through thorough cleaning steps and streamlit was having a hard time handling it. Not only does this make the data look nicer and more digestible, it also ensures the model runs smoothly
df = df.dropna(subset=[target_col])  #drop rows where target is missing

#remove columns with more than 50% missing values
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

#fill remaining missing values in numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#drop unique non-numeric columns (like names, IDs)
high_card_cols = [col for col in df.select_dtypes(include='object').columns
        if df[col].nunique() > 20]
df = df.drop(columns=high_card_cols)

#encode remaining categorical columns
df = pd.get_dummies(df, drop_first=True)

#redefine target and features
y = df[target_col]
X = df.drop(columns=[target_col])

#split data
test_size = st.slider("Test set size (%)", 10, 50, 20) / 100 #this allows the user to choose the size of the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

#optional scaling
scale = st.checkbox("Scale features (StandardScaler)") #this allows the user to choose whether or not to scale the features
if scale:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

#model selection
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"]) #this allows the user to choose which type of model to display in the app

#hyperparameters
if model_choice == "Logistic Regression":
    c = st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=c, max_iter=1000)
elif model_choice == "Decision Tree":
    max_depth = st.slider("Max depth", 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=max_depth)
elif model_choice == "K-Nearest Neighbors":
    k = st.slider("Number of neighbors (k)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

#train model
if st.button("Train Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Performance metrics: ")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")
    st.markdown("""Accuracy measures the proportion of correct predictions out of all predictions made. 
        It’s a good general metric, but it can be misleading if the data is imbalanced.
        """)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.markdown("""
            - **Precision:** out of the predicted positives, how many were actually positive?
            - **Recall:** out of all actual positives, how many did the model correctly find?
            - **F1 Score:** the harmonic mean of precision and recall — this is often a more holistic measurement as it balances the two.
        """)


    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    st.markdown("""
        Each cell shows how many examples fall into each category:
        - True positives and true negatives: Correct predictions
        - False positives and false negatives: Incorrect predictions
        """)

#ROC Curve (only for binary classification)
if len(np.unique(y_test)) == 2:
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    st.write(f"AUC Score: {auc_score:.2f}")
    st.write("ROC Curve:")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle='--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("""
            The ROC Curve shows the tradeoff between true positive rate and false positive rate at various thresholds.
            - A good model is concentrated in the top-left corner.
            - A random model follows the diagonal.

            The AUC summarizes this curve into a single number between 0 and 1:
            - 1.0 = perfect model
            - 0.5 or less = no better than random
            """)
else:
    st.info("ROC Curve & AUC Score are available only for binary classification problems.")