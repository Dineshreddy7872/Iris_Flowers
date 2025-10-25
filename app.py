
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------------------------
# 1️⃣ App Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="🌸 Iris ML Dashboard",
    page_icon="🌺",
    layout="wide"
)
st.title("🌸 Iris Flower Classification & Visualization")
st.markdown("An interactive data science dashboard built with **Streamlit** to explore, visualize, and predict Iris species.")

# ---------------------------------------------------
# 2️⃣ Load Data
# ---------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    df = X.copy()
    df['species'] = pd.Categorical.from_codes(y, iris.target_names)
    return df, iris

df, iris = load_data()

# ---------------------------------------------------
# 3️⃣ Sidebar Navigation
# ---------------------------------------------------
menu = st.sidebar.radio(
    "📊 Choose Section",
    ["🏠 Home", "🔍 Data Exploration", "📈 Visualizations", "🤖 Model Training", "🎯 Predict"]
)

# ---------------------------------------------------
# 🏠 HOME
# ---------------------------------------------------
if menu == "🏠 Home":
    st.subheader("Welcome to the Iris ML Web App 🌼")
    st.markdown("""
    **Project Features:**
    - Interactive EDA and visualizations
    - PCA dimensionality reduction
    - Random Forest-based classification
    - Real-time predictions
    - Clean, modern Streamlit UI
    """)

# ---------------------------------------------------
# 🔍 DATA EXPLORATION
# ---------------------------------------------------
elif menu == "🔍 Data Exploration":
    st.subheader("📋 Dataset Overview")
    st.dataframe(df.head())

    st.markdown("**Summary Statistics**")
    st.write(df.describe())

    st.markdown("**Class Distribution**")
    st.bar_chart(df['species'].value_counts())

# ---------------------------------------------------
# 📈 VISUALIZATIONS
# ---------------------------------------------------
elif menu == "📈 Visualizations":
    st.subheader("🌸 Feature Relationships & Patterns")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pairplot**")
        sns.pairplot(df, hue="species", corner=True)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("**Correlation Heatmap**")
        plt.figure(figsize=(6, 4))
        sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()

    st.markdown("**Boxplots by Feature**")
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=pd.melt(df, id_vars=["species"], value_vars=iris.feature_names),
                x="variable", y="value", hue="species")
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("**PCA Visualization (2D)**")
    X = df.iloc[:, :4]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    X_pca_df["species"] = df["species"]
    sns.scatterplot(data=X_pca_df, x="PC1", y="PC2", hue="species", s=100)
    st.pyplot(plt.gcf())
    plt.clf()

# ---------------------------------------------------
# 🤖 MODEL TRAINING
# ---------------------------------------------------
elif menu == "🤖 Model Training":
    st.subheader("Train and Evaluate a Random Forest Classifier 🌲")

    n_estimators = st.slider("Number of Trees", 10, 300, 100, 10)
    max_depth = st.slider("Maximum Tree Depth", 1, 10, 3)
    test_size = st.slider("Test Size (Proportion)", 0.1, 0.5, 0.2)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

    X = df.iloc[:, :4]
    y = df['species'].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully! Accuracy = **{acc:.3f}**")

    st.markdown("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    st.pyplot(fig)

    st.markdown("**Classification Report:**")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Save trained model
    joblib.dump(model, "iris_rf_model.joblib")

# ---------------------------------------------------
# 🎯 PREDICT
# ---------------------------------------------------
elif menu == "🎯 Predict":
    st.subheader("🌼 Predict Iris Flower Species")

    st.markdown("Enter flower measurements below:")
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.3)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Load trained model or train a new one if not present
    try:
        model = joblib.load("iris_rf_model.joblib")
    except:
        model = RandomForestClassifier(random_state=42).fit(df.iloc[:, :4], df['species'].astype('category').cat.codes)

    pred_idx = model.predict(input_data)[0]
    pred_species = iris.target_names[pred_idx]

    st.success(f"🌸 Predicted Species: **{pred_species.capitalize()}**")

    # Feature importance
    st.markdown("**Feature Importance:**")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": iris.feature_names, "Importance": importances})
    sns.barplot(x="Importance", y="Feature", data=imp_df.sort_values("Importance", ascending=False))
    st.pyplot(plt.gcf())
    plt.clf()
