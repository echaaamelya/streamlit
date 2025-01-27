import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# app.py (corrected imports)
from utils import (
    determine_sentiment, 
    load_data, 
    generate_wordcloud,
    plot_sentiment_counts,
    train_model,  # This is now properly imported
    plot_class_metrics
)  # Removed unused imports
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data once
data = load_data("data/Labelling-Lexion.csv")
y_test, y_pred = train_model(data)
accuracy = accuracy_score(y_test, y_pred)

st.title("Analisis Sentimen Kebocoran Data Kominfo")

# Input form
st.header("Input Data")
text_input = st.text_area("Masukkan teks untuk dianalisis")
if st.button("Analisis Sentimen"):
    sentiment = determine_sentiment(text_input)
    st.write(f"Sentimen: {sentiment}")

# Visualization Section
st.header("Hasil Analisis Data")
if st.checkbox("Tampilkan Tabel Data"):
    st.dataframe(data.head())

# Sentiment counts bar chart
st.subheader("Distribusi Sentimen (Bar Chart)")
sentiment_counts = data['sentiment'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
plot_sentiment_counts(sentiment_counts, ax)
st.pyplot(fig)

# Pie chart
st.subheader("Distribusi Sentimen (Pie Chart)")
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#0077b6', 'orange', '#C5ADED']
)
ax.set_title("Distribusi Sentimen", fontsize=14, pad=15)
st.pyplot(fig)

# Confusion Matrix Section - FIXED
st.header("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                    index=['positif', 'netral', 'negatif'], 
                    columns=['positif', 'netral', 'negatif'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
st.pyplot(plt)

# Model Accuracy Section - FIXED
st.header("Model Accuracy")
fig, ax = plt.subplots(figsize=(6, 4))
bar = ax.bar(['Logistic Regression'], [accuracy], color='#0077b6')
ax.text(bar[0].get_x() + bar[0].get_width()/2, 
        accuracy, 
        f"{accuracy:.2%}",
        ha='center', 
        va='bottom',
        fontsize=12)
ax.set_ylim(0, 1)
ax.set_title("Model Accuracy", fontsize=14)
ax.set_ylabel("Accuracy Score", fontsize=12)
st.pyplot(fig)
# Train model once and reuse the results


# Create single bar chart
fig, ax = plt.subplots(figsize=(6, 4))
bar = ax.bar(['Logistic Regression'], [accuracy], color='#0077b6')

# Classification Metrics Section - FIXED
st.header("Classification Metrics")
fig = plot_class_metrics(y_test, y_pred)
st.pyplot(fig)

# Remove the redundant "Tampilkan Performa Model" button section