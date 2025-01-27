import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import determine_sentiment, load_data, generate_wordcloud, plot_sentiment_counts, plot_model_accuracies, model_comparison_results

# Load data
data = load_data("data/Hasil-Preprocessing.csv")

# Tambahkan kolom sentiment jika belum ada
if 'sentiment' not in data.columns:
    data['sentiment'] = data['steming_data'].apply(determine_sentiment)

st.title("Analisis Sentimen dengan Streamlit")

# Input form
st.header("Input Data Baru")
text_input = st.text_area("Masukkan teks untuk dianalisis")
if st.button("Analisis Sentimen"):
    sentiment = determine_sentiment(text_input)
    st.write(f"Sentimen: {sentiment}")

# Visualisasi
st.header("Hasil Analisis Data")
if st.checkbox("Tampilkan Tabel Data"):
    st.dataframe(data.head())

# Sentiment counts visualization
sentiment_counts = data['sentiment'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
plot_sentiment_counts(sentiment_counts, ax)
st.pyplot(fig)  # Explicitly passing the figure

# Word Cloud
st.header("Word Cloud")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Positif")
    fig = generate_wordcloud(data, 'positif', 'Greens')
    st.pyplot(fig)

with col2:
    st.subheader("Negatif")
    fig = generate_wordcloud(data, 'negatif', 'Reds')
    st.pyplot(fig)

with col3:
    st.subheader("Netral")
    fig = generate_wordcloud(data, 'netral', 'Blues')
    st.pyplot(fig)

# Model comparison
st.header("Perbandingan Model")
results = model_comparison_results()

# Plot and display the model comparison with labels
fig = plot_model_accuracies(results)
st.pyplot(fig)

# Model comparison
results = model_comparison_results()

# Menampilkan metrik evaluasi Logistic Regression
st.subheader("Hasil Logistic Regression")
logistic_metrics = results.get("Logistic Regression", {})
st.write("Akurasi:", f"{logistic_metrics.get('accuracy', 0):.2%}")
st.write("Presisi:", f"{logistic_metrics.get('precision', 0):.2%}")
st.write("Recall:", f"{logistic_metrics.get('recall', 0):.2%}")
st.write("F1-Score:", f"{logistic_metrics.get('f1_score', 0):.2%}")

