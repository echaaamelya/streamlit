import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath).dropna(subset=['steming_data'])

def determine_sentiment(text):
    positive_lexion = set(pd.read_csv("data/positive.tsv", sep="\t", header=None)[0])
    negative_lexion = set(pd.read_csv("data/negative.tsv", sep="\t", header=None)[0])

    positive_count = sum(1 for word in text.split() if word in positive_lexion)
    negative_count = sum(1 for word in text.split() if word in negative_lexion)

    if positive_count > negative_count:
        return "positif"
    elif positive_count < negative_count:
        return "negatif"
    else:
        return "netral"

def generate_wordcloud(data, sentiment, colormap):
    text = ' '.join(data[data['sentiment'] == sentiment]['steming_data'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_sentiment_counts(sentiment_counts, ax):
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=['#0077b6', 'orange', '#C5ADED'])
    ax.set_title("Jumlah Analisis Sentimen", fontsize=14, pad=20)
    ax.set_xlabel("Kelas Sentimen", fontsize=12)
    ax.set_ylabel("Jumlah", fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom', fontsize=12)

def plot_class_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    
    classes = ['positif', 'netral', 'negatif']
    metrics = {
        'Precision': [report[cls]['precision'] for cls in classes],
        'Recall': [report[cls]['recall'] for cls in classes],
        'F1-score': [report[cls]['f1-score'] for cls in classes]
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes)) 
    width = 0.25  
    
    # matrix
    bars = []
    for i, (metric_name, values) in enumerate(metrics.items()):
        bar = ax.bar(x + (i * width), values, width, label=metric_name)
        bars.append(bar)
        
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
            
    ax.set_title('Klasifikasi Matrix per kelas', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)
    
    return fig

def train_model(data): 
    X = data['steming_data']
    y = data['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    return y_test, y_pred
