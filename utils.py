# utils.py
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Load data
def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

# Determine sentiment
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

# Generate WordCloud
def generate_wordcloud(data, sentiment, colormap):
    text = ' '.join(data[data['sentiment'] == sentiment]['steming_data'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Plot sentiment counts
def plot_sentiment_counts(sentiment_counts, ax):
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=['#0077b6', 'orange', '#C5ADED'])
    ax.set_title("Jumlah Analisis Sentimen", fontsize=14, pad=20)
    ax.set_xlabel("Class Sentiment", fontsize=12)
    ax.set_ylabel("Jumlah", fontsize=12)

# Model comparison results
def model_comparison_results():
    """Train and evaluate models, returning results."""
    data = pd.read_csv("data/Labelling-Lexion.csv").dropna(subset=['steming_data'])
    X = data['steming_data']
    y = data['sentiment']

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=42),
        'MultinomialNB': MultinomialNB()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    return results

# Plot model accuracies
def plot_model_accuracies(results):
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting bars
    bars = ax.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy Comparison", fontsize=16)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)

    # Adding text (percentages) on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f"{height:.2%}",  # Convert to percentage
            ha='center', 
            va='bottom',
            fontsize=12
        )

    return fig

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    """Evaluates the model and returns accuracy, precision, recall, and f1-score."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

def model_comparison_results():
    """Evaluates models and returns their evaluation metrics."""
    results = {}

    # Example: Logistic Regression
    logistic_regression = {
        "y_true": [1, 0, 1, 1, 0],  # Replace with actual labels
        "y_pred": [1, 0, 1, 0, 0],  # Replace with predictions from Logistic Regression
    }
    results["Logistic Regression"] = evaluate_model(logistic_regression["y_true"], logistic_regression["y_pred"])

    # Add other models here (e.g., SVM, Random Forest, etc.)

    return results
