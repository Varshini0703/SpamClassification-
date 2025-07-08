# 📧 Spam Detection App

A simple yet effective machine learning project that classifies text messages as **spam** or **not spam** using a Multinomial Naive Bayes classifier. Built with `scikit-learn` and deployed using **Streamlit** for real-time predictions.

---

## 🚀 Live Demo

🔗 [Streamlit App](https://spamclassification-git-3wqdqv56uonwx86qxumgk6.streamlit.app/)

---

## 🧠 Project Overview

- **Goal**: Detect whether a given message is spam or not
- **Model**: Multinomial Naive Bayes
- **Vectorization**: CountVectorizer with English stopwords
- **Accuracy**: ~98.6% on test data
- **Deployment**: Streamlit web app

---

## 🗂️ Dataset

- Source: [`spam.csv`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Contains ~5,500 labeled SMS messages
- Columns:
  - `Category`: `spam` or `ham`
  - `Message`: the actual text message

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

---

## 📈 Model Performance

-------------------------------------------------
| Metric     | Score                            |
|------------|----------------------------------|
| Accuracy   | 98.6%                            |
| Vectorizer | CountVectorizer (with stopwords) |
| Classifier | Multinomial Naive Bayes          |
-------------------------------------------------
---

## 🧪 How It Works

1. Load and clean the dataset
2. Split into training and testing sets (80/20)
3. Convert text to numerical features using `CountVectorizer`
4. Train a Naive Bayes classifier
5. Predict and display results in a Streamlit app

---

