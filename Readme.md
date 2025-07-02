# 📝 Action Items Classification API

This project provides an API for detecting **action items** in textual content using machine learning and deep learning models. It supports both **Logistic Regression** and **BERT-based classifiers**, and exposes web-based and programmatic access.

---

## 📁 Project Structure

### 1. `wsgi.py`
- Entry point for running the API with **Gunicorn**.
- Imports `app_class_Machine_learning_classifier.py` to serve the application.

---

### 2. `app_class_Machine_learning_classifier.py`

This is the main API class file containing `ActionItemsAPI`. The class includes the following methods:

#### ▸ `sentence_prediction()`
- Takes a sentence as input.
- Validates that the sentence length is greater than 18 characters.
- Returns `0` (non-action) or `1` (action item).

#### ▸ `action_items()`
- Handles POST requests with `.txt` files.
- Reads the file content line-by-line and sends each sentence to `sentence_prediction()` for classification.

#### ▸ `get(self)`
- Serves the Action Items web interface (UI).

#### ▸ `post(self)`
- Accepts text input from the UI.
- Classifies the input using `sentence_prediction()` and displays the result on the page.

---

### 3. `app_class_Bert_model.py`

- This file provides the same functionality as `app_class_Machine_learning_classifier.py` but utilizes a **BERT-based model** for more accurate classification.
- Designed for higher performance in semantic understanding.

---

## 📁 Folder Structure

- `recieved_text_files/`  
  Stores uploaded `.txt` files submitted by users.

- `bert/`  
  Contains BERT-based model code, tokenizer, weights, and configurations.

- `logistic-regression/`  
  Contains Logistic Regression model code, trained model file, and supporting scripts.

---

## ✅ Features

- Accepts plain text or individual sentences.
- Fast prediction using logistic regression.
- Deep contextual prediction with BERT.
- Web-based form UI for quick testing.
- Ready for deployment via Gunicorn.

---

## 🚀 Usage

```bash
gunicorn wsgi:app

```
Visit ```bash http://localhost:8000 ``` to access the web interface.
## 🧠 Tech Stack

- **Python 3.8+**
- **Flask**
- **Scikit-learn** (Logistic Regression)
- **HuggingFace Transformers** (BERT)
- **Gunicorn** (for deployment)

---

## 📌 Notes

- Ensure all model files are correctly placed in their respective folders:
  - `bert/`
  - `logistic-regression/`
- Input sentences must be **longer than 18 characters** to avoid false predictions.

