# SpamGuard – AI-Powered SMS Spam Detection

SpamGuard is a machine learning-based SMS spam detection application that classifies text messages as **Spam** or **Ham (Normal)**.

It uses Natural Language Processing (NLP), TF-IDF feature extraction, and Multinomial Naive Bayes to identify suspicious messages.

The project also provides an interactive web interface built using Streamlit.

## Features

- SMS spam and ham classification.
- Text preprocessing using regular expressions.
- TF-IDF feature extraction.
- Multinomial Naive Bayes classification.
- Probability-based predictions.
- Validation-based classification threshold selection.
- Model evaluation using accuracy, precision, recall, and F1-score.
- Interactive Streamlit web interface.

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Matplotlib
- Joblib
- Streamlit
- Natural Language Processing (NLP)

## Machine Learning Workflow

1. Dataset collection and exploration.
2. Data validation and duplicate removal.
3. Text preprocessing.
4. Train-test splitting.
5. TF-IDF vectorization.
6. Multinomial Naive Bayes model training.
7. Classification threshold optimization using validation data.
8. Model evaluation.
9. Model serialization using Joblib.
10. Streamlit application development.

## Model Performance

Final evaluation on the held-out test dataset:

| Metric | Ham | Spam |
|---|---:|---:|
| Precision | 98% | 93% |
| Recall | 99% | 85% |
| F1-score | 99% | 89% |

**Overall Accuracy: 97%**

The classification threshold was selected using validation data rather than the final test set.

## Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains SMS messages labelled as spam or ham.

## Project Structure

```text
SpamGuard/
│
├── dataset/
│   └── SMSSpamCollection
│
├── notebooks/
│   └── SpamGuard_ML.ipynb
│
├── model/
│   ├── spam_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── spam_threshold.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd SpamGuard
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Limitations

- The model relies on statistical word patterns rather than contextual language understanding.
- Performance depends on the quality and diversity of the training dataset.
- Unfamiliar spam messages may be incorrectly classified.
- Predicted probabilities are model estimates, not guarantees of message safety.

## Future Improvements

- Experiment with Logistic Regression and Support Vector Machines.
- Expand the training dataset with more diverse spam examples.
- Explore advanced NLP techniques.
- Improve detection of previously unseen spam messages.
- Deploy the application online.

## Author

Developed as a machine learning and NLP portfolio project.