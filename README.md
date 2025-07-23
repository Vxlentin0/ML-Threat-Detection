# Threat-Detection
This Python-based Threat Detection System uses machine learning to identify and classify potential threats within structured datasets. It is designed as a modular, extensible framework that supports both basic and advanced threat detection using supervised and unsupervised learning techniques.


🔍 What It Does
=================

The system ingests a dataset containing labeled examples of both benign and malicious activity (e.g., network events, login attempts), then preprocesses the data, trains a deep learning model, and evaluates its performance in detecting threats.

It supports:

Binary classification (threat vs. no threat)

Model training and evaluation

Live prediction on new data samples

Future extensions for anomaly detection and explainability


⚙️ How It Works
===================

**Data Preprocessing**

The system loads a CSV dataset containing features and an is_threat label (0 or 1).

Features are scaled using StandardScaler to ensure consistent input to the neural network.

The dataset is split into training and testing sets (80/20 split by default).

**Model Training**

A Multi-Layer Perceptron (MLP) model is built using Keras.

The model consists of dense layers with dropout regularization to prevent overfitting.

It's trained using binary cross-entropy loss and evaluated using accuracy, precision, recall, and F1-score.

**Evaluation & Reporting**

Once trained, the model outputs performance metrics including a classification report and a confusion matrix.

Predictions are tested on held-out test data to evaluate real-world performance.

**Live Detection**

New data samples can be fed into the system.

The system scales the input and returns a binary threat prediction along with a confidence score.


▶️ How to Run
===============

Run this command in the terminal to downlaod the dependancies the program needs:

**pip install -r requirements.txt**

To run the program enter this command:

**python scripts/threat_detection.py**
