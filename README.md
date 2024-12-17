# Credit-Card-Fraud-Detection-Model

Overview

This repository contains the implementation of a machine learning model designed to detect fraudulent transactions in credit card datasets. The model leverages various supervised learning algorithms to classify transactions as either fraudulent or legitimate.

Features

Preprocessing of unbalanced datasets using techniques such as undersampling, oversampling (SMOTE), or cost-sensitive learning.

Implementation of multiple algorithms, including:

Logistic Regression

Decision Trees

Random Forest

Gradient Boosting (e.g., XGBoost)

Neural Networks

Evaluation using performance metrics such as:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Curve

Cross-validation for robust model performance.

Dataset

The dataset used for this project is sourced from Kaggle and contains 284,807 transactions, of which 492 are fraudulent. It includes features derived from PCA transformations to protect sensitive information.

Prerequisites

Ensure you have the following installed:

Python 3.8 or higher

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

XGBoost

Install dependencies with:

pip install -r requirements.txt

Installation

Clone this repository:

git clone https://github.com/brendon-tk/Credit-Card-Fraud-Detection-Model.git

Navigate to the project directory:

cd Credit-Card-Fraud-Detection-Model

Install dependencies:

pip install -r requirements.txt

Usage

Download and extract the dataset into the data/ directory.

Run the Jupyter notebook or Python script to train and evaluate the model:

Jupyter Notebook:

jupyter notebook Credit_Card_Fraud_Detection.ipynb

Python Script:

python train_model.py

Review results in the output directory, which includes plots and a report summarizing model performance.

Results

The Random Forest model achieved an ROC-AUC score of 0.99 on the validation dataset. Precision and Recall were optimized to handle the imbalanced nature of the dataset.

File Structure

Credit-Card-Fraud-Detection-Model/
|—— data/
|      |__ creditcard.csv
|—— notebooks/
|      |__ Credit_Card_Fraud_Detection.ipynb
|—— scripts/
|      |__ train_model.py
|      |__ preprocess.py
|—— requirements.txt
|—— README.md

Contributions

Contributions are welcome! Feel free to submit issues or pull requests for enhancements.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

The dataset was made available by the Machine Learning Group of ULB.

Inspiration from various Kaggle notebooks and open-source resources.

Contact

If you have any questions or suggestions, feel free to reach out at [brendonmatsikinya@gmail.com].

