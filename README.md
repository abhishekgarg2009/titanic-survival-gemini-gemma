# Titanic Survival Prediction using Gemini and Gemma

## Description
This repository contains code for predicting survival on the Titanic using Google's Gemini model. The initial solution provides the entire training dataset as context for Gemini to predict the survival of each passenger in the test set. Future development will include fine-tuning Gemma 2 and using it for predictions.

## Data
1. Download the `train.csv` and `test.csv` files from the Kaggle Titanic competition: [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
2. Place the downloaded files in the same directory as the Python script.

## Environment Variables
Set the `GEMINI_API_KEY` environment variable with your Google Gemini API key.

## Usage
Run the `predict_gemini.py` script to generate a `submission.csv` file with the survival predictions.

## Future Work
* Fine-tune Gemma 2 on the Titanic dataset.
* Use the fine-tuned Gemma 2 model to make predictions.
