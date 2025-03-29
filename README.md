# Titanic Survival Prediction using Gemini and Gemma

## Description
This repository contains code for predicting survival on the Titanic using Google's Gemini model. The initial solution provides the entire training dataset as context for Gemini to predict the survival of each passenger in the test set. Future development will include fine-tuning Gemma 2 and using it for predictions.

## Data
1. Download the `train.csv` and `test.csv` files from the Kaggle Titanic competition: [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
2. Place the downloaded files in the same directory as the Python script.

## Environment Variables
Set the `GEMINI_API_KEY` environment variable with your Google Gemini API key.

## Usage
The `predict_gemini.py` script works by converting the Titanic survival prediction problem into a text completion task. For each passenger in the test set, the script provides the entire training dataset as context to the Gemini model ("gemini-2.0-flash"). The model then predicts whether the passenger survived or not.

To generate a `submission.csv` file with the survival predictions, run the script:
```bash
python predict_gemini.py
```

## Performance
The Gemini-based prediction (`predict_gemini.py`) achieved a Kaggle submission score of **0.78947**. This result outperforms the official Kaggle Titanic tutorial's score of 0.77511, which was achieved using a Random Forest classifier.

## Future Work
* Fine-tune Gemma 2 on the Titanic dataset.
* Use the fine-tuned Gemma 2 model to make predictions.
