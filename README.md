# Titanic Survival Prediction using Gemini and Gemma

## Description
This repository contains code for predicting survival on the Titanic using Google's Gemini and Gemma models.

## Data
1. Download the `train.csv` and `test.csv` files from the Kaggle Titanic competition: [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
2. Place the downloaded files in the same directory as the Python scripts.

## Gemini

### Environment Variables
Set the `GEMINI_API_KEY` environment variable with your Google Gemini API key.

### Usage
The `predict_gemini.py` script works by converting the Titanic survival prediction problem into a text completion task. For each passenger in the test set, the script provides the entire training dataset (`train.csv`) as context to the Gemini model ("gemini-2.0-flash"). The model then predicts whether the passenger survived or not based on this context.

To generate a `submission_gemini.csv` file with the survival predictions, run the script:
```bash
python predict_gemini.py
```

Additionally, the `predict_gemini_thinking.py` script uses the experimental model "gemini-2.0-flash-thinking-exp-01-21". It follows the same logic as `predict_gemini.py` but generates the submission file `submission_gemini_thinking.csv`.

```bash
python predict_gemini_thinking.py
```

### Performance
- The `predict_gemini.py` script (using "gemini-2.0-flash") achieved a Kaggle submission score of **0.78947**. This result outperforms the official Kaggle Titanic tutorial's score of 0.77511, which was achieved using a Random Forest classifier.
- The `predict_gemini_thinking.py` script (using "gemini-2.0-flash-thinking-exp-01-21") achieved a Kaggle submission score of **0.77272**.

## Gemma

### Usage

#### 1. Fine-tuning Gemma-2 2b
The `gemma_2_fine_tune.py` script fine-tunes the `google/gemma-2-2b` model on the `train.csv` dataset using LoRA. The fine-tuning approach is based on a notebook published on Hugging Face: [https://www.kaggle.com/code/heidichoco/gemma-fine-tuning-for-beginners-with-huggingface](https://www.kaggle.com/code/heidichoco/gemma-fine-tuning-for-beginners-with-huggingface).

**Prerequisites:**
*   Ensure you have the necessary libraries installed (`transformers`, `trl`, `peft`, `datasets`, `torch`, `bitsandbytes`, `accelerate`).
*   Ensure `train.csv` is in the same directory.
*   A CUDA-enabled GPU is recommended for faster training.

Run the fine-tuning script:
```bash
python gemma_2_fine_tune.py
```
This will save the fine-tuned model adapter weights to the `outputs_gemma2_base` directory.

**Note on Ignoring Empty Features During Training:** To focus the model's training only on the features present for each passenger, the `gemma_2_fine_tune_ignore_empty.py` script was introduced. This script modifies the prompt generation during fine-tuning to exclude features with missing values (e.g., `NaN` or `None`). This prevents the model from being trained on placeholder values for missing data, potentially improving focus on relevant information. Use this script if you want to train a model that specifically ignores absent features during the training process.

```bash
python gemma_2_fine_tune_ignore_empty.py
```
This will save the fine-tuned model adapter weights to a directory like `outputs_gemma2_base_eos_ignore_empty_X` (depending on the `output_dir` set in the script).

**Note:** To use the instruction-tuned version of Gemma-2 2b (`gemma-2-2b-it`) instead of the base model, modify the `model_id` variable in `gemma_2_fine_tune.py` before running the script:
```python
# Change this line in gemma_2_fine_tune.py
model_id="google/gemma-2-2b-it" 
```
Remember to adjust the `output_dir` in `load_gemma_fine_tuned.py` accordingly if you change the output directory name in `gemma_2_fine_tune.py` when using the instruction-tuned model.

**Note on EOS Token:** Initial fine-tuning runs sometimes resulted in the model generating extra text beyond the desired '0' or '1' prediction. To address this and focus the model's output, the `gemma_2_fine_tune_eos.py` script was created. This script explicitly appends the End-of-Sequence (`<eos>`) token to the training examples in the `formatting_func`. Both `gemma_2_fine_tune_eos.py` and `gemma_2_fine_tune_ignore_empty.py` incorporate this EOS token logic.

#### 2. Generating Predictions with Fine-tuned Gemma
- The `load_gemma_fine_tuned.py` script loads a fine-tuned Gemma model adapter (trained with `gemma_2_fine_tune.py` or `gemma_2_fine_tune_eos.py`) and uses it to predict survival for passengers in `test.csv`. It includes all features in the prompt, even those with missing values.
- The `load_gemma_fine_tuned_ignore_empty.py` script also loads a fine-tuned model adapter but specifically ignores features with empty values when generating the prompt *during inference*. This can be used with models trained using any of the fine-tuning scripts (`gemma_2_fine_tune.py`, `gemma_2_fine_tune_eos.py`, or `gemma_2_fine_tune_ignore_empty.py`) if you want to exclude missing features only at prediction time. For models trained with `gemma_2_fine_tune_ignore_empty.py`, using this inference script ensures consistency between training and prediction.

**Prerequisites:**
*   Ensure `test.csv` is in the same directory.
*   Ensure the fine-tuned model checkpoint exists (e.g., in `outputs_gemma2_base/checkpoint-XXX` or `outputs_gemma2_base_eos/checkpoint-XXX` after running the relevant fine-tuning script). **Note:** You will need to adjust the `output_dir` variable within `load_gemma_fine_tuned.py` to point to the specific checkpoint directory you want to use for predictions (e.g., `outputs_gemma2_base_eos/checkpoint-512`).

Run the prediction script (after setting the correct `output_dir` in the script):
```bash
python load_gemma_fine_tuned.py
```
This will generate a submission file (e.g., `submission_gemma_base_eos_512.csv`, depending on how you name it in the script) with the predictions.

### Performance

The following table summarizes the Kaggle submission scores achieved with different fine-tuned Gemma-2 2b configurations:

| Model Configuration                        | Fine-tuning Script                 | Training Steps | Kaggle Score |
| :----------------------------------------- | :--------------------------------- | :------------: | :----------: |
| Gemma-2 2b (Base)                          | `gemma_2_fine_tune.py`             |      128       | **0.77751**  |
| Gemma-2 2b (Instruction-Tuned)             | `gemma_2_fine_tune.py`             |      128       | **0.78468**  |
| Gemma-2 2b (Base + EOS)                    | `gemma_2_fine_tune_eos.py`         |      128       | **0.77751**  |
| Gemma-2 2b (Base + EOS)                    | `gemma_2_fine_tune_eos.py`         |      256       | **0.77511**  |
| Gemma-2 2b (Base + EOS)                    | `gemma_2_fine_tune_eos.py`         |      512       | **0.78947**  |
| Gemma-2 2b (Base + EOS)                    | `gemma_2_fine_tune_eos.py`         |     1024       | **0.78229**  |
| Gemma-2 2b (Base + EOS + Ignore Empty Inference Only)| `gemma_2_fine_tune_eos.py`         |     1024       | **0.78468**  |
| Gemma-2 2b (Base + EOS + Ignore Empty Train)| `gemma_2_fine_tune_ignore_empty.py`|      512       | **0.75598**  |
| Gemma-2 2b (Base + EOS + Ignore Empty Train)| `gemma_2_fine_tune_ignore_empty.py`|      800       | **0.79904**  |
| Gemma-2 2b (Base + EOS + Ignore Empty Train)| `gemma_2_fine_tune_ignore_empty.py`|     1024       | **0.78947**  |
| Gemma-2 2b (Base + EOS + Ignore Empty Train)| `gemma_2_fine_tune_ignore_empty.py`|     1600       | **0.78708**  |

*Note: The highest score (**0.79904**) was achieved with the Gemma-2 2b base model, fine-tuned for 800 steps using the `gemma_2_fine_tune_ignore_empty.py` script. This script ignores features with missing values during the training phase, focusing the model on relevant data. This score notably outperforms the Gemini "gemini-2.0-flash" model's score of 0.78947. The entry marked "Ignore Empty Inference Only" refers to using the `load_gemma_fine_tuned_ignore_empty.py` script for inference on a model originally trained with `gemma_2_fine_tune_eos.py`, which improved the score for that specific 1024-step model compared to using the standard inference script.*
