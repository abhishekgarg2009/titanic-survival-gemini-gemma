# My Journey: Predicting Titanic Survival with Gemini and Fine-Tuned Gemma

This document chronicles the iterative process of applying Large Language Models (LLMs) – specifically Google's Gemini and Gemma – to the classic Kaggle Titanic survival prediction challenge. My goal was not just to achieve a good score, but also to understand how different models and fine-tuning strategies perform on this structured data problem, refining the approach based on observed results.

## Iteration 1: Baseline with Gemini Flash

My initial exploration began with the powerful, general-purpose Gemini models. I formulated the problem as a text-completion task: given the details of a passenger from the test set, and the entire training dataset as context, could Gemini predict survival ('1') or non-survival ('0')?

Using the `predict_gemini.py` script and the `gemini-2.0-flash` model, I provided the full training data context for each test passenger. This approach yielded a promising Kaggle score of **0.78947**. Notably, this already surpassed the 0.77511 score achieved by the Random Forest classifier in the official Kaggle tutorial, suggesting that the broad world knowledge embedded within Gemini provided an advantage even without specific fine-tuning for this task.

## Iteration 2: Exploring Gemma-2 2b Fine-Tuning

While Gemini provided a strong baseline, I hypothesized that a smaller model, specifically fine-tuned on the Titanic dataset, might capture the underlying patterns more effectively. I turned my attention to Gemma-2 2b, a powerful open model.

My initial fine-tuning efforts (`gemma_2_fine_tune.py`) involved using LoRA (Low-Rank Adaptation) to adapt the base `google/gemma-2-2b` model to the training data. I also tested the instruction-tuned variant (`gemma-2-2b-it`). The instruction-tuned model showed a slight edge initially (0.78468 vs 0.77751 for the base model after 128 steps).

## Iteration 3: Enforcing Output Format with EOS Token

I noticed the fine-tuned models sometimes generated extraneous text beyond the simple '0' or '1' prediction. To enforce stricter output formatting, I introduced the End-of-Sequence (`<eos>`) token explicitly into the training data formatting (`gemma_2_fine_tune_eos.py`). This helped focus the model's output. Experimenting with training duration using this EOS approach revealed that performance improved and then plateaued/decreased:
*   128 steps: 0.77751
*   256 steps: 0.77511
*   **512 steps: 0.78947** (Matching the best Gemini score)
*   1024 steps: 0.78229

## Iteration 4: Testing Gemini Thinking Model

Concurrently with Gemma fine-tuning, I experimented with an experimental Gemini "thinking" variant, `gemini-2.0-flash-thinking-exp-01-21`, using `predict_gemini_thinking.py`. Interestingly, this model performed slightly worse than the standard Gemini Flash, scoring **0.77272**. This suggested that the Titanic survival prediction might lean more towards pattern recognition within the provided data rather than complex, multi-step reasoning, potentially making the overhead of the "thinking" process less beneficial for this specific task.

## Iteration 5: Refining Inference - Ignoring Missing Data

Before changing the training process, I explored whether ignoring missing features (`NaN` values) *during inference* could improve predictions, even for models trained *without* this logic. I created `load_gemma_fine_tuned_ignore_empty.py` which modified the prompt generation at prediction time to exclude missing features.

Applying this inference script to the model trained for 1024 steps with the EOS token (from Iteration 3) improved its score from 0.78229 to **0.78468**. This indicated that handling missing data was indeed important, even if only addressed at inference time.

## Iteration 6: Refining Training - Ignoring Missing Data

Based on the success of ignoring missing data during inference, I theorized that incorporating this logic into the *training* phase itself would be even more beneficial. Could the model perform better if it only focused on the *present* features for each passenger during training?

This led to the creation of `gemma_2_fine_tune_ignore_empty.py`. This script modified the prompt generation during fine-tuning (within the `formatting_func`) to dynamically exclude any feature-value pairs where the value was missing. This ensured the model learned associations based only on the information actually available for each passenger in the training set, combined with the EOS token for focused output.

This strategy proved highly effective. Fine-tuning the base Gemma-2 2b model with both the EOS token and the "ignore empty features during training" logic yielded my best results:
*   512 steps: 0.75598
*   **800 steps: 0.79904** (My highest score!)
*   1024 steps: 0.78947
*   1600 steps: 0.78708

The peak performance of **0.79904** at 800 steps demonstrated the significant benefit of this refined training approach. It showed that carefully curating the training input by removing noise (missing values) allowed the smaller Gemma model to learn the relevant patterns more effectively than any previous approach.

## Conclusion: Lessons Learned

My iterative journey through Gemini and Gemma for the Titanic challenge yielded several key insights:

1.  **LLMs for Structured Data:** Even general-purpose LLMs like Gemini can outperform traditional ML baselines (like Random Forest) on structured data tasks, likely by leveraging their vast pre-trained knowledge.
2.  **Fine-Tuning Smaller Models:** A smaller model (Gemma-2 2b) specifically fine-tuned on the task data can outperform a larger, general-purpose model (Gemini-Flash). This highlights that task-specific adaptation is crucial and not all problems require the largest model available.
3.  **Task Specificity Matters:** The slightly lower performance of the Gemini "thinking" model suggests that understanding the nature of the task (pattern recognition vs. complex reasoning) is important for model selection.
4.  **The Power of Pre-training:** The ability of these LLMs to achieve strong results demonstrates the power of starting with a model that already possesses a broad understanding of language and concepts, which can then be effectively adapted to specific, even traditionally ML-oriented, problems.

Ultimately, achieving a score of **0.79904** with a fine-tuned Gemma-2 2b model showcased the potential of adapting LLMs for structured data prediction.
