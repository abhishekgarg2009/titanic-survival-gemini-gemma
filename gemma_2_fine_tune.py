import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from datasets import Dataset
import pandas as pd

original_data = pd.read_csv("train.csv")

print('original_data shape:',original_data.shape)

print(original_data.sample(2))

dataset = Dataset.from_pandas(original_data)

print(dataset)

print(dataset[0]["PassengerId"])
print(dataset[0]["Survived"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available devices print
print("device:",device)

model_id="google/gemma-2-2b"

bnbConfig = BitsAndBytesConfig(
    load_in_4bit=True, # Enable loading of the model in 4-bit quantized format.
    bnb_4bit_quant_type="nf4", # Specify the quantization type. "nf4" refers to a specific 4-bit quantization scheme.
    bnb_4bit_compute_dtype=torch.bfloat16, # Define the data type for computations. bfloat16 offers a good balance between precision and speed.
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnbConfig)

print(model)

def generate_prompt(passengerData):
    prompt = "Based on the passenger information given below, predict whether they survived the Titanic shipwreck. The value is 1 if they survived, 0 otherwise.\n"
    for k in passengerData:
        if k != "PassengerId" and k != "Survived":
            prompt = prompt + k + " : " + str(passengerData[k]) + "\n"
    prompt = prompt + "\nSurvived:"

    return prompt




def generate_response(model, tokenizer, prompt, device, max_new_tokens=128):
    """
    This function generates a response to a given prompt using a specified model and tokenizer.

    Parameters:
    - model (PreTrainedModel): The machine learning model pre-trained for text generation.
    - tokenizer (PreTrainedTokenizer): A tokenizer for converting text into a format the model understands.
    - prompt (str): The initial text prompt to generate a response for.
    - device (torch.device): The computing device (CPU or GPU) the model should use for calculations.
    - max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 128.

    Returns:
    - str: The text generated in response to the prompt.
    """
    # Convert the prompt into a format the model can understand using the tokenizer.
    # The result is also moved to the specified computing device.
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)

    # Generate a response based on the tokenized prompt.
    outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=max_new_tokens)

    # Convert the generated tokens back into readable text.
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract and return the response text. Here, it assumes the response is formatted as "Response: [generated text]".
    # response_text = text.split("Survived:")[1]
    
    return text

prompt = generate_prompt(dataset[1])

response_text = generate_response(model, tokenizer, prompt, device, 32)

print(response_text)

lora_config = LoraConfig(
    r = 8,  # Rank of the adaptation matrices. A lower rank means fewer parameters to train.
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],  # Transformer modules to apply LoRA.
    task_type = "CAUSAL_LM",  # The type of task, here it is causal language modeling.
)

def formatting_func(example):
    line = generate_prompt(example) + " " + str(example["Survived"])
    return line


# Setup for the trainer object that will handle fine-tuning of the model.
trainer = SFTTrainer(
    model=model,  # The pre-trained model to fine-tune.
    train_dataset=dataset,  # The dataset used for training.
    args=TrainingArguments(  # Arguments for training setup.
        per_device_train_batch_size=16,  # Batch size per device (e.g., GPU).
        gradient_accumulation_steps=4,  # Number of steps to accumulate gradients before updating model weights.
        warmup_steps=16,  # Number of steps to gradually increase the learning rate at the beginning of training.
        max_steps=128,  # Total number of training steps to perform.
        learning_rate=2e-4,  # Learning rate for the optimizer.
        fp16=True,  # Whether to use 16-bit floating point precision for training. False means 32-bit is used.
        logging_steps=1,  # How often to log training information.
        output_dir="outputs_gemma2_base",  # Directory where training outputs will be saved.
        optim="paged_adamw_8bit"  # The optimizer to use, with 8-bit precision for efficiency.
    ),
    peft_config=lora_config,  # The LoRA configuration for efficient fine-tuning.
    formatting_func=formatting_func,  # The function to format the dataset examples.
)

trainer.train()

prompt = generate_prompt(dataset[1])

response_text = generate_response(trainer.model.to(torch.half), tokenizer, prompt, device, 32)

print(response_text)