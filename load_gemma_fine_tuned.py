import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
import pandas as pd
import csv

output_dir="outputs_gemma2_base/checkpoint-128"

bnbConfig = BitsAndBytesConfig(
    load_in_4bit=True, # Enable loading of the model in 4-bit quantized format.
    bnb_4bit_quant_type="nf4", # Specify the quantization type. "nf4" refers to a specific 4-bit quantization scheme.
    bnb_4bit_compute_dtype=torch.bfloat16, # Define the data type for computations. bfloat16 offers a good balance between precision and speed.
)

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto", quantization_config=bnbConfig)

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


original_data = pd.read_csv("test.csv")

print('original_data shape:',original_data.shape)

print(original_data.sample(2))

dataset = Dataset.from_pandas(original_data)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



prompt = generate_prompt(dataset[1])

response_text = generate_response(model, tokenizer, prompt, device, 32)

print(response_text.splitlines()[12])


predictions = []
for row in dataset:
    prompt = generate_prompt(row)
    response_text = generate_response(model, tokenizer, prompt, device, 32)
    print(response_text)
    final_response = response_text.splitlines()[12]
    print(final_response)
    if "1" in final_response:
        predictions.append({"PassengerId":row["PassengerId"], "Survived":1})
    else:
        predictions.append({"PassengerId":row["PassengerId"], "Survived":0})

print(predictions)

with open("submission_gemma_base_128.csv", 'w') as csvFile:
    writer = csv.DictWriter(csvFile, predictions[0].keys())
    writer.writeheader()
    writer.writerows(predictions)