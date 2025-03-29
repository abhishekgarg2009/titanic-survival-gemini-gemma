import csv
import json
from google import genai
import os
import time

data = []
with open("train.csv") as csvFile:
    csvRead = csv.DictReader(csvFile)
    for rows in csvRead:
        data.append(rows)
    
    print(json.dumps(data, indent=4))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
predictions = []
with open("test.csv") as csvFile:
    csvRead = csv.DictReader(csvFile)
    for rows in csvRead:
        prompt = []
        prompt.append("Here is the data of titanic passengers, describing different features and whether they survied or not:")
        prompt.append(json.dumps(data, indent=4))
        prompt.append("Now based on the data above, predict whether the passenger given below would have surived or not")
        prompt.append(json.dumps(rows))
        prompt.append("Just generate simple Yes or No.")
        print(json.dumps(rows))
        for attempt in range(10):
            try:
                response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            except:
                time.sleep(10)
            else:
                break
        print(response.text)
        if "no" in response.text.lower():
            predictions.append({"PassengerId":rows["PassengerId"], "Survived":0})
        else:
            predictions.append({"PassengerId":rows["PassengerId"], "Survived":1})
        print(predictions)

with open("submission.csv", 'w') as csvFile:
    writer = csv.DictWriter(csvFile, predictions[0].keys())
    writer.writeheader()
    writer.writerows(predictions)