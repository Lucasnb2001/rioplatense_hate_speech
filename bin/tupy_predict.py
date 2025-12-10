from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import numpy as np
from scipy.special import softmax

def classify_hate_speech(model_name, text):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Tokenize input text and prepare model input
    model_input = tokenizer(text, padding=True, return_tensors="pt")

    # Get model output scores
    with torch.no_grad():
        output = model(**model_input)
        scores = softmax(output.logits.numpy(), axis=1)
        ranking = np.argsort(scores[0])[::-1]

    # Print the results
    for i, rank in enumerate(ranking):
        label = config.id2label[rank]
        score = scores[0, rank]
        print(f"{i + 1}) Label: {label} Score: {score:.4f}")

# Example usage
model_name = "Silly-Machine/TuPy-Bert-Large-Multilabel"
text = "Bom dia, flor do dia!!"
classify_hate_speech(model_name, text)

