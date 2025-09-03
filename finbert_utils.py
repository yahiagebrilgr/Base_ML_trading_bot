# This file is for the sentiment analysis model, has been seperated for debugging and can be reused for future work.
# Imports key components from hugging face transformers library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple
# autotokenizer is a universal translator, it chops text into tokens that the model can understand
# AutoModelForSequenceClassification loads a pre-trained transformer model for classifying sequences
# PyTorch is a library for tensor computations and deep learning
# Tuple is used for type hinting the function's return type

device = "cuda:0" if torch.cuda.is_available() else "cpu" # Check if a GPU is available, otherwise uses CPU

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert") # This line contacts Hugging Face and downloads the specific tokenizer for "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device) # Downloads the actual pre-trained FinBERT model
labels = ["positive", "negative", "neutral"] # Maps the model's numerical outputs to human-readable words.
# 0 = positive, 1= negative, 2= Neutral

def estimate_sentiment(news):
    if news: # First, it checks if there are any news headlines to analyze
        # Tokenize the input headlines and move them to the GPU if available
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        # Get the model's raw output (these are called logits)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        
        # The code below converts the raw scores into probabilities to get the final result
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        
        # Find the highest probability and the matching sentiment label
        probability = result[torch.argmax(result)].item() # .item() gets the number out of the tensor
        sentiment = labels[torch.argmax(result)]
        
        return probability, sentiment
    else:
        # If there's no news, just return a neutral result with 0 confidence
        return 0, labels[-1]

# The estimate_sentiment function takes a list of news headlines and figures out the overall mood. It uses a tokenizer
# to chop up the headlines into numbers that the model can understand. It then feeds those numbers into the FinBERT model,
# which is pre-trained to understand the language of financial news. The model outputs scores for each sentiment
# (positive, negative, neutral). The function sums these scores across all headlines, converts them to probabilities,
# and then picks the one with the highest confidence. If there isn't any news, it just returns "neutral".