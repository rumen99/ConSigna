import random 
import requests
import json
import re
import torch.nn.functional as F
from Levenshtein import distance as levenshtein_distance
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load the spaCy model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(device)

# def get_bert_embedding(word):
#     encoded_input = tokenizer(word, return_tensors='pt').to(device)
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     embeddings = model_output.last_hidden_state.mean(dim=1)
#     return embeddings

# def find_most_similar_word(input_word, word_embeddings, words):
#     input_embedding = get_bert_embedding(input_word).to(device)  # Ensure embedding is on GPU
#     word_embeddings_tensor = word_embeddings.to(device)  # Ensure embeddings tensor is on GPU
    
#     # Normalize the embeddings to unit length
#     input_embedding_normalized = F.normalize(input_embedding, p=2, dim=1)
#     word_embeddings_normalized = F.normalize(word_embeddings_tensor, p=2, dim=1)
    
#     # Compute cosine similarity as dot product of normalized embeddings
#     # Since both embeddings are normalized, dot product will give cosine similarity
#     similarities = torch.mm(input_embedding_normalized, word_embeddings_normalized.t())
    
#     # Find the index of the highest similarity score
#     most_similar_idx = torch.argmax(similarities, dim=1).item()
    
#     # Retrieve the most similar word
#     most_similar_word = words[most_similar_idx]
    
#     # print("most_similar_word: ", most_similar_word)
#     return most_similar_word
def find_closest_word(target, word_embeddings, words_dataset):
    if target in words_dataset:
        return target
    if target in word_embeddings:
        return word_embeddings[target]
    return target

def get_video_from_word(word, words):
    
    for i in words:
        if word.lower() in i['token']['ur']:
            #print (f"word: {word} in words")
            #print (f"video: {i['label']}")
            return [(word,i['label'])]
    return None

def get_character (letter, words):

    video = get_video_from_word(letter, words)
    if video == None:
        video = (letter+"-r",random.choice(words)['label'])

    #print ("letter: ", letter)
    #print (f"video: {video}")
    return [video]

def get_video_character_by_character (word, words):
    #print (f"{word} not in words")
    word_list = []
    for i in word:
        word_list.extend(get_character(i, words))
    return word_list

def get_video (word, words):
    video = get_video_from_word(word, words)
    if video == None:
        if len(word) < 5:
            return []
        else:
            video = [(word+'-r',random.choice(words)['label'])]
        # if len(word) > 5:
        #    video = [(word+'-r',random.choice(words)['label'])]
        # else:
        #     video = get_video_character_by_character(word, words)
    #video = [(word+'-r',random.choice(words)['label'])]
    #print ("get_video: ", word, video)
    return video

def get_video_from_text (text, words, word_embeddings, words_dataset):
    clean_text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', text)

    # Split the text into words
    separated_words = clean_text.split()

    # Convert all words to lowercase
    lowercase_words = [word.lower() for word in separated_words]

    # Get the video for each word
    video_list = []
    for word in lowercase_words:
        # add all videos in get_video in the video list
        video_list.extend(get_video( find_closest_word(word, word_embeddings, words_dataset), words))
    
    return video_list

