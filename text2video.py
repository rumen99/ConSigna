import random 
import requests
import json
import re

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
        if len(word) < 3:
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

def get_video_from_text (text, words):
    clean_text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', text)

    # Split the text into words
    separated_words = clean_text.split()

    # Convert all words to lowercase
    lowercase_words = [word.lower() for word in separated_words]

    # Get the video for each word
    video_list = []
    for word in lowercase_words:
        # add all videos in get_video in the video list
        video_list.extend(get_video(word, words))
    
    return video_list

