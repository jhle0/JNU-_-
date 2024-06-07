import numpy as np
import pandas as pd
import typing
#import List
import re
from collections import defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample



def filter_text(text:str) -> str:
    '''
    Remove URLs and non-alphanumeric characters and Convert to lowercase and split into words.
    '''
    text = re.sub(r'http\S+|[^a-zA-Z0-9\s]', '', text)
    return text.lower().split()

def make_filtered_word(data) -> dict:
    '''
    Return filtered word Dict which are used at least 69 rows.
    '''
    word_count = defaultdict(int)
    for post in data['posts']:
        words = set(filter_text(post))
        for word in words:
            word_count[word] += 1
    # Filter out words that appear less than 69 rows.
    filtered_words = {word:count for word, count in word_count.items() if count >= 69}
    return filtered_words

def make_filtered_post(text:str, filtered_words_set:dict) -> str:
    '''
    Use filter_text func. 
    '''
    posts = text.split('|||')
    # Filter each post.
    filtered_posts = [' '.join([word for word in list(filter_text(post)) if word in filtered_words_set]) for post in posts]
    # Join the filtered posts with "|||".
    return '|||'.join(filtered_posts)

def nltk_preprocess(text):
    '''
    Preprocess by lemmatizing and excluding stop_words.
    '''
    words = text.split()
    cleaned_text = []
    for word in words:
        if word.lower() not in stop_words:
            lemmatized_word = lemmatizer.lemmatize(word.lower())
            cleaned_text.append(lemmatized_word)

    return ' '.join(cleaned_text)

if __name__ == '__main__':
    data = pd.read_csv('MBTI 500.csv')

    # Make filtered column which is prepocessed string of column posts
    filtered_words = make_filtered_word(data)
    data['filtered_posts'] = data['posts'].apply(lambda x: make_filtered_post(x, set(filtered_words)))
    data['filtered_posts'] = data['filtered_posts'].str.replace('|||', '', regex=False)

    # Preprocess with nltk library.
    expanded_data = data
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    expanded_data['filtered_posts'] = expanded_data['filtered_posts'].apply(nltk_preprocess)
    expanded_data = expanded_data[expanded_data['filtered_posts'].str.strip() != '']

    # Make Lable.
    label_encoder = LabelEncoder()
    expanded_data['encoded_labels'] = label_encoder.fit_transform(expanded_data['type'])
    class_counts = expanded_data['encoded_labels'].value_counts()

    # Do upsampling and Shuffle.
    max_class_count = class_counts.max()
    data_upsampled = pd.DataFrame()
    for class_index, count in class_counts.items():
        data_class = expanded_data[expanded_data['encoded_labels'] == class_index]
        data_class_upsampled = resample(data_class, 
                                        replace=True, 
                                        n_samples=max_class_count, 
                                        random_state=123)
        data_upsampled = pd.concat([data_upsampled, data_class_upsampled])
    data_upsampled = data_upsampled.sample(frac=1).reset_index(drop=True)

    # Saving.
    data_upsampled.to_csv('filtered_dataset.csv', index=False)
