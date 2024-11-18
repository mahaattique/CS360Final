'''
Description: This file contains utility functions to create a Pandas dataframe from a csv file, preprocess the data,
             build train and test datasets, and produce graphs for preliminary analysis.
Authors: Andric Brena, Maha Attique
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from movie import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

def read_file(path):
    '''
    Reads from a csv file and returns a pandas dataframe.
    Params:
        path: (str) the csv file path
    Returns:
        df: pandas dataframe
    '''
    df = pd.read_csv(path, index_col=0)
    return df

def preprocess_data(df, subset_size):
    '''
    Preprocesses the IMBD movie reviews dataset stored in a dataframe. 
    Params:
        df: pandas dataframe
        subset_size: (int) number of examples to grab from the dataset
    Returns:
        df: preprocessed dataframe
        emotion_dict: dictionary highlighting the count of positive or negative reviews for each emotion in df
    '''
    drop_columns = ['Resenhas','genres'] 
    df = df.drop(drop_columns, axis=1) #drops irrelevant columns
    df = df.drop_duplicates(['Reviews'], keep='first') #drops review duplicates prioritizing the one with lower index
    df = df.sample(frac = 1) #randomly samples the data
    df = df.iloc[0:subset_size] #grabs the first subset_size examples

    emotion_dict = {}  # Initialize an empty dictionary for emotions
    rating_labels = [] #initializes an empty array for rating labels
    for index, row in df.iterrows():
        #rating label is 1 if rating > 5 else it is -1
        if row['Ratings'] > 5:
            rating_labels.append(1)
        else:
            rating_labels.append(-1)
        
        # Extract emotion and rating
        emotion = row['emotion']
        rating = row['Ratings']

        # Check if emotion already exists in the dictionary
        if emotion not in emotion_dict:
            emotion_dict[emotion] = {'pos': 0, 'neg': 0}
           
        # emotion_dict.add(emotion)
        # Increment 'pos' or 'neg' based on rating
        if rating > 5:
            emotion_dict[emotion]['pos'] += 1
        else:
            emotion_dict[emotion]['neg'] += 1


    df = df.assign(rating_labels=rating_labels)
    df = df.drop(['Ratings'], axis=1)

    df = df.sample(frac = 1)
    df = df.iloc[:]
    df['rating_labels'] = rating_labels
    df['Description'] = df['Description'].apply(_filter_reviews)
    df['Reviews'] = df['Reviews'].apply(_filter_reviews)
    return df, emotion_dict

def _filter_reviews(review):
    '''
    Filters a review by removing stopwords or words common in the English language that emit 
    little to no sentiment. Also breaks down each word in the review to their root form depending on their
    part of speech.
    Params:
        review: (str) a movie review
    Returns:
        filtered_review: (str) the given review after being filtered
    '''
    tokens = word_tokenize(review.lower()) #break down the review into tokens

    removed_stopwords = [] #will contain tokens in the review not considered to be a stopword
    for token in tokens:
        if token not in stopwords.words('english'): #assuming each review is in English
            removed_stopwords.append(token) 
    
    lemmatizer = WordNetLemmatizer() #creates a lemmatizer object 
    lemmatized_tokens = [] #will contain each token broken down to their root form depending on their part of speech
    for token in removed_stopwords:
        lemmatized_tokens.append(lemmatizer.lemmatize(token)) #lemmatizes the token
    
    filtered_review = ' '.join(lemmatized_tokens) #adds every token back together as a string
    return filtered_review

def data_statistics(data):
    '''
    Provides statistics relating to the movie review dataset. Highlights number of reviews,
    average word size in each review, and the distribution of positive or negative labels.
    Params:
        data: the pandas dataframe to analyze
    '''
    print("Number of Reviews: ", data.shape[0])
    print("Average Word Size in a Review: ", np.mean([len(row["Reviews"]) for index, row in data.iterrows()]))
    distribution = Counter([row["rating_labels"] for index,row in data.iterrows()])
    print("Distribution of Labels: ", distribution)

def build_datasets(data):
    '''
    Builds train and test datasets as a Partition object containing bags of movies as
    defined in movie.py.
    Params:
        data: movie review dataframe
    Returns:
        train_dataset: dataframe containing the data for training
        test_dataset: dataframe containing the data for testing
    '''
    unique_movies = data['movie_name'].unique() #grabs all distinct movies
    movies_dict = {movie:[] for movie in unique_movies} #dictionary where each movie is paired with its list of examples
    for index, row in data.iterrows():
        ex = Example(row['Review'], row['Ratings'], row['emotion']) #makes an Example object
        movies_dict[row['movie_name']].append(ex) #adds example to the list of reviews under the same movie name
    
    movie_bags = [] #list of movie bags
    for movie in movies_dict.keys():
        movie_bags.append(MovieBag(movie, movies_dict[movie])) #creates a MovieBag object and adds it to the list

    train, test = _split_data(movie_bags) #splits movie_bags in test and train datasets

    #creates a Partition object for both datasets
    train_dataset = Partition(train) 
    test_dataset = Partition(test)
    return train_dataset, test_dataset

def _split_data(data):
    '''
    Splits the data into train and test datasets.
    Params:
        data: a pandas dataframe
    Returns:
        train: dataframe containing 70% of data
        test: dataframe containing 30% of data
    '''
    train_percentage = int(data.shape[0] * .70) #train dataset to continue 70% of the data
    train = data[0:train_percentage]
    test = data[train_percentage:] #test dataset contains the remaining 30% 
    return train, test

def create_color_dictionary(classes):
    '''
    Creates a dictionary with key:class and value:color where each value is a distinct color
    Parameter:
        classes: list of classes
    Return:
        color_dict: dictionary assigning a color to each class
    '''
    unique_classes = np.unique(classes) 
    num_classes = len(unique_classes) 

    
    color_map = plt.get_cmap('Paired')

    
    color_dict = {class_label: color_map(i / num_classes) for i, class_label in enumerate(unique_classes)}
    return color_dict

def plot_emotion_distribution(emotion_dict, subset_size):
    '''
    Plots the distribution for the emotions in the movie reviews dataset, highlighting the amount of positive and 
    nagative labeled reviews for each emotion.
    Params:
        emotion_dict: dictionary containing the rating label distribution for each emotion
    Returns:
        color_dict: dictionary containing the color assigned for each emotion on the plot
    '''
    emotions = list(emotion_dict.keys()) #list of distinct emotions in the data
    total_counts = [emotion_dict[emotion]['pos'] + emotion_dict[emotion]['neg'] for emotion in emotions] #total occurrence for each emotion

    #percentage of positive labeled reviews for each emotion
    pos_percentages = [(emotion_dict[emotion]['pos'] / total_count) * 100 for emotion, total_count in zip(emotions, total_counts)]
    #percentage of negative labeled reviews for each emotion
    neg_percentages = [(emotion_dict[emotion]['neg'] / total_count) * 100 for emotion, total_count in zip(emotions, total_counts)]

    color_dict = create_color_dictionary(emotions) #assigns a color to each emotion

    plt.figure(figsize=(10, 6))

    # Plotting bars with different colors for each emotion
    bars = plt.bar(emotions, total_counts, color=[color_dict[emotion] for emotion in emotions], label='Total Count')

    text_space = max(total_counts) * 0.05

    for bar, pos_percentage, neg_percentage, total_count in zip(bars, pos_percentages, neg_percentages, total_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, total_count + (3*text_space), f'Pos: {pos_percentage:.2f}%', ha='center', va='top', color='blue')
        plt.text(bar.get_x() + bar.get_width() / 2, total_count + text_space, f'Neg: {neg_percentage:.2f}%', ha='center', va='top', color='red')

    plt.xlabel('Emotion')
    plt.ylabel('Total Count')
    #plt.ylim(0, 10000)
    plt.title('Emotion Distribution on %i examples' % (subset_size))

    plt.xticks(rotation=45, ha='right')

    plt.legend(handles=[plt.Line2D([], [], color='blue', label='Positive Percent'),
                        plt.Line2D([], [], color='red', label='Negative Percent')])

    
    plt.tight_layout()
    plt.savefig("split_emotion_distribution_%i.png" % (subset_size))
    # plt.show()
    return color_dict

def read_emotion_dict_from_json(json_file):
    '''
    Reads from a json file contents pertaining to the spread of emotions and pos/neg labeled reviews in a dataset.
    Params:
        json_file: (str) json file path
    Returns:
        emotion_dict: dictionary highlighting the amount of pos/neg labeled reviews for each emotion
    '''
    #reading from the json file
    with open(json_file, 'r') as file:
        emotion_dict = json.load(file)
    return emotion_dict

def plot_rating_distribution(df):
    '''
    Plots a bar graph highlighting the spread of ratings in the movie review dataset.
    Params:
        df: movie reviews dataframe 
    '''
    ratings = df['Ratings'] #grabs the Ratings column
    num_ex = df.shape[0]
    ratings_dict = ratings.value_counts()

    #plots the bar graph 
    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*ratings_dict.items()), align='center')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution across %i examples' % (num_ex))

    plt.xticks(range(1, 11))

    plt.grid(True)

    plt.tight_layout()
    plt.savefig("rating_distribution_bar_%i.png" % (num_ex))