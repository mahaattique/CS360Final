'''
Description: Main file used to select which data to preprocess and send through the SVM and RNN model. 
'''

#!/usr/bin/env python3.8

import sys
import util
import json
from movie import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import json
from sentiment_analysis import *



def main():
    '''
    Loading and preprocessing the movie review dataset using subset sizes of 2500, 10000, and 20000. The preprocessed
    data gets written as a CSV file to get stored as a local file. The corresponding emotion_dict also gets stored locally.
    This serves to quicken the process of running the data through our methods.
    '''
    #file_path = 'C:/Users/andri/github-classroom/haverford-cs/cs360_final_data\Movies_Reviews_modified_version1.csv'
    
    file_path = '/homes/mattique/Desktop/CS360/cs360-project-andric-maha/data/Dataset_movies.csv'
    df = util.read_file(file_path)
    subset_size = df.size
    
    dataset, _ = util.preprocess_data(df, subset_size)
    # util.plot_emotion_distribution(emotion_dict)
    dataset.to_csv("/homes/mattique/Desktop/CS360/cs360-project-andric-maha/processed_data.csv")
    
    # Save emotion_dict to a file
    # with open("/homes/mattique/Desktop/CS360/cs360-project-andric-maha/emotion_dict.json", "w") as json_file:
    #     json.dump(emotion_dict, json_file)

    # emotion_dict = util.read_emotion_dict_from_json("/homes/mattique/Desktop/CS360/cs360-project-andric-maha/emotion_dict.json")
    # color_dict = util.plot_emotion_distribution(emotion_dict)
    # util.plot_emotion_distribution(emotion_dict)
    # dataset.to_csv("/homes/mattique/Desktop/CS360/cs360-project-andric-maha/genre_data.csv")
    
    dataset = util.read_file(file_path) #should be reading from the preprocessed data csv file
    emotion_dict = util.read_emotion_dict_from_json("emotion_dict.json")
    color_dict = util.plot_emotion_distribution(emotion_dict, dataset.shape[0])
    util.plot_rating_distribution(dataset)
    util.data_statistics(dataset) #printing statistics of the preprocessed data
    subset_size = dataset.shape[0]
    train_ds, test_ds = util._split_data(dataset) #splitting the dataset

    #extracting the desired features and class for running svm
    train_features = train_ds['Reviews']
    train_labels = train_ds['rating_labels']
    test_features = test_ds['Reviews']
    test_labels = test_ds['rating_labels']

    #vectorizing the reviews using bag of words
    vectorizer = TfidfVectorizer(min_df=10, token_pattern=r'[a-zA-Z]+') #ignores words with counts lower than 10
    train_features_vec = vectorizer.fit_transform(train_features)
    test_features_vec = vectorizer.transform(test_features)
    print("Vocabulary size from text vectorization: ", len(vectorizer.vocabulary_))


    # Running SVM
    run_svm(train_features_vec, train_labels, test_features_vec, test_labels, vectorizer, subset_size)
    feature_importance(train_features, train_labels)

    # Running RNN
    run_rnn_with_descriptions(dataset)
    


if __name__ == "__main__":
    print(sys.path)
    main()
