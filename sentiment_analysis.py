import pandas as pd
import numpy as np
from util import *
from sklearn import svm

import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit
from sklearn.datasets import load_digits
from sklearn.model_selection import LearningCurveDisplay

def run_rnn_with_descriptions(df):
    """
    Trains a recurrent neural network (RNN) model using the descriptions from the DataFrame.
    The model is trained to classify emotions based on the provided descriptions.
    Params:
        df (DataFrame): DataFrame containing the data, including 'Description' and 'emotion' columns.
    Returns:
        None
    """

    num_classes = len(df['emotion'].unique())
   
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Description'])
    X = tokenizer.texts_to_sequences(df['Description'])
    test = tokenizer.texts_to_sequences(df['Reviews'])
    X = pad_sequences(X)
    test = pad_sequences(test, maxlen=X.shape[1])

    y = pd.get_dummies(df['emotion']).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

    X_test = test
    y_test = y


    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 128
    max_length = X.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   
    history = model.fit(X_train, y_train, epochs=7, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    y_pred = model.predict(X_test)

  
    y_pred_labels = [np.argmax(pred) for pred in y_pred]
    y_test_labels = [np.argmax(label) for label in y_test]

   
    print(classification_report(y_test_labels, y_pred_labels))

    cm = confusion_matrix(y_test_labels, y_pred_labels)

    emotion_names = df['emotion'].unique()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=emotion_names, yticklabels=emotion_names, cmap='coolwarm')
    plt.ylabel('Actual Emotion', fontsize=13)
    plt.xlabel('Predicted Emotion', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.savefig("rnn_confusion_matrix.png")
    plt.clf() 

    history_dict = history.history
    print('Analyzing Model Performance:')
    print('Plotting RNN train learning curves')
    plot_learning_curve_rnn("rnn", history_dict)


def run_svm(X_train, y_train, X_test, y_test, vectorizer, subset_size):
    '''
    Runs SVM on the data, plotting the model's learning curve, confusion matrix, and the most important features.
    Params:
        X_train: matrix containing the features for train dataset
        y_train: vector containing the labels for train dataset
        X_test: matrix containing the features for the test dataset
        y_test: vector containing the labels for the train dataset
        vectorizer: TfidfVectorizer object used to vectorize the text features for both datasets
        subset_size: (int) number of total examples from the preprocessed data being used
    '''
    model = svm.SVC(C=10.0, kernel='linear') #creates SVC object
    print('Analyzing Model Performance:')
    print('Plotting SVM train learning curves')
    plot_learning_curve_svm(model, X_train, y_train)
   
    #performs cross validation to see how well the model generalizes the data
    model_scores = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5, n_jobs=-1)
    print("Average Score from performing Cross-Validation: ", model_scores.mean())

    print('\nRunning SVM on the data:')
    model.fit(X_train,y_train) #fits training data
    y_pred = model.predict(X_test) #classifies test data

    # Plot the confusion matrix 
    print('Constructing the confusion matrix')
    cm = confusion_matrix(y_test, y_pred) #builds the confusion matrix using the svm model
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train), cmap='YlGnBu')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('SVM Confusion Matrix on %i examples' % (subset_size), fontsize=17)
    plt.savefig("svm_cm_%i.png" % (subset_size))
    plt.show()
    
    # Plotting most important features 
    print('Plotting the most important features')
    feature_importance(model, vectorizer, subset_size)

def plot_learning_curve_svm(estimator, X_train, y_train):
    '''
    Plots the learning curves to measure how well the model performs as number of training examples increases.
    Assumes that the training data is getting passed to measure train and validation scores.
    Params:
        estimator: classifier method
        X_train: matrix containing features of train dataset
        y_train: vector containing the labels of train dataset
    '''
    num_ex = y_train.size #number of examples used for producing the learning curves
    
    #Performs 5-fold cross validation to see how well model generalizes training data
    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, 
                                                             X=X_train, 
                                                             y=y_train, 
                                                             train_sizes=np.linspace(0.1, 1.0, 10), 
                                                             cv=5
    )

    #Calculating the mean score for each data subset
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    #Plotting the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue", marker="o")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="blue"
    )
    plt.plot(train_sizes, valid_mean, label="Validation score", color="green", marker="o")
    plt.fill_between(
        train_sizes,
        valid_mean - valid_std,
        valid_mean + valid_std,
        alpha=0.15,
        color="green"
    )
    plt.xlabel("Number of training examples", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title("SVM Learning Curves on %i examples" % (num_ex), fontsize=17)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("svm_learning_curve_%i.png" % (num_ex))



def plot_learning_curve_rnn(m, history_dict):
    '''
    Plots the learning curves to measure how well the model performs as number of training examples increases.
    Assumes that the training data is getting passed to measure train and validation scores.
    Params:
        m: model type
        history_dict : Dictionary containing models training history
    '''
    #Plot loss learning curves
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "m", label="Training Loss")       
    plt.plot(epochs, val_loss_values, "c", label="Validation Loss")  
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{m}_loss_curve.png")
    plt.clf() 

    #Plot accuracy learning curves
    accuracy = history_dict["accuracy"]
    val_accuracy = history_dict["val_accuracy"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, accuracy, "m", label="Training Accuracy")       
    plt.plot(epochs, val_accuracy, "c", label="Validation Accuracy")  
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f"{m}_accuracy_curve.png")
    plt.clf() 

def feature_importance(model, vectorizer, sub_size):
    '''
    Given a fitted model, plots the 15 most important features(words) for both positive and
    negative sentiment labels using their attached weights. 
    Params:
        model: fitted classifier method
        vectorizer: TfidfVectorizer object used to vectorize the text features of the data
    '''
    weights = model.coef_.toarray().flatten() #grabs the weights for each feature
    top_indices_pos = np.argsort(weights)[::-1][:10] #grabs the top 10 positive weights
    top_indices_neg = np.argsort(weights)[:10] #grabs the top 10 negative weights
    words = np.array(vectorizer.get_feature_names_out()) #grabs the word name for each feature

    #builds the dataframe highlighting each word, its weight, and attached sentiment
    feature_importance_df = pd.DataFrame({'Feature_Word': words[np.concatenate((top_indices_pos, top_indices_neg))],
                                          'Importance': weights[np.concatenate((top_indices_pos, top_indices_neg))],
                                          'Sentiment': ['pos' for i in range(len(top_indices_pos))] + ['neg' for i in range(len(top_indices_neg))]})
    
    sns.barplot(x = 'Feature_Word', y = 'Importance', data = feature_importance_df,
                hue = 'Sentiment', dodge=False,
                order = feature_importance_df.sort_values('Importance').Feature_Word)
    plt.xlabel("Most Important Feature Words")
    plt.ylabel("Feature Weights")
    plt.xticks(rotation=80)
    plt.title("Feature Importance of the Top 10 Words")
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("svm_important_features_%i.png" % (sub_size))
    plt.show()
