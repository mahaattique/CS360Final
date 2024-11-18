[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/pTJgFjpP)
# cs360-project
slides: https://docs.google.com/presentation/d/16twnulwDPJc2yNEfVRSZWGWwroHOEx8sQnMgIEibmB4/edit?usp=sharing 

Project Setup:
The data for our project comes from Kaggle: https://www.kaggle.com/datasets/fahadrehman07/movie-reviews-and-emotion-dataset. We downloaded this dataset as a csv file so that we can read it in as 
a pandas dataframe. Using a threshold of 5, reviews whose rating was >5 were given a positive sentiment(1) while those with a rating<=5 were given a negative sentiment(-1). These sentiments as integer values were treated as the labels for each review. Prelimary analysis was done to highlight the distribution of emotions, ratings, and the sentiment across the reviews using bar plots. During the preprocess step, we dropped duplicate reviews, created a subset, and randomized the data. The reviews were filtered using the nltk library to lemmatize and remove stopwords. Because this step takes a while, we created subsets of different sizes containing the preprocessed data and stored them locally to quicken future model testing.

Our first goal was to use SVM from sklearn to analyze movie reviews. Using sklearn, we plotted the SVM's training curves, built its confusion matrix, and plotted the most important features(words) using the model's weights. 

Our second goal was to use an RNN trained on the descriptions of the movies and their corresponding emotion labels to see if the reviews reflect the same emotions as the description. 

To run the file from command line, please download the dataset, and add the correct file path to main.py. Uncomment the lines of code in main.py to preprocess the raw data and output it to a CSV for processing later. Other than that, running main.py should run all the methods.