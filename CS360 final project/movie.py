'''
Description: Three classes are defined here being 'Partition', 'MovieBag', 'Example'. These classes serve to organize
             the IMBD movie reviews dataset by breaking it down into a partition containing a collection of bags 
             where each bag is defined under a movie_name such that each movie gets grouped with its reviews. 
Authors: Andric Brena, Maha Attique
'''

class Partition:
    '''
    Creates a Partition object defined as containing a list of bags and attributes pertaining to 
    the number of distinct movies and the number of total reviews in the dataset.
    Params:
        bags: list of MovieBag objects
    '''
    def __init__(self, bags):
        self.data = bags #list of movie bags
        self.num_movies = len(bags) #number of bags 
        self.num_examples = 0 #number of examples in the partition, intitialized first to 0
        for bag in bags:
            self.num_examples += bag.n

class MovieBag:
    '''
    Creates a MovieBag object defined as containing a list of reviews under a shared movie name.
    Params:
        name: (str) movie name
        data: list of examples/reviews for the specified movie
    '''
    def __init__(self, name, data):
        self.name = name #name of the movie
        self.data = data #list of examples/reviews
        self.n = len(self.data) #number of examples 

class Example:
    '''
    Creates an Example object defined as containing the review and rating label(pos=1, neg=-1) 
    using a threshold and the emotion attached to the movie. 
    '''
    def __init__(self, review, rating, emotion):
        self.review = review #review of the example
        # Create a rating label (positive or negative) using threshold=5
        if rating > 5:
            self.rating_label = 1
        else:
            self.rating_label = -1
        self.emotion_label = emotion #emotion of the example

