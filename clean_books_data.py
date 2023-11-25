import pandas as pd
from nltk.tokenize import word_tokenize 
from intents_preprocessing import * 

# load data
books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')


def calculate_weighted_rating():
    #   v is the number of ratings for the book
    #   m is the minimum ratings required to be listed in the chart
    #   R is the average rating of the book
    #   C is the mean rating across the whole report
    # Weighted Rating (WR) =  (vv+m.R)+(mv+m.C)
    v = books['ratings_count']
    m = books['ratings_count'].quantile(0.95)
    R = books['average_rating']
    C = books['average_rating'].mean()
    W = (R*v + C*m) / (v + m)
    return W


def get_genres(x):
    "extract genre for each book title"
    t = book_tags[book_tags.goodreads_book_id==x]
    return [i.lower().replace(" ", "") for i in tags.tag_name.loc[t.tag_id].values]


def clean_data_books():
    ""
    books['original_publication_year'] = books['original_publication_year'].fillna(-1).apply(lambda x: int(x) if x != -1 else -1)
    books['genres'] = books.book_id.apply(get_genres)
    books['weighted_rating'] = calculate_weighted_rating()
    return books

def clean_data_ratings():
    # for the collaborative filtering it is better to have more ratings per user. 
    # remove users who have rated fewer than 3 books
    ratings_rmv_duplicates = ratings.drop_duplicates()
    unwanted_users = ratings_rmv_duplicates.groupby('user_id')['user_id'].count()
    unwanted_users = unwanted_users[unwanted_users < 3]
    unwanted_ratings = ratings_rmv_duplicates[ratings_rmv_duplicates.user_id.isin(unwanted_users.index)]
    new_ratings = ratings_rmv_duplicates.drop(unwanted_ratings.index)
    new_ratings['title'] = books.set_index('id').title.loc[new_ratings.book_id].values
    return new_ratings

def treat_text(a):
    ""
    word_tokens = word_tokenize(str(a).lower()) 
    filtered_sentence = [] 
    for w in word_tokens:
        stem_word=porter.stem(w)
        if stem_word not in stop_words: 
            filtered_sentence.append(stem_word) 
    return filtered_sentence


