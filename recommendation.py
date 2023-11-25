import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download("stopwords")
from clean_books_data import*
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
porter = PorterStemmer()
stop_words = set(stopwords.words('english')) 

# load data

books = clean_data_books()
new_ratings = clean_data_ratings()
content = books[['original_title','authors', "genres"]].dropna().astype(str).reset_index(drop=True)
bookmat = new_ratings.pivot_table(index='user_id', columns='title', values='rating')
questions = pickle.load(open('questions.pkl','rb'))
preprocessed_text = [treat_text(i) for i in questions]
flat_list = [item for sublist in preprocessed_text for item in sublist]
string_list = [i for i in content["original_title"]]

def find_title_content_based(user_request):
    "find_book_title_in_sentence"
    preprocessed_request = treat_text(user_request)
    title=[i for i in preprocessed_request if i not in flat_list]
    new_title = ' '.join(title)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = tfidf_vectorizer.fit_transform([new_title]+string_list)
    cosine_sim = cosine_similarity(sparse_matrix[0,:],sparse_matrix[1:,:])
    return pd.DataFrame({'cosine':cosine_sim[0],'strings':string_list}).sort_values('cosine',ascending=False).iloc[0]["strings"]

def asking_about_information(title):
    ""
    HP_OP_INDEX=books.index[books['title']==title].tolist()[0]
    answers=[]
    answer={'_type':"information",
            'title': title,
            'authors': books.iloc[HP_OP_INDEX]["authors"],
            'Pulication year': books.iloc[HP_OP_INDEX]["original_publication_year"],
            'rating': books.iloc[HP_OP_INDEX]["weighted_rating"]}
    answers.append(answer)
    chatbot_message={"answer":answers}
    return chatbot_message


def top_rated():
    ""
    tf_top_rated  = books.sort_values('weighted_rating', ascending=False).head(10)
    answers=[]
    for i in range(0, 10):
        answer={'_type':"rated", 'book_title':tf_top_rated.iloc[i]["title"], 'rating': tf_top_rated.iloc[i]["weighted_rating"]}
        answers.append(answer)
    chatbot_message = {"answer": answers}  
    return chatbot_message


def create_mat_cos_sim():
    ""
    #removing stopwords
    tfidf = TfidfVectorizer(stop_words='english')    
    #Construct the required TF-IDF matrix by fitting and transforming the data    
    tfidf_matrix_author = tfidf.fit_transform(content['authors'])
    cosine_sim_author = cosine_similarity(tfidf_matrix_author, tfidf_matrix_author)
    tfidf_matrix_genre = tfidf.fit_transform(content['genres'])
    cosine_sim_genre = cosine_similarity(tfidf_matrix_genre, tfidf_matrix_genre)
    return cosine_sim_author, cosine_sim_genre


def get_recommendation_author_or_genre(title, cosine_sim):
    ""    
    id=pd.Series(content.index, index=content['original_title'])
    idx=id[title]
    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]
    # Get the book indices
    book_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar books
    answers=[]
    for i in range(0, 10):
        answer={'_type':"rated", 'book_title':list(content['original_title'].iloc[book_indices])[i]}
        answers.append(answer)
    chatbot_message = {"answer": answers}
    return chatbot_message


def get_recommendation_collaborative(title, mat):
    "Item-based collaborative filtering: which measures the similarity between the items that target users rate or interact with and other items."
    title_user_ratings = mat[title]
    similar_to_title = mat.corrwith(title_user_ratings)
    corr_title = pd.DataFrame(similar_to_title, columns=['correlation'])
    corr_title.dropna(inplace=True)
    corr_title.sort_values('correlation', ascending=False, inplace=True)
    answers = [1]
    for i in range(0, 10):
        answer={'_type':"collaborative", 'book_title':corr_title.index[i]}
        answers.append(answer)
    chatbot_message = {"answer": answers}
    return chatbot_message
