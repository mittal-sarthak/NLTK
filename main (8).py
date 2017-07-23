import csv
import pandas as pd
import tweepy
import numpy as np
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from time import time
import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



# __future__ import division

consumer_key = 'ej1YHcWxGlk58K8hp5J5MzSWH'
consumer_secret = 'q06qaRoaJsAb9mHpQCxAh8BaMMrtUMYWhCz1mqJlLbvPrieIEe'
access_token = '823112127762284544-eB55CMUip6xdayRzcfU7BfADm81PmSo'
access_token_secret = 'ThJ64wd6jipv853QQERCQoTJ4V1eC2mBjjJHnVLdeiGxU'

class MyStreamListener(StreamListener):
    def on_status(self, status):
        print(status.text)


# OAuth process, using the keys and tokens
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creation of the actual interface, using authentication
api = tweepy.API(auth)
myStream = Stream(auth, MyStreamListener())

finalfile = []

# hold all tweets
alltweets = []

global totaltweets
totaltweets = pd.DataFrame( columns=['id', 'created_at', 'text', 'user_name'])

tweettextastring = ''

#users = ['narendramodi','ArvindKejriwal']
#users = ['narendramodi','ArvindKejriwal','SushmaSwaraj','BarackObama','ShashiTharoor','arunjaitley','quizderek','SalmanSoz','KDSingh_India','abdullah_omar','yadavakhilesh','digvijaya_28','PawarSpeaks','msisodia','realDonaldTrump','HillaryClinton']

users = ['narendramodi','ArvindKejriwal','BarackObama','ShashiTharoor','yadavakhilesh','realDonaldTrump']

def clean_tweet(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w +:\ / \ / \S +)", " ", text).split())

def get_all_tweets(user):
    print "Fetching Data for", user.name

    train = pd.DataFrame(columns=['id', 'created_at', 'text', 'user_name'])
    test = pd.DataFrame(columns=['id', 'created_at', 'text', 'user_name'])

    # make initial request for most recent 200 tweets
    new_tweets = api.user_timeline(screen_name=username, count=200)
    tweet_c = []
    len = 0
    usertweetsfile = []

    for tweet in new_tweets:
        tempfile = []

        tempfile.append(tweet.id)
        tempfile.append(tweet.created_at)
        cleaned = clean_tweet(tweet.text)
        tempfile.append(cleaned.encode("utf-8"))
        tempfile.append(user.name)
        finalfile.append(tempfile)
        usertweetsfile.append(tempfile)
        tweet_c.append(tweet.text)
        len+=1


        # print "Tweet %s" % (tweet.id), tweet

        # save most recent tweets
        alltweets.extend(new_tweets)

    df = pd.DataFrame(data=usertweetsfile, columns=['id', 'created_at', 'text', 'user_name'])

    print len


    global totaltweets
    totaltweets = totaltweets.append(df)



for username in users:
    user = api.get_user(username)
    get_all_tweets(user)

print "Successfully scraped tweet data "

totaltweets.to_csv('Tweets.csv', index=False)

filename = 'Tweets.csv'
tweets = pd.read_csv(filename, encoding='utf-8')

features = tweets["text"].values.astype('U')
labels = tweets["user_name"].values.astype('U')

from sklearn.model_selection import train_test_split

target_train, target_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=42)

count_vect = CountVectorizer()
traindata = count_vect.fit_transform(target_train)

#train
tfidf_transformer = TfidfTransformer()
tfs = tfidf_transformer.fit_transform(traindata)
print "TFS SHAPE"
print tfs

#test
testdata = count_vect.transform(target_test)
new_tfidf = tfidf_transformer.transform(testdata)

#nb
from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB().fit(tfs, label_train)
pred_nb = clf_nb.predict(new_tfidf)
data_final = zip(pred_nb,target_test,label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['pred','text','actual'])
my_solution.to_csv("my_solution_nb.csv")


#sgd
from sklearn.linear_model import SGDClassifier
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(tfs, label_train)
pred_sgd = clf_sgd.predict(new_tfidf)
data_final = zip(target_test, pred_sgd, label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['text', 'pred', 'actual'])
my_solution.to_csv("my_solution_sgd.csv")


#tree
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(tfs, label_train)
pred_tree = clf_tree.predict(new_tfidf)
data_final = zip(target_test, pred_tree, label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['text', 'pred', 'actual'])
my_solution.to_csv("my_solution_tree.csv")


#random forest
from sklearn.ensemble import RandomForestClassifier
clf_forest= RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100 , random_state = 1).fit(tfs, label_train)
pred_forest = clf_forest.predict(new_tfidf)
data_final = zip(target_test, pred_forest, label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['text', 'pred', 'actual'])
my_solution.to_csv("my_solution_forest.csv")


print "PREDICTIONS FOR TWEETS"
print "accuracy NB", np.mean(pred_nb == label_test)
print "accuracy Sgdc", np.mean(pred_sgd == label_test)
print "accuracy tree", np.mean(pred_tree == label_test)
print "accuracy Forest", np.mean(pred_forest == label_test)
#print "accuracy neural", np.mean(pred_neural == label_test)

data_final = zip(pred_nb,target_test,label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['pred','text','actual'])
my_solution.to_csv("my_solution_nb.csv")

data_final = zip(pred_sgd,target_test,label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['pred','text','actual'])
my_solution.to_csv("my_solution_nb.csv")

data_final = zip(pred_tree,target_test,label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['pred','text','actual'])
my_solution.to_csv("my_solution_nb.csv")

data_final = zip(pred_forest,target_test,label_test)
my_solution = pd.DataFrame(data = data_final, columns = ['pred','text','actual'])
my_solution.to_csv("my_solution_nb.csv")














