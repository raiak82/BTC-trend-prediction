# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
from textblob import TextBlob



# Importing the dataset

dataset = pd.read_csv('train_tweets.csv',encoding="ISO-8859-1",parse_dates=['created_date'])
dataset['text'] = dataset['text'].str.replace(r'[^\x00-\x7F]+', '')
dataset['text'] = dataset['text'].str.replace(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', " ")




def analyze_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

dataset['SA'] = np.array([ analyze_sentiment(tweet) for tweet in dataset['text'].astype(str) ]) 

columns_to_be_removed=['tweet_id','text','retweet_count','favorite_count','follower_count','account']
dataset.drop(columns_to_be_removed, axis=1, inplace=True)


dataset['created_date'] = pd.to_datetime(dataset['created_date'], errors='coerce')
dataset['created_date'] = pd.to_datetime(dataset['created_date']).dt.date




#df2 = dataset['created_date'].astype(str)
#dataset['created_date'] = dataset['created_date'].str.split(" ")
#dataset['created_date'].shape
#
#for i in range(0,dataset.shape[0]):
#    dataset['created_date'][i]=dataset['created_date'][i][0]

#
df1=dataset.groupby(['created_date','SA']).size().reset_index().groupby(['created_date','SA'])[[0]].max().unstack()

#changing from multi index dataset to single index dataset
mi= df1.columns
ind = pd.Index([e[0] + e[1] for e in mi.tolist()])
df1.columns = ind
df1=df1.fillna(0)
#rename columns of dataset from -1,0,1 to Negative, neutral and positive
df1.rename(columns={-1:'Negative_tweets',0:'Neutral_tweets',1:'Positive_tweets'}, inplace= True)