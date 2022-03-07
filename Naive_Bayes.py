import re, string, json, nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


nltk.download('stopwords')



class NaiveBayes:
    def __init__(self):
        self.loglikelihood = {}
        self.logprior = 0
        

    def train(self, x_train, y_train):
        freqs = self.__count_tweets(x_train, y_train)
        vocab = set([pair[0] for pair in freqs.keys()])
        V = len(vocab)

        N_pos = N_neg = 0
        for pair in freqs.keys():
            if pair[1] > 0:
                N_pos += freqs[pair]
            else:
                N_neg += freqs[pair]

        D = len(self.y_train)
        D_pos = sum(self.y_train)
        D_neg = D - D_pos
        self.logprior = np.log(D_pos) - np.log(D_neg)

        for word in vocab:
            freq_pos = self.__lookup(freqs, word, 1)
            freq_neg = self.__lookup(freqs, word, 0)
            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)

            self.loglikelihood[word] = np.log(p_w_pos / p_w_neg)

        print("Done training...")



    def predict(self, tweet):
        if len(self.loglikelihood) < 1:
            print("You have to train the model before trying to predict...")
            print("Or you can call the load method to load a model file")
        else:
            word_l = self.__process_tweet(tweet)
            p = 0
            p += self.logprior
            for word in word_l:
                if word in self.loglikelihood:
                    p += self.loglikelihood[word]

            return p



    def test(self, x_test, y_test):
        if len(self.loglikelihood) < 1:
            print("You have to train the model before trying to predict...")
            print("Or you can call the load method to load a model file")
        else:
            accuracy = 0
            y_hats = []
            for tweet in x_test:
                if self.predict(tweet) > 0:
                    y_hat_i = 1
                else:
                    y_hat_i = 0

                y_hats.append(y_hat_i)
            error = np.sum(np.abs(y_hats - y_test))/len(y_test)
            accuracy = 1 - error
            
            print(f"The model's accuracy is {accuracy * 100}%...")



    def save(self):
        if len(self.loglikelihood) < 1:
            print("You have to train the model before saving it...")
            print("Or you can call the load method to load a model file")
        else:
            total_objects = {
                'loglikelihood': self.loglikelihood,
                'logprior': self.logprior
            }

            with open('model.json', 'w') as file:
                json.dump(total_objects, file)

            print("Model saved successfully")



    def load(self, file_path):
        try:
            with open(file_path, 'r') as file:
                model_parameters = json.load(file)
            self.loglikelihood = model_parameters['loglikelihood']
            self.logprior = model_parameters['logprior']

            print("Model loaded successfully")
        except:
            print("Please provide a valid filepath, the file should be a json file")



    def __count_tweets(self, tweets, ys):
        result = {}
        for y, tweet in zip(ys, tweets):
            for word in self.__process_tweet(tweet):
                pair = (word, y)
                if pair in result:
                    result[pair] += 1
                else:
                    result[pair] = 1
        return result



    def __lookup(self, freqs, word, label):
        n = 0
        pair = (word, label)
        if (pair in freqs):
            n = freqs[pair]
        return n



    def __process_tweet(self, tweet):
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')

        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)

        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)

        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)

        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                reduce_len=True)
                                
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        whitelist = ["n't", "not", "no"]
        for word in tweet_tokens:
            if ((word not in stopwords_english or word in whitelist) and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
                stem_word = stemmer.stem(word)  # stemming word
                tweets_clean.append(stem_word)

        return tweets_clean