import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #ignore gpu if there is one

def read_data(real_news, fake_news):
    real = pd.read_csv(real_news)
    fake = pd.read_csv(fake_news)
    return real,fake


def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text
def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)


    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

if __name__ == "__main__":

    nltk.download('stopwords')

    #set the path for the fake news and real news paths
    real_news = "../fake_news_data/data/True.csv"
    fake_news = "../fake_news_data/data/Fake.csv"
    embedding_file = "../archive/glove.twitter.27B.100d.txt"

    real, fake = read_data(real_news, fake_news)

    # checking for null values in data
    assert (real.isnull().sum().title) == 0, "null values in true data"
    assert (fake.isnull().sum().title) == 0, "null values in fake data"

    #adding labels to the data 
    real['category'] = 1
    fake['category'] = 0

    df = pd.concat([real,fake])
    df["text"] = df["text"] + " " + df["title"]
    del df["title"]
    del df["subject"]
    del df["date"]
    print(df.head())

    df['text']=df['text'].apply(denoise_text)

    x_train,x_test,y_train,y_test = train_test_split(df.text,df.category,random_state = 0)
    print(y_test)

    tokenizer = text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(x_train)
    tokenized_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(tokenized_train, maxlen=300)

    tokenized_test = tokenizer.texts_to_sequences(x_test)
    X_test = sequence.pad_sequences(tokenized_test, maxlen=300)

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(10000, len(word_index))
    embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= 10000: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    batch_size = 256
    epochs = 2
    embed_size = 100

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

    model = Sequential()
   
    model.add(Embedding(10000, output_dim=embed_size, weights=[embedding_matrix], input_length=300, trainable=False))
    
    model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
    model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
    model.add(Dense(units = 32 , activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(x_train, y_train, batch_size = batch_size , validation_data = (X_test,y_test) , epochs = epochs , callbacks = [learning_rate_reduction])
    print("Accuracy on Training Data is - " , model.evaluate(x_train,y_train)[1]*100 , "%")
    print("Accuracy on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")




    pred = model.predict(X_test)
    pred[:5]    

    print(pred[:10]) #need to convert these with a function to map the value to either 0 or 1

    pred_int = list(map(lambda x: 0 if x < 0.5 else 1, pred))

    print(pred_int[:10])
    print(classification_report(y_test, pred_int, target_names = ['Fake','Not Fake'])) #checking with pred_int instead of pred


    cm = confusion_matrix(y_test,pred_int)
    cm = pd.DataFrame(cm , index = ['Fake','Original'] , columns = ['Fake','Original'])
    
    print(cm)


