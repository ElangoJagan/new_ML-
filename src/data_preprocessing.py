## starting Preprocessing step 
import pandas as pd
import os
import nltk
import re
import yaml
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

## download ntlk data
nltk.download('stopwords')
nltk.download('punkt')

def load_config(config_path = 'config/config.yaml'):
    with open(config_path, 'r')as f:
        config= yaml.safe_load(f)
    return config

def clean_text(text):
    
    ps= PorterStemmer()
    
    #1.make it to lowercase
    text= text.lower()
    
    
    #2.remove numbers and punctuation 
    text = re.sub(r'[^a-z\s]', '', text)
    
    
    #3.remove stopwords and apply stemming 
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def preprocess_data(df, config):
    
    ## keep only useful columns 
    df= df[['v1', 'v2']].copy()
    
    ## rename column
    df.columns = ['label', 'message']
    
    ##encode the label spam = 1, ham=0
    df['label'] = df['label'].map({'ham':0, 'spam':1})   
    
    ## clean the message 
    print('cleaning the messages')
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    ## save the processed data   
    processed_path = config['data']['processed_path']  
    df.to_csv(processed_path, index = False)
    
    
    print('preprocessing done')
    print(f'shape {df.shape}')
    print ('sample as below : ')
    print((df[['message', 'cleaned_message', 'label']].head(3)))
    
    return df

if __name__ == "__main__":
    
    ## load_config: 
    config = load_config()
    
    ##load raw dataset first: 
    raw_df = pd.read_csv(config['data']['raw_path'], encoding = 'latin-1')
    
    ## preprocess
    df = preprocess_data(raw_df, config)
    
    
    
    
    
     