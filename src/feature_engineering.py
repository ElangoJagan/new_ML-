'''Feature engineering means before progressing to ML, we needs to 
convert text to number (ie.vectors), that will be achieved in this step'''

import pandas as pd
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def load_config(config_path = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_features(df, config):
    
    ## initialize Vectorizer TF-IDF:
    vectorizer = TfidfVectorizer(max_features = 3000)
    
    ##fit and transform the cleaned messages:
    print('Extracting TF-IDF features...........')
    df['cleaned_message'] = df['cleaned_message'].fillna('')
    X= vectorizer.fit_transform(df['cleaned_message']).toarray()
    
    
    Y= df['label'].values
    
    print(f'Feature extractuion done . and its shape is {X.shape}')
    print(f'total vocabulary size is {len(vectorizer.vocabulary_)}')
    
    
    ##save vector to artifacts:
    vectorizer_path = config['model']['vectorizer_path']
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer,f)
    print(f'vectors saved to {vectorizer_path}')
    
    return X, Y
    
    
if __name__ == "__main__":
    # load config
    config = load_config()
    
    # load processed data 
    processed_path = config['data']['processed_path']
    df = pd.read_csv(processed_path)
    
    ##Extract features:
    X ,Y = extract_features(df, config)
    
    print(f"\n🔍 Sample feature shape: {X[0].shape}")
    print(f"📊 Label distribution:\n  Ham: {sum(Y==0)}\n  Spam: {sum(Y==1)}")