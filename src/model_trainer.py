import pandas as pd
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_config(config_path = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(X, Y, config):
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    x_train, x_test, y_train, y_test = train_test_split(X ,Y, test_size=test_size, random_state=random_state)
    print(f'train size : {x_train.shape[0]} | test size : {x_test.shape[0]}')
    
    
    ##Train the Naive Bayes Model: 
    print('Training Naive Baiyes Model ....................')
    model = MultinomialNB()
    model.fit( x_train, y_train)

    #Evaluate
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred )
    precisionscore = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f'Model Perfromance..........')
    print(f'accuracy score: {accuracy:.4f}')
    print(f'precision score: {precisionscore:.4f}')
    print(f'recall score: {recall:.4f}')
    print(f'f1 score : {f1:.4f}')
    print(f'confusion matrix: {cm}')


    ## save model 
    model_path = config['model']['model_path']
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'model saved to {model_path}')

    return model

if __name__ == "__main__":
    # load config 
    config = load_config()
    
    ## LoadProcessed Data:
    processed_path = config['data']['processed_path']
    df = pd.read_csv(processed_path)
    
    ## Load vectorizer:
    vectorizer_path = config['model']['vectorizer_path']
    with open (vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    ##Transform Data:
    df['cleaned_message'] = df['cleaned_message'].fillna('')
    X = vectorizer.transform(df['cleaned_message']).toarray()
    Y= df['label'].values
    
    #Train Model
    model = train_model(X, Y, config)
        
