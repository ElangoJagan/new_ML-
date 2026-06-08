import pandas as pd
import yaml
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

nltk.download('stopwords', quiet= True)

def load_config(config_path = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def clean_text(text):
    ps = PorterStemmer()
    
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]','',text)
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def load_model(config):
    # load model
    model_path = config['model']['model_path']
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # load vectorizer: 
    vectorizer_path  = config['model']['vectorizer_path']
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer


def predict(message, model, vectorizer):
    
    # clean the message 
    cleaned = clean_text(message)
    
    ## transform using vectorize
    features = vectorizer.transform([cleaned]).toarray()
    
    ## prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    result = 'Spam' if prediction ==1 else 'HAM (not spam)'
    
    if prediction == 1:
        confidence = probability[1] * 100
    else:
        confidence = probability[0] * 100
    
    return result, confidence

if __name__ == '__main__':
    
    config= load_config()
    model, vectorizer = load_model(config)
    
    
    # Test messages
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "FREE entry! Win £1000 cash prize. Text WIN to 80085 now!",
        "Can you please send me the project files?",
        "URGENT: Your bank account has been compromised. Call now!"
    ]
    
    print("🔍 Testing Predictor:\n")
    print("-" * 60)
    
    for msg in test_messages:
        result, confidence = predict(msg, model, vectorizer)
        print(f"📩 Message : {msg[:50]}...")
        print(f"🎯 Result  : {result}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print("-" * 60)
    