import os 
import sys

# add proejct root to the path 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import load_config, load_model, predict

def run_prediction_pipeline(message):
    # load config 
    config = load_config()
    
    ## load model and vectorizer
    model, vectorizer = load_model(config)
    
    #predict 
    
    result, confidence = predict(message, model, vectorizer)
    
    return result, confidence

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Starting Prediction Pipeline")
    print("=" * 60)
    
    # Test messages
    test_messages = [
        "Congratulations! You've won a free iPhone!",
        "Hey, are we still meeting for lunch tomorrow?",
        "FREE entry! Win £1000 cash prize. Text WIN now!",
    ]
    
    for msg in test_messages:
        result, confidence = run_prediction_pipeline(msg)
        print(f"\n📩 Message    : {msg[:50]}...")
        print(f"🎯 Result     : {result}")
        print(f"📊 Confidence : {confidence:.2f}%")
        print("-" * 60)