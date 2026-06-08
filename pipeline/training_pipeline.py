import os
import sys

## add project root to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import load_config, load_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import  extract_features
from src.model_trainer import train_model

def run_training_pipeline():
    print('='*60)
    print('Starting Training Pipeline ')
    print('='*60)
    
    # step 1 Load config
    print('Loading Config...')
    config = load_config()
    print('Config Loading Completed')
    
    #data ingestion
    print('Loading data')
    df = load_data(config)
    print('Data Loaded')
    
    #pre-processing
    df= preprocess_data(df, config)
    print('preprocessing done')
    
    #Feature Engineering
    X , Y = extract_features(df, config)
    print('features extracted')
    
    #Model_Training
    model = train_model(X,Y,config)
    print('model trained ')
    
    
    print("\n" + "=" * 60)
    print("🎉 Training Pipeline Completed Successfully!")
    print("=" * 60)
    
if __name__ == "__main__":
    run_training_pipeline()