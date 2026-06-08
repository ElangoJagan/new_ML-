## Kadavulae Thunai 

import pandas as pd
import yaml
import os

def load_config(config_path = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    raw_path = config['data']['raw_path']
    
    df = pd.read_csv(raw_path, encoding='latin-1')
    
    print('DATA loaded successfully')
    print(f'showing the shape of df as {df.shape}')
    print('first 5 rows')
    print(df.head(5))
    print('null counts as follows:')
    df.isnull().sum()
    
    return df

if __name__ == "__main__":
    config = load_config()
    df = load_data(config)
