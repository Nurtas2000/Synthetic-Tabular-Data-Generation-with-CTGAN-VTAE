import yaml
from ctgan import CTGANSynthesizer
from data_preprocessing import load_and_preprocess
from utils import save_model

def train_ctgan(config_path, dp_config_path=None):
    # Load configs
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dp_config = None
    if dp_config_path:
        with open(dp_config_path) as f:
            dp_config = yaml.safe_load(f)
            config.update(dp_config)
    
    # Load and preprocess data
    data, discrete_columns = load_and_preprocess()
    
    # Initialize and train model
    ctgan = CTGANSynthesizer(**config)
    ctgan.fit(data, discrete_columns)
    
    return ctgan
