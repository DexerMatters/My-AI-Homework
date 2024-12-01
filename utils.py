import yaml

config = None

def load_config():
    with open("./cfg.yml", 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config():
    if config is None:
        config = load_config()
    return config