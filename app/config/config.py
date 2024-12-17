import yaml
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    environment: str
    one_step_model_labs_path: str
    one_step_model_nolabs_path: str

    class Config:
        yaml_file = Path('./app/config.yaml')

# Function to load the YAML file into Pydantic settings
def load_settings_from_yaml() -> Settings:
    with open(Settings.Config.yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return Settings(**yaml_data)

settings = load_settings_from_yaml()
