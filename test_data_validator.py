from src.data_loader import DataLoader
from src.data_validator import DataValidator


loader = DataLoader()

validator = DataValidator()

df = loader.load_training_data()

validator.validate(df)

print("\nAll validations completed successfully.")