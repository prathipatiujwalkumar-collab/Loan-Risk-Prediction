from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter

# -------------------------
# Load Dataset
# -------------------------

loader = DataLoader()
df = loader.load_training_data()

# -------------------------
# Validate Dataset
# -------------------------

validator = DataValidator()
validator.validate(df)

# -------------------------
# Feature Engineering
# -------------------------

engineer = FeatureEngineer()
df = engineer.transform(df)

# -------------------------
# Split Dataset
# -------------------------

splitter = DataSplitter()

X_train, X_test, y_train, y_test = splitter.split(df)

print()

print("Training Set")
print(X_train.shape)

print()

print("Testing Set")
print(X_test.shape)

print()

print("Target Distribution")

print(y_train.value_counts(normalize=True))
print()
print(y_test.value_counts(normalize=True))