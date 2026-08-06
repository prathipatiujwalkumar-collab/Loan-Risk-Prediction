from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer

# Load Data
loader = DataLoader()
df = loader.load_training_data()

# Validate
validator = DataValidator()
validator.validate(df)

# Engineer Features
engineer = FeatureEngineer()

new_df = engineer.transform(df)

print("\nOriginal Shape:", df.shape)
print("New Shape:", new_df.shape)

print("\nNew Features Added:\n")

print(
    new_df[
        [
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "CREDIT_GOODS_RATIO",
            "EMPLOYMENT_AGE_RATIO",
            "CHILDREN_RATIO",
            "INCOME_PER_FAMILY_MEMBER",
            "ANNUITY_CREDIT_RATIO",
            "GOODS_INCOME_RATIO"
        ]
    ].head()
)