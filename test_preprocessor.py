from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.preprocessor import Preprocessor

loader = DataLoader()
df = loader.load_training_data()

validator = DataValidator()
validator.validate(df)

engineer = FeatureEngineer()
df = engineer.transform(df)

splitter = DataSplitter()
X_train, X_test, y_train, y_test = splitter.split(df)

preprocessor = Preprocessor()

X_train_processed = preprocessor.fit_transform(X_train)
preprocessor.save("models/preprocessor.joblib")

feature_names = preprocessor.get_feature_names()

print("\nFirst 15 Features:\n")

print(feature_names[:15])
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)