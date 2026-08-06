from src.data_loader import DataLoader

loader = DataLoader()

df = loader.load_training_data()

print("\nFirst 5 rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)