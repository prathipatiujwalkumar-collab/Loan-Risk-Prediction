from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.trainer import ModelTrainer
from src.evaluator import Evaluator

# Load
loader = DataLoader()
df = loader.load_training_data()

# Validate
validator = DataValidator()
validator.validate(df)

# Feature Engineering
engineer = FeatureEngineer()
df = engineer.transform(df)

# Split
splitter = DataSplitter()

X_train, X_test, y_train, y_test = splitter.split(df)

# Train
from sklearn.linear_model import LogisticRegression

trainer = ModelTrainer(

    model=LogisticRegression(

        random_state=42,

        max_iter=2000,

        class_weight="balanced",

        solver="lbfgs",
    )
)


trainer.train(X_train, y_train)

# Predict
pred = trainer.predict(X_test)

prob = trainer.predict_proba(X_test)

# Evaluate
evaluator = Evaluator()

results = evaluator.evaluate(
    y_test,
    pred,
    prob,
)


print()

print("=" * 50)

for k, v in results.items():

    print(k)

    print(v)

    print()
print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

for key, value in results.items():

    print(f"\n{key}")

    print("-" * 40)

    print(value)

trainer.save()