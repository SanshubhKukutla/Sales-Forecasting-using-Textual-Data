from src.data_preprocessing import load_data, preprocess_text
from src.model import load_model, generate_predictions
from src.train import train_regression
from src.predict import predict_sales

# Load and preprocess data
data = load_data("data/sales_data.csv")
inputs = preprocess_text(data['review_text'], "google/gemma-2B")

# Load Gemma model and generate predictions
gemma_model = load_model()
predictions = generate_predictions(gemma_model, inputs)

# Train regression model
sales_model = train_regression(predictions, data['sales'])

# Predict future sales
future_reviews = ["Great product!", "Disappointing service."]
future_sales = predict_sales(sales_model, future_reviews, gemma_model, inputs)
print(future_sales)
