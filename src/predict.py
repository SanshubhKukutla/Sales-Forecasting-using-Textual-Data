def predict_sales(model, future_reviews, gemma_model, tokenizer):
    inputs = tokenizer(future_reviews, return_tensors="pt", padding=True, truncation=True)
    outputs = gemma_model.generate(inputs['input_ids'], max_length=50)
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    future_sales = model.predict(predictions)
    return future_sales
