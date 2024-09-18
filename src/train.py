from sklearn.linear_model import LinearRegression

def train_regression(predictions, sales):
    model = LinearRegression()
    model.fit(predictions, sales)
    return model
