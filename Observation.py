import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    "text": [
        "The market is booming with new opportunities.",
        "Investors are worried about potential losses.",
        "Stock prices are stable for now.",
        "A major crash in the market is expected.",
        "The economy is showing positive growth."
    ],
    "sentiment_score": [0.8, -0.7, 0.1, -0.9, 0.9]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Text Preprocessing: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Target variable: Sentiment scores
y = df['sentiment_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sentiment scores for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Test with a new sentence
new_text = ["The market is uncertain with mixed signals."]
new_features = vectorizer.transform(new_text)
predicted_score = model.predict(new_features)
print(f"Predicted Sentiment Score: {predicted_score[0]:.2f}")
