from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load only two categories: rec.autos and comp.sys.mac.hardware
categories = ["rec.autos", "comp.sys.mac.hardware"]
data = fetch_20newsgroups(
    subset="train", categories=categories, remove=("headers", "footers", "quotes")
)
# print(data)

# # Use TfidfVectorizer to convert text to vectors.
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data.data)
y = data.target

# # Train a Logistic Regression classifier to predict the category.
model = LogisticRegression(max_iter=1000)
model.fit(X, y)


# # Print accuracy and show 5 most important words per class.
y_predics = model.predict(X)
print("Accuracy:", accuracy_score(y, y_predics))
words = vectorizer.get_feature_names_out()

for i, category in enumerate(categories):
    top_indices = (
        model.coef_[0].argsort()[-5:] if i == 1 else model.coef_[0].argsort()[:5]
    )
    print(f"Top 5 words for {category}: {words[top_indices]}")
