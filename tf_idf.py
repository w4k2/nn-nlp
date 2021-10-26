from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(X_train, X_test, num_features=100):
    tfidf_vectorizer = TfidfVectorizer(max_features=num_features)
    tfidf_vectorizer.fit(X_train)
    train_features = tfidf_vectorizer.transform(X_train)
    test_features = tfidf_vectorizer.transform(X_test)
    return train_features, test_features
