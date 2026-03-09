import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Write To Feature Counts File
def write_feature_counts_file(vector, feature_names):
    print("Writing To features")
    counts = vector.sum(axis=0).A1
    with open("feature_counts.txt", "w", encoding="utf-8") as f:
        for word, count in zip(feature_names, counts):
            f.write(f"{word} {int(count)}\n")

# Define Main Function
def main():
    # Load CSV Data
    post_df = pd.read_csv('social-media-release-tokenized.csv')
    post_df = post_df[post_df['tokens_joined'].notna()]

    print("Splitting Task")
    # Split Test Train Data. 
    X_train, X_test, y_train, y_test = train_test_split(
        post_df['tokens_joined'],
        post_df['class_label'].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=post_df['class_label']
    )
    print("End Splitting Task")

    # Build A Vectorizer - Set Max Features to Reduce From 37000+ Features To 5000
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()

    return 


if __name__ == "__main__":
    main()