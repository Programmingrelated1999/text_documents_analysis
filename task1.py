# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# CNN Class
class CNN():
    def __init__(self, train_data, train_labels, max_words):
        self.train_data = train_data
        self.train_labels = train_labels
        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.sequence = self.prepare_sequence()
        self.max_len = 0

    # Build Vocabulary With Tokens, Then Transform Texts Into Tokens, Then Padded 0 To Front So Every Sentences Is Same Length, Return Padded
    def prepare_sequence(self):
        self.tokenizer.fit_on_texts(self.train_data)
        sequences = self.tokenizer.texts_to_sequences(self.train_data)
        self.max_len = max(len(seq) for seq in sequences)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return padded
    
    # Train Model
    def train_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Embedding(self.max_words, output_dim = 100, input_length = self.max_len))
        
# Split Test And Train Data
def split_test_train(post_df, test_size):
    print("Start of Splitting Test And Train Data")
    X_train, X_test, y_train, y_test = train_test_split(
        post_df['tokens_joined'],
        post_df['class_label'].astype(int),
        test_size=test_size,
        random_state=42,
        stratify=post_df['class_label']
    )
    train_data = pd.DataFrame({'token_joined': X_train, 'class_label': y_train})
    test_data = pd.DataFrame({'token_joined': X_test, 'class_label': y_test})
    train_data.to_csv("traindata.csv", index=False)
    test_data.to_csv("testdata.csv", index=False)
    print("End of Splitting Test And Train Data")
    return X_train, X_test, y_train, y_test

# Vectorize Data - For Train Vectorizer Learn, For Test Do Not Learn. 
def vectorize_data(train_data, test_data, vectorizer):
    train_vec = vectorizer.fit_transform(train_data)
    test_vec = vectorizer.transform(test_data)
    return train_vec, test_vec

# MLP Function 
def simpleMLP(hidden_layers, X_train_vec, y_train, X_test_vec, y_test):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=500,
        activation='relu',
        solver='adam',
        random_state=42,
        verbose=True
    )
    mlp.fit(X_train_vec, y_train)
    accuracy = mlp.score(X_test_vec, y_test)
    joblib.dump(mlp, 'tfidf.pkl') 
    return mlp, accuracy

# Define Main Function
def main():
    print("Start of Program")
    # Load CSV Data - Drop Tokens Joined is NA.
    post_df = pd.read_csv('social-media-release-tokenized.csv')
    post_df = post_df[post_df['tokens_joined'].notna()]

    # Split Test Train Data. 
    X_train, X_test, y_train, y_test = split_test_train(post_df, 0.2)

    # Build A TFIDF Vectorizer - Set Max Features to Reduce From 37000+ Features To 5000
    tfidf_vec = TfidfVectorizer(max_features=5000)
    X_train_vec, X_test_vec = vectorize_data(X_train, X_test, tfidf_vec)

    # For saving Time Rerun
    # mlp, accuracy = simpleMLP((128,64), X_train_vec, y_train, X_test_vec, y_test)
    # load the 
    # mlp = joblib.load('tfidf.pkl')
    # y_pred = mlp.predict(X_test_vec)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    basic_cnn = CNN(X_train, y_train, 5000)

if __name__ == "__main__":
    main()