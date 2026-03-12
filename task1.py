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
import os

# HYPERPARAMETERS FOR MLP
mlp_parameters = [{'hidden_layer_sizes': (64,32), 'alpha': 0.0001, 'learning_rate_init': 0.001},
                #   {'hidden_layer_sizes': (128,64), 'alpha': 0.0001, 'learning_rate_init': 0.001},
                #   {'hidden_layer_sizes': (256,128), 'alpha': 0.0001, 'learning_rate_init': 0.001},
                #   {'hidden_layer_sizes': (64,32), 'alpha': 0.001, 'learning_rate_init': 0.001},
                #   {'hidden_layer_sizes': (64,32), 'alpha': 0.01, 'learning_rate_init': 0.001},
                #   {'hidden_layer_sizes': (64,32), 'alpha': 0.0001, 'learning_rate_init': 0.01},
                  {'hidden_layer_sizes': (64,32), 'alpha': 0.0001, 'learning_rate_init': 0.1}]

# MLP Class
class MLP():
    def __init__(self, hidden_layers, X_train_vec, y_train, X_test_vec, y_test, alpha, learning_rate):
        self.hidden_layers = hidden_layers
        self.X_train_vec = X_train_vec
        self.y_train = y_train
        self.X_test_vec = X_test_vec
        self.y_test = y_test
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layers,max_iter=500,activation='relu',solver='adam',alpha=alpha, learning_rate_init=learning_rate, random_state=42,verbose=True)
        
    # Train The Model - Returns Accuracy and MLP
    def train(self):
        self.mlp.fit(self.X_train_vec, self.y_train)
        accuracy = self.mlp.score(self.X_test_vec, self.y_test)
        return self.mlp, accuracy
    
    # Save The Model
    def save(self, v):
        filepath = f'mlp/{v}.pkl'
        joblib.dump(self.mlp, filepath)

# CNN Class
class CNN():
    def __init__(self, train_data, train_labels, max_words):
        self.train_data = train_data
        self.train_labels = train_labels
        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words=max_words)
        self.max_len = 0
        self.sequence = self.prepare_sequence()
        self.model = self.build_model()

    # Build Vocabulary With Tokens, Then Transform Texts Into Tokens, Then Padded 0 To Front So Every Sentences Is Same Length, Return Padded
    def prepare_sequence(self):
        self.tokenizer.fit_on_texts(self.train_data)
        sequences = self.tokenizer.texts_to_sequences(self.train_data)
        self.max_len = max(len(seq) for seq in sequences)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return padded
    
    # Train Model
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.max_len,)))
        model.add(layers.Embedding(self.max_words, output_dim = 100))
        model.add(layers.Conv1D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv1D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())        
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        history = self.model.fit(self.sequence, self.train_labels, epochs=5, batch_size=32, validation_split=0.2)
        return history
        
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

# Define Main Function
def main():
    # Load CSV Data - Drop Tokens Joined is NA.
    post_df = pd.read_csv('social-media-release-tokenized.csv')
    post_df = post_df[post_df['tokens_joined'].notna()]

    # Check if Train or Test Data Exists 
    if os.path.isfile("traindata.csv") and os.path.isfile("testdata.csv"):
        train_df = pd.read_csv('traindata.csv')
        test_df = pd.read_csv('testdata.csv')
        X_train = train_df['token_joined']
        y_train = train_df['class_label']
        X_test = test_df['token_joined']
        y_test = test_df['class_label']
    else:
        # Split Test Train Data. 
        X_train, X_test, y_train, y_test = split_test_train(post_df, 0.2)

    # Check If MLP Vectorizer Is Already There, If There Load And Transform Training And Test Data. Else Vectorizer Fit The Load/Train Data.
    if os.path.isfile("mlp/vec.pkl"):
        tfidf_vec = joblib.load('mlp/vec.pkl')
        train_vec = tfidf_vec.transform(X_train)
        test_vec = tfidf_vec.transform(X_test)
    else:
        # Build A TFIDF Vectorizer - Set Max Features to Reduce From 37000+ Features To 5000
        tfidf_vec = TfidfVectorizer(max_features=5000)
        train_vec = tfidf_vec.fit_transform(X_train)
        test_vec = tfidf_vec.transform(X_test)
        joblib.dump(tfidf_vec, 'mlp/vec.pkl')

    # Start Of MLP Training Across All Hyperparameters
    results = []
    for i, params in enumerate(mlp_parameters):
        print("Training MLP Version ", i)
        mlp_obj = MLP(params['hidden_layer_sizes'], train_vec, y_train, 
                      test_vec, y_test, params['alpha'], params['learning_rate_init'])
        print(mlp_obj)
        model, acc = mlp_obj.train()
        results.append({'params': params, 'accuracy': acc})
        mlp_obj.save(f'mlp_model_{i}')
        print(f"Test Accuracy: {acc:.4f}")
        y_pred_mlp = model.predict(test_vec)
        print(confusion_matrix(y_test, y_pred_mlp))
        print(classification_report(y_test, y_pred_mlp, target_names=['Fake (0)', 'Real (1)']))

    # CNN
    # basic_cnn = CNN(X_train, y_train, 5000)
    # print(basic_cnn.model.summary())
    # train_model = basic_cnn.train()
    # print(train_model.history['accuracy'])

    # # CNN Evaluation on Test Set
    # X_test_seq = basic_cnn.tokenizer.texts_to_sequences(X_test)
    # X_test_pad = pad_sequences(X_test_seq, maxlen=basic_cnn.max_len)
    # y_test_probs = basic_cnn.model.predict(X_test_pad)
    # y_pred = (y_test_probs > 0.5).astype(int).flatten()

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()