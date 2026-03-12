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

# HYPERPARAMETERS FOR CNN
cnn_parameters = [{'filters': [128,256], 'dropout': 0.5, 'epochs': 5},
                #   {'filters': [128,256], 'dropout': 0.5, 'epochs': 10},
                #   {'filters': [128,256], 'dropout': 0.5, 'epochs': 3},
                #   {'filters': [32,64], 'dropout': 0.5, 'epochs': 5},
                #   {'filters': [64,128], 'dropout': 0.5, 'epochs': 5},
                #   {'filters': [128,256], 'dropout': 0.3, 'epochs': 5},
                  {'filters': [128,256], 'dropout': 0.7, 'epochs': 5}]

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
    def save(self, name):
        filepath = f'mlp/{name}.pkl'
        joblib.dump(self.mlp, filepath)

# CNN Class
class CNN():
    def __init__(self, train_data, train_labels, tokenizer, max_len, max_words, filters, dropout, epochs):
        self.train_data = train_data
        self.train_labels = train_labels
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_len = max_len
        self.filters = filters
        self.dropout = dropout
        self.epochs = epochs
        self.sequence = self.prepare_sequence()
        self.model = self.build_model()

    # Build Vocabulary With Tokens, Then Transform Texts Into Tokens, Then Padded 0 To Front So Every Sentences Is Same Length, Return Padded
    def prepare_sequence(self):
        sequences = self.tokenizer.texts_to_sequences(self.train_data)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return padded
    
    # Train Model
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.max_len,)))
        model.add(layers.Embedding(self.max_words, output_dim = 100))
        model.add(layers.Conv1D(filters = self.filters[0], kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Conv1D(self.filters[1], kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Flatten())        
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        history = self.model.fit(self.sequence, self.train_labels, epochs=self.epochs, batch_size=32, validation_split=0.2)
        return history
    
    # Save The Model
    def save(self, name):
        self.model.save(f'cnn/{name}.h5')
        
# Train ALL MLP Models Defined In Hyperparameters
def train_all_mlp(train_vec, y_train, test_vec, y_test):
    for i, params in enumerate(mlp_parameters):
        print("Training MLP Version ", i)
        mlp_obj = MLP(params['hidden_layer_sizes'], train_vec, y_train, test_vec, y_test, params['alpha'], params['learning_rate_init'])
        model, acc = mlp_obj.train()
        mlp_obj.save(f'mlp_model_{i}')
        print(f"Test Accuracy for MLP Version {i}: {acc:.4f}")
        evaluate_mlp(model, test_vec, y_test)
    return

# Evaluate MLP - The Model Is Tested on Reserved Test Vec. Print Evaluation Data (Confusion Matrix), (Report). Then Return F1 Score. 
def evaluate_mlp(model, test_vec, y_test):
    y_pred_mlp = model.predict(test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred_mlp))
    report = classification_report(y_test, y_pred_mlp, target_names=['Fake (0)', 'Real (1)'], output_dict=True)
    print("Report")
    print(report)
    return report['macro avg']['f1-score']

# Load All MLP MOdels in mlp and then evaluate them.
def evaluate_all_mlp_models(test_vec, y_test):
    best_score = 0
    best_model_name = None
    best_model = None

    for filename in os.listdir('mlp'):
        if filename.startswith('mlp_model') and filename.endswith('.pkl'):
            model = joblib.load(f'mlp/{filename}')
            score = evaluate_mlp(model, test_vec, y_test)
            if score > best_score:
                best_score = score
                best_model_name = filename
                best_model = model
    print(f"Evaluating Results, Best Model - {best_model_name}")
    joblib.dump(best_model, f'best_models/best_mlp.pkl')
    return best_model

# Get Best MLP Model
def get_best_mlp_model():
    model = joblib.load('best_models/best_mlp.pkl')
    return model

def train_all_cnn(X_train, y_train, X_test, y_test):
    max_words = 5000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    max_len = max(len(seq) for seq in train_sequences)

    # Save Tokenizer For CNN
    joblib.dump(tokenizer, 'cnn/tokenizer.pkl')
    joblib.dump(max_len, 'cnn/max_len.pkl')

    for i, params in enumerate(cnn_parameters):
        print("Training CNN Version ", i)
        cnn_obj = CNN(X_train, y_train, tokenizer, max_len, max_words, params['filters'], params['dropout'], params['epochs'])
        print(cnn_obj.model.summary())
        train_result = cnn_obj.train()
        print(train_result.history['accuracy'])

        X_test_seq = cnn_obj.tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
        evaluate_cnn(cnn_obj.model, X_test_pad, y_test)
        cnn_obj.save(f'cnn_model_{i}')
    return

def evaluate_cnn(model, X_test_pad, y_test):
    y_test_probs = model.predict(X_test_pad)
    y_pred = (y_test_probs > 0.5).astype(int).flatten()
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Report")
    print(report)
    return report['macro avg']['f1-score']

# Load All MLP MOdels in mlp and then evaluate them.
def evaluate_all_cnn_models(X_train, y_train, X_test, y_test):
    best_score = 0
    best_model_name = None
    best_model = None

    tokenizer = joblib.load('cnn/tokenizer.pkl')
    max_len = joblib.load('cnn/max_len.pkl')

    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    for filename in os.listdir('cnn'):
        if filename.startswith('cnn_model') and filename.endswith('.h5'):
            model = models.load_model(f'cnn/{filename}')
            score = evaluate_cnn(model, X_test_pad, y_test)
            if score > best_score:
                best_score = score
                best_model_name = filename
                best_model = model

    print(f"Evaluating Results, Best Model - {best_model_name}")
    best_model.save('best_models/best_cnn.h5')
    return best_model

def get_best_cnn():
    model = models.load_model('best_models/best_cnn.h5')
    return model

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
    train_all_mlp(train_vec, y_train, test_vec, y_test)
    evaluate_all_mlp_models(test_vec, y_test)
    best_mlp = get_best_mlp_model()

    train_all_cnn(X_train, y_train, X_test, y_test)
    evaluate_all_cnn_models(X_train, y_train, X_test, y_test)
    best_cnn = get_best_cnn()

if __name__ == "__main__":
    main()