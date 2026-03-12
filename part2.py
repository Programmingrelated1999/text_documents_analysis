# Import Libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

# Declare Global Variables
n_topics = [3,5,7,10,15]

# Save Wordcloud Images
def save_wordcloud(model, vocabulary):
    for index, topic in enumerate(model.components_):
        arr = np.array(topic)
        highest_indexes = arr.argsort()[-10:][::-1]
        topics_words = [vocabulary[i] for i in highest_indexes]
        text = ' '.join([str(i) for i in topics_words]) 
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.savefig(f'lda/wordcloud/topic{index}.png')
        plt.close()

# Plot And Save Bow Experiment With Different Perplexities For LDA
def plot_bow(bow_lda_perplexities):
    plt.figure(figsize=(8, 5))
    plt.plot(n_topics, bow_lda_perplexities, 'o-')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('Perplexity')
    plt.title('LDA: Perplexity vs Number of Topics')
    plt.tight_layout()
    plt.savefig('lda/perplexity_vs_k.png')
    plt.show()

# Plot And Save Comparison Of Bow And TFIDF With Different Perplexities For LDA
def plot_comparison(bow_lda_perplexities, tfidf_lda_perplexities):
    plt.figure(figsize=(8, 5))
    plt.plot(n_topics, bow_lda_perplexities, 'o-', label='BoW')
    plt.plot(n_topics, tfidf_lda_perplexities, 's-', label='TF-IDF')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('Perplexity')
    plt.title('BoW vs TF-IDF: Perplexity Comparison')
    plt.legend()
    plt.xticks(n_topics)
    plt.tight_layout()
    plt.savefig('lda/bow_vs_tfidf_perplexity.png')
    plt.show()

def train_lda(corpus_bow):
    models = []
    perplexities = []
    for n in n_topics:
        lda = LatentDirichletAllocation(n_components=n, random_state=0, max_iter=5, learning_method='batch', verbose = True)
        lda.fit_transform(corpus_bow)
        models.append(lda)
        perplexity = lda.perplexity(corpus_bow)
        perplexities.append(perplexity)
    return models, perplexities

# Main Function
def main():
    # Read Preprocessed CSV Data - Drop Any NA values
    post_df = pd.read_csv("social-media-release-tokenized.csv")
    post_df = post_df[post_df['tokens_joined'].notna()]
    train_df, test_df = train_test_split(post_df, test_size=0.2, random_state=42, stratify=post_df['class_label'])

    # Build A Corpus And Vectorizer, Then Vectorizer Build Bag of Words And Vocabulary
    posts = train_df['tokens_joined'].tolist()
    vectorizer = CountVectorizer(max_features = 2000)
    posts_bow = vectorizer.fit_transform(posts)
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulary size:", len(vocabulary))
    print("Vocabulary sample:", vocabulary[:300])

    # LDA
    print("Start of LDA")
    bow_models, bow_lda_perplexities = train_lda(posts_bow)

    highest_index = np.argmin(bow_lda_perplexities)
    highest_bow_model = bow_models[highest_index]
    save_wordcloud(highest_bow_model, vocabulary)
    
    joblib.dump(highest_bow_model, 'lda/best_lda_model.pkl')
    joblib.dump(vectorizer, 'lda/bow_vectorizer.pkl')

    plot_bow(bow_lda_perplexities)

    tfidf_vec = TfidfVectorizer(max_features=5000)
    posts_tfidf = tfidf_vec.fit_transform(posts)
    tfidf_models, tfidf_lda_perplexities = train_lda(posts_tfidf)

    plot_comparison(bow_lda_perplexities, tfidf_lda_perplexities)

# Main
if __name__ == "__main__":
    main()