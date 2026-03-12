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
N_TOPICS = [3,5,7,10,15]

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
    plt.plot(N_TOPICS, bow_lda_perplexities, 'o-')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('Perplexity')
    plt.title('LDA: Perplexity vs Number of Topics')
    plt.tight_layout()
    plt.savefig('lda/perplexity_vs_k.png')
    plt.show()

# Plot And Save Comparison Of Bow And TFIDF With Different Perplexities For LDA
def plot_comparison(bow_lda_perplexities, tfidf_lda_perplexities):
    plt.figure(figsize=(8, 5))
    plt.plot(N_TOPICS, bow_lda_perplexities, 'o-', label='BoW')
    plt.plot(N_TOPICS, tfidf_lda_perplexities, 's-', label='TF-IDF')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('Perplexity')
    plt.title('BoW vs TF-IDF: Perplexity Comparison')
    plt.legend()
    plt.xticks(N_TOPICS)
    plt.tight_layout()
    plt.savefig('lda/bow_vs_tfidf_perplexity.png')
    plt.show()

# Plot Bow Labels Vs Class Label Analysis
def plot_bow_topic_and_label_analysis(result):
    result.T.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Topic')
    plt.ylabel('Average Probability')
    plt.title('Topic Distribution: Real vs Fake Posts')
    plt.legend(['Fake', 'Real'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('lda/topics_by_class.png')
    plt.show()
    plt.close()

# Train LDA From Corpus Bag Of Words
def train_lda(corpus_bow):
    models = []
    perplexities = []
    for n in N_TOPICS:
        lda = LatentDirichletAllocation(n_components=n, random_state=0, max_iter=5, learning_method='batch', verbose = True)
        lda.fit_transform(corpus_bow)
        models.append(lda)
        perplexity = lda.perplexity(corpus_bow)
        perplexities.append(perplexity)
    return models, perplexities

# Main Function
def main():
    # # Read Preprocessed CSV Data - Drop Any NA values
    post_df = pd.read_csv("social-media-release-tokenized.csv")
    post_df = post_df[post_df['tokens_joined'].notna()]
    train_df, test_df = train_test_split(post_df, test_size=0.2, random_state=42, stratify=post_df['class_label'])
    posts = train_df['tokens_joined'].tolist()

    # Build A Corpus And Vectorizer, Then Vectorizer Build Bag of Words And Vocabulary
    vectorizer = CountVectorizer(max_features = 2000)
    posts_bow = vectorizer.fit_transform(posts)
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulary size:", len(vocabulary))
    print("Vocabulary sample:", vocabulary[:300])

    # Bow LDA - Train Bow LDA, Get Highest Bow Model, Draw WordCloud, Save Bow Model And Vectorizer, Plot Bow Perplexisites
    bow_models, bow_lda_perplexities = train_lda(posts_bow)
    highest_index = np.argmin(bow_lda_perplexities)
    best_bow_model = bow_models[highest_index]
    save_wordcloud(best_bow_model, vocabulary)
    joblib.dump(best_bow_model, 'lda/best_bow_lda_model.pkl')
    joblib.dump(vectorizer, 'lda/bow_vectorizer.pkl')
    plot_bow(bow_lda_perplexities)

    # TFIDF LDA - Train TFIDF LDA, Get Highest TFIDF Model, Save TFIDF Model And Vectorizer, Plot TFIDF VS BOW Perplexisites
    tfidf_vec = TfidfVectorizer(max_features=5000)
    posts_tfidf = tfidf_vec.fit_transform(posts)
    tfidf_models, tfidf_lda_perplexities = train_lda(posts_tfidf)
    highest_index = np.argmin(tfidf_lda_perplexities)
    best_tfidf_model = tfidf_models[highest_index]
    joblib.dump(best_tfidf_model, 'lda/best_tfidf_lda_model.pkl')
    joblib.dump(tfidf_vec, 'lda/tfidf_vectorizer.pkl')
    plot_comparison(bow_lda_perplexities, tfidf_lda_perplexities)

    # Topic With Ground Truth Analysis
    best_bow_model = joblib.load('lda/best_bow_lda_model.pkl')
    bow_vectorizer = joblib.load('lda/bow_vectorizer.pkl')
    class_label_df = train_df['class_label']
    docs = best_bow_model.transform(bow_vectorizer.transform(posts))
    number_of_topics = len(docs[0])
    column_names = []
    for i in range(number_of_topics):
        column_names.append(f'topic_{i}')
    best_bow_model_df = pd.DataFrame(docs, columns=column_names)
    best_bow_model_df = best_bow_model_df.join(class_label_df)
    best_bow_model_df_result = best_bow_model_df.groupby('class_label').mean()
    plot_bow_topic_and_label_analysis(best_bow_model_df_result)

# Main
if __name__ == "__main__":
    main()