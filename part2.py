# Import Libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

n_topics = [5,6,7,8,9,10,15,20]
bow_lda_perplexity = []
bow_models = []

def main():
    # Read Preprocessed CSV Data
    post_df = pd.read_csv("social-media-release-tokenized.csv")
    post_df = post_df[post_df['tokens_joined'].notna()]
    post_df = post_df[:2000]

    # Build A Corpus And Vectorizer, Then Vectorizer Build Bag of Words And Vocabulary
    posts = post_df['tokens_joined'].tolist()
    vectorizer = CountVectorizer(max_features = 2000)
    posts_bow = vectorizer.fit_transform(posts)
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulary size:", len(vocabulary))
    print("Vocabulary sample:", vocabulary[:300])

    # LDA
    print("Start of LDA")
    for n in n_topics:
        lda = LatentDirichletAllocation(n_components=n, random_state=0, max_iter=5, verbose = True)
        docs = lda.fit_transform(posts_bow)
        topics_words = []
        for topic in lda.components_:
            arr = np.array(topic)
            highest_indexes = arr.argsort()[-10:][::-1]
            topics_words.append([vocabulary[i] for i in highest_indexes])
            # text_for_wordcloud = ' '.join([str(element) for element in topics_words ]) 
            # wordcloud = WordCloud().generate(text_for_wordcloud)
            # plt.imshow(wordcloud, interpolation='bilinear')
            # plt.axis("off")
            # plt.show()
        bow_models.append(lda)
        perplexity = lda.perplexity(posts_bow)
        bow_lda_perplexity.append(perplexity)
        print("Perplexity: ", perplexity)

    highest_index = np.argmin(bow_lda_perplexity)
    highest_bow_model = bow_models[highest_index]
    joblib.dump(highest_bow_model, 'lda/best_lda_model.pkl')
    joblib.dump(vectorizer, 'lda/bow_vectorizer.pkl')

    plt.figure(figsize=(8, 5))
    plt.plot(n_topics, bow_lda_perplexity, 'o-')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('Perplexity')
    plt.title('LDA: Perplexity vs Number of Topics')
    plt.tight_layout()
    plt.savefig('lda/perplexity_vs_k.png')
    plt.show()

# Main
if __name__ == "__main__":
    main()