import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Clean Data - Make Text Lowercase, Remove @Names, Replace Html Punctuations With Word, Remove Newline Escape String
def clean_data(df, column):
    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace(r'@\w+', '', regex=True)
    df[column] = df[column].str.replace(r'&amp;', 'and', regex=True)
    df[column] = df[column].str.replace(r'\n+', ' ', regex=True)
    return df

# Tokenize Columns Using Spacy - Ignore Punctuation, Spaces, Currency And Stop
def tokenize_column(df, column, nlp):
    tokens = []
    for i, text in enumerate(df[column]):
        print("Record: ", i)
        doc = nlp(text)
        tokens.append(
            [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_currency and not token.is_stop]
        )
    df['tokens'] = tokens
    return df

# Define Main Function
def main():
    post_df = pd.read_csv("social-media-release.csv")

    post_df['post'] = post_df['post'].fillna('')
    post_df = clean_data(post_df, 'post')
    nlp = spacy.load('en_core_web_sm')

    post_df = tokenize_column(post_df, 'post', nlp)
    post_df['tokens_joined'] = post_df['tokens'].apply(' '.join)
    post_df.to_csv('social-media-release-tokenized.csv', index=False)

if __name__ == "__main__":
    main()