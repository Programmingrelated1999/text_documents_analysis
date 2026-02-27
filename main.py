# Import Library
import pandas as pd

# Return Number Of Records, And Columns
def explore_dataset_information(df):
    shape = df.shape
    return shape

# Returns Incomplete Columns As A List
def get_incomplete_columns(df):
    missing_counts = df.isna().sum()
    incomplete_columns = missing_counts[missing_counts > 0].index.tolist()
    return incomplete_columns

# Returns Number Of Real And Fake Posts
def get_num_real_and_fake_posts(df):
    real_posts_count = (df.class_label == True).sum()
    fake_posts_count = (df.class_label == False).sum()
    return real_posts_count, fake_posts_count

# Returns Unique News Headlines
def get_unique_headlines(df):
    return df.news_headline.unique()

# Text Cleaning Function
def clean_text(text_column):
    return

# Get Average Post Length
def get_average_post_len(df):
    post_len_list = df.post.str.len()
    return post_len_list.sum()/len(df.post)

# Main Function
def main():
    if __name__ == "__main__":
        post_df = pd.read_csv("social-media-release.csv")
        shape = explore_dataset_information(post_df)
        incomplete_columns = get_incomplete_columns(post_df)
        real_posts_count, fake_posts_count = get_num_real_and_fake_posts(post_df)
        real_fake_ratio = round(fake_posts_count / real_posts_count, 2)
        unique_headlines = get_unique_headlines(post_df)
        average_post_len = get_average_post_len(post_df)
        print("DATASET ANALYSIS")
        print("===========================================")
        print("Number Of Records: ", shape[0])
        print("Incomplete Columns: ", incomplete_columns)
        print("Real Posts: ", real_posts_count)
        print("Fake Posts: ", fake_posts_count)
        print("Real To Fake Ratio: 1 :", real_fake_ratio)
        print("Unique Headlines Count: ", len(unique_headlines))
        print(f"Average Post Length: {average_post_len:.2f}")

# Call Main Function
main()