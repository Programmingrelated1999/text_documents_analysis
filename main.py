# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

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

# Check Duplicate Records
def get_duplicate_records(df):
    return df[df.duplicated()]

# Get Average Post Length
def get_average_post_len(df):
    post_len_list = df.post.str.len()
    return post_len_list.sum()/len(df.post)

# Get CrossTab Between Ground Truth And Majority Votes
def get_majority_votes(df):
    return pd.crosstab(df['news_headline_ground_truth'], df['majority_votes'])

# CrossTab Graph
def plot_majority_votes_crosstab(majority_votes):
    majority_votes.plot.bar(rot=0)
    plt.title("Groupd Truth And Majority Votes")
    plt.savefig("ground_truth_vs_majority_votes_bar_graph.png")
    plt.show()
    plt.close()

# Pie Plot Graph
def plot_pie_graph(real_posts_count, fake_posts_count):
    labels = [f"Real Post Count: {real_posts_count}", f"Fake Post Count: {fake_posts_count}"]
    fig, ax = plt.subplots()
    ax.pie([real_posts_count, fake_posts_count], labels=labels, autopct='%1.2f%%')
    plt.title("Real And Fake Post Distribution")
    plt.savefig("real_and_fake_post_distribution_pie_plot.png")
    plt.show()
    plt.close()

# Main Function
def main():
    if __name__ == "__main__":
        # Read CSV And Call Functions To Get Descriptive Analysis Information
        post_df = pd.read_csv("social-media-release.csv")
        shape = explore_dataset_information(post_df)
        incomplete_columns = get_incomplete_columns(post_df)
        real_posts_count, fake_posts_count = get_num_real_and_fake_posts(post_df)
        real_fake_ratio = round(fake_posts_count / real_posts_count, 2)
        unique_headlines = get_unique_headlines(post_df)
        average_post_len = get_average_post_len(post_df)
        duplicate_posts = get_duplicate_records(post_df)
        majority_votes = get_majority_votes(post_df)

        # Print Output Text Of Descriptive Analysis
        print("DATASET ANALYSIS")
        print("===========================================")
        print("Number Of Records: ", shape[0])
        print("Number Of Duplicate Records: ", len(duplicate_posts))
        print("Incomplete Columns: ", incomplete_columns)
        print("Number Of Real Posts: ", real_posts_count)
        print("Number Of Fake Posts: ", fake_posts_count)
        print("Real To Fake Ratio: 1 :", real_fake_ratio)
        print(f"Share Of Read And Fake Posts: {(real_posts_count/shape[0])*100:.2f}%, {(fake_posts_count/shape[0])*100:.2f}%")
        print("Unique Headlines Count: ", len(unique_headlines))
        print(f"Average Post Length: {average_post_len:.2f}")
        print("Ground Truth And Majority Votes Breakdown")
        print(majority_votes)

        # Visualizations - Plot Pie Graph, 
        plot_pie_graph(real_posts_count, fake_posts_count)
        plot_majority_votes_crosstab(majority_votes)

# Call Main Function
main()