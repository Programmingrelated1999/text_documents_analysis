# Write To Feature Counts File
def write_feature_counts_file(vector, feature_names):
    print("Writing To features")
    counts = vector.sum(axis=0).A1
    with open("feature_counts.txt", "w", encoding="utf-8") as f:
        for word, count in zip(feature_names, counts):
            f.write(f"{word} {int(count)}\n")

