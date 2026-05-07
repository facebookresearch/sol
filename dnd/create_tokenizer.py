import argparse
import csv
import os
import time
from collections import defaultdict


def create_tokenizer(args):
    word_counts = defaultdict(int)

    for chunk_id in range(1, args.num_chunks+1):
        path = os.path.join(args.data_dir, f"counts_chunk_{chunk_id}.csv")

        with open(path, mode='r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                word = row[0]
                count = int(row[1])

                if count < args.min_count:
                    print("Truncating file at", word, count)
                    break

                word_counts[word] += count

    tokens = []
    sorted_items = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    for i, (word, count) in enumerate(sorted_items):
        if i < args.num_tokens:
            # print("Adding", word, "at", count)
            tokens.append(word)
        elif i < args.num_tokens + 100:
            print("Not adding", word, "at", count)
        else:
            break

    tokens_path = os.path.join(args.data_dir, "tokens.txt")
    with open(tokens_path, mode='w') as file:
        for t in tokens:
            file.write(f"{t}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/checkpoint/michaelmatthews/nle_tokenizer_2h')
    parser.add_argument('--num_chunks', type=int, default=30)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--num_tokens', type=int, default=3000)
    args = parser.parse_args()

    create_tokenizer(args)