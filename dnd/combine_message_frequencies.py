import argparse
import os
import re
import time

import nle.dataset as nld

from sf_examples.nethack.utils.nle_tokenizer.tokenizer import DND_TOKENIZER, DND_DETOKENIZER


def format_word(word):
    if "[" in word:
        return None
    word = word.replace("--more--", "")
    word = ''.join(char for char in word if char not in "!?.,\"'-|()[]*%:;/\\")
    return word


def format_message(message_str):
    # We format then tokenize and then detokenize the message, to only retain minimal words and maximise message collapse

    words = [format_word(word) for word in message_str.split(" ")]
    words = [word for word in words if word]

    tokens = [DND_TOKENIZER[word] for word in words]
    tokens = [token for token in tokens if token]
    tokens = list(sorted(tokens))

    detokenized_words = [DND_DETOKENIZER[token] for token in tokens]
    detokenized_message = " ".join(detokenized_words)

    return detokenized_message

def parse_dataset(args):
    message_counts = {}

    for dlvl in range(args.max_dlvl):
        for xplvl in range(args.max_xplvl):
            context_key = (args.character, dlvl, xplvl)

            message_counts[context_key] = {}

            for chunk_id in range(args.num_chunks):
                path = os.path.join(args.chunks_dir, f"{context_key[0]}/{context_key[0]}_dlvl{context_key[1]}_xplvl{context_key[2]}_{str(chunk_id)}.csv")

                if os.path.exists(path):
                    with open(path) as f:
                        chunk_counts = [l.split(",") for l in f.read().split("\n")]
                        chunk_counts = [(l[0], int(l[1])) for l in chunk_counts if len(l) == 2]

                        for (message, count) in chunk_counts:
                            if count <= 5:
                                break

                            if message not in message_counts[context_key]:
                                message_counts[context_key][message] = count
                            else:
                                message_counts[context_key][message] += count

    for context_key, context_counts in message_counts.items():
        sorted_items = sorted(context_counts.items(), key=lambda item: item[1], reverse=True)
        sorted_str = ""
        for word, count in sorted_items:
            sorted_str += word + "," + str(count) + "\n"

        path = os.path.join(args.output_dir,
                            f"{context_key[0]}/{context_key[0]}_dlvl{context_key[1]}_xplvl{context_key[2]}.csv")

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "w") as file:
            file.write(sorted_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks_dir', type=str, default='/checkpoint/michaelmatthews/nle_messages')
    parser.add_argument('--output_dir', type=str, default='/checkpoint/michaelmatthews/nle_messages/combined')
    parser.add_argument('--num_chunks', type=int, default=269)
    parser.add_argument('--character', type=str, default="Val-Dwa-Fem-Law")
    parser.add_argument('--max_dlvl', type=int, default=8)
    parser.add_argument('--max_xplvl', type=int, default=8)
    args = parser.parse_args()
    t0 = time.time()

    print("starting.")
    parse_dataset(args)

    print(time.time() - t0)
