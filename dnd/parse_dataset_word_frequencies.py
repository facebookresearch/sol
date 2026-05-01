import argparse
import os
import time

import nle.dataset as nld



def parse_dataset(args):
    if not nld.db.exists():
        nld.db.create()
        nld.add_altorg_directory("/checkpoint/michaelmatthews/nld-nao/nao/nld-nao-unzipped", "nld-nao-v0")

    word_counts = {}

    dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=args.batch_size)

    punctuation = "!?.,\"'-|()[]*%:;/\\"

    for i, mb in enumerate(dataset):
        toplines = mb["tty_chars"][:, :, 0, :]
        for j in range(toplines.shape[0]):
            for k in range(toplines.shape[1]):
                message = "".join([chr(x) for x in toplines[j, k]])
                words = message.lower().split(" ")
                for word in words:
                    if "[" in word:
                        break
                    word = word.replace("--more--", "")
                    # word = word.replace("!", "")
                    # word = word.replace(".", "")
                    # word = word.replace(",", "")
                    # word = word.replace("'", "")
                    # word = word.replace('"', "")
                    # word = word.replace('?', "")
                    word = ''.join(char for char in word if char not in punctuation)
                    # word = word.rstrip('?!.-')

                    if word:
                        if word in word_counts:
                            word_counts[word] = word_counts[word] + 1
                        else:
                            word_counts[word] = 1
        if i % args.chunk_size == 0 and i > 0:
            sorted_items = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
            sorted_str = ""
            for word, count in sorted_items:
                sorted_str += word + "," + str(count) + "\n"

            chunk_id = i // args.chunk_size
            path = os.path.join(args.output_dir, f"counts_chunk_{chunk_id}.csv")

            with open(path, 'w') as file:
                file.write(sorted_str)

            print("Chunk", chunk_id, "written to", path)
            word_counts = {}

        if i >= args.chunk_size * args.num_chunks:
            print("All chunks completed, exiting.")
            break




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/checkpoint/michaelmatthews/nle_tokenizer')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--num_chunks', type=int, default=3050)
    args = parser.parse_args()
    t0 = time.time()

    parse_dataset(args)

    print(time.time() - t0)
