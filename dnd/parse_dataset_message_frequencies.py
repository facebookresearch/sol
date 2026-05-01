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
    if not nld.db.exists():
        nld.db.create()
        nld.add_altorg_directory("/checkpoint/michaelmatthews/nld-nao/nao/nld-nao-unzipped", "nld-nao-v0")

    message_counts = {}

    dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=args.batch_size)

    for i, mb in enumerate(dataset):
        toplines = mb["tty_chars"][:, :, 0, :]
        bottomlines = mb["tty_chars"][:, :, -2:, :]

        for j in range(toplines.shape[0]):
            # Early exit for efficiency (we will miss a few game starts however)
            if args.character and mb["game_data"][j, 0] != args.character:
                continue

            for k in range(toplines.shape[1]):
                character = mb["game_data"][j, k]

                if args.character and character != args.character:
                    continue

                message = "".join([chr(x) for x in toplines[j, k]]).rstrip(" ")
                messages = re.split(r'[.!]', message)
                messages = [msg for msg in messages if msg]

                blstatsl2 = "".join([chr(x) for x in bottomlines[j, k, 1]]).rstrip(" ")

                dlvl = None
                xplvl = None

                for stat in blstatsl2.split(" "):
                    stat_comps = stat.split(":")
                    if len(stat_comps) >= 2:
                        if stat_comps[0] == "Dlvl":
                            dlvl = stat_comps[1]
                        elif stat_comps[0] == "Xp":
                            try:
                                # showexp:true
                                if "/" in stat_comps[1]:
                                    xplvl = int(stat_comps[1].split("/")[0])
                                # showexp:false
                                else:
                                    xplvl = int(stat_comps[1])
                            except ValueError:
                                pass

                if dlvl is not None and xplvl is not None:
                    dlvl_context = str(int(dlvl) // 3) if dlvl.isdigit() else dlvl
                    xplvl_context = str(xplvl // 2)
                    context_key = (character, dlvl_context, xplvl_context)

                    if context_key not in message_counts:
                        message_counts[context_key] = {}

                    for message in messages:
                        message_fmt = format_message(message)
                        if message_fmt:
                            if message_fmt in message_counts[context_key]:
                                message_counts[context_key][message_fmt] += 1
                            else:
                                message_counts[context_key][message_fmt] = 1

        if i % args.chunk_size == 0 and i > 0:
            chunk_id = i // args.chunk_size
            for context_key, c_message_counts in message_counts.items():
                sorted_items = sorted(c_message_counts.items(), key=lambda item: item[1], reverse=True)
                sorted_str = ""
                for word, count in sorted_items:
                    sorted_str += word + "," + str(count) + "\n"

                path = os.path.join(args.output_dir, f"{context_key[0]}/{context_key[0]}_dlvl{context_key[1]}_xplvl{context_key[2]}_{str(chunk_id)}.csv")

                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))

                with open(path, "w") as file:
                    file.write(sorted_str)

            print("Chunk", chunk_id, "written to")
            message_counts = {}

        if i >= args.chunk_size * args.num_chunks:
            print("All chunks completed, exiting.")
            break




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/checkpoint/michaelmatthews/nle_messages')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--num_chunks', type=int, default=300)
    parser.add_argument('--character', type=str, default="Val-Dwa-Fem-Law")
    args = parser.parse_args()
    t0 = time.time()

    print("starting.")
    parse_dataset(args)

    print(time.time() - t0)
