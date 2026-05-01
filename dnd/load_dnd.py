import nle.dataset as nld

if not nld.db.exists():
    nld.db.create()
    # NB: Different methods are used for data based on NLE and data from NAO.
    # nld.add_nledata_directory("/private/home/michaelmatthews/data/nld-aa-taster/nle_data/", "nld-aa-taster-v0")
    nld.add_altorg_directory("/checkpoint/michaelmatthews/nld-nao/nao", "nld-nao-a-v0")

word_counts = {}

dataset = nld.TtyrecDataset("nld-nao-a-v0", batch_size=32)
for i, mb in enumerate(dataset):
    toplines = mb["tty_chars"][:, :, 0, :]
    for j in range(toplines.shape[0]):
        for k in range(toplines.shape[1]):
            message = "".join([chr(x) for x in toplines[j, k]])
            words = message.lower().split(" ")
            for word in words:
                word = word.replace("--more--", "")
                word = word.replace("!", "")
                word = word.replace(".", "")
                word = word.replace(",", "")
                word = word.replace('"', "")

                if word:
                    if word in word_counts:
                        word_counts[word] = word_counts[word] + 1
                    else:
                        word_counts[word] = 1
    print(i)

    if i >= 100:
        break

sorted_items = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
sorted_str = ""
for word, count in sorted_items:
    sorted_str += word + "," + str(count) + "\n"

with open('output.txt', 'w') as file:
    file.write(sorted_str)