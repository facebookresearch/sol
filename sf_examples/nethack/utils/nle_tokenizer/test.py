import os

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tokens.txt'), "r") as f:
    tokens1 = f.read().split("\n")

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'full_tokens_3k.txt'), "r") as f:
    tokens2 = f.read().split("\n")


cnt1 = 0
for t in tokens1:
    if t not in tokens2:
        print("Removed:", t)
        cnt1 += 1

cnt2 = 0
for t in tokens2:
    if t not in tokens1:
        print("Added:", t)
        cnt2 += 1

print(cnt1, cnt2)