from Dataset import pretest_2, risk_ratings_2

# create a DataFrame from the pretest_2 csv file
df = pretest_2

# initialize results lists
present_words = []
word_counts = {}

# loop through all words in DataFrame
for word in df.values.flatten():
    if word in present_words:
        word_counts[word] += 1
    else:
        present_words.append(word)
        word_counts[word] = 1

# print results
print("Words present in the DataFrame:")
for word in present_words:
    print(word)
print("\nWord counts:")
for word, count in word_counts.items():
    print(f"{word}: {count}")

risk2 = risk_ratings_2
names2 = risk_ratings_2.iloc[0, ]
total = []

for name in names2:
    if name in present_words:
        total.append(name)
    if name not in present_words:
        print(f"{name} is not in present_words df")

print(len(total))
