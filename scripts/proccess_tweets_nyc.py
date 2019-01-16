import more_itertools
import pandas as pd
import langdetect
import json
import re


{}

path = "./data/tweets_processados.tsv"
rows = ["idx", "nbr1", "nbr2", "position", "nbr3", "nbr4", "nbr5", "text"]
keywords = ["stomach", "throw up", "mylanta", "pepto bismol", "tummy", "threw up", "diarrhea", "poisoning", "nausea"]
# foodborne, vomit?
min_chars = 16
min_words = 3
verbose = True

all_rows = []

for chunk_idx, chunk in enumerate(pd.read_csv(path, sep="\t", chunksize=10**6,   names=rows)):

    original_len = len(chunk)

    # Treats text, removing errors and redundancy
    chunk.loc[:, "text"] = chunk.text.apply(lambda x: x[9:-1] if type(x) is str else "")

    # Filters tweets by keyword matching
    chunk.loc[:, "selected"] = chunk.text.apply(lambda x: match_all(x, keywords)).values
    chunk = chunk[chunk["selected"]]
    keywords_len = len(chunk)

    # Handles annoying emojis
    chunk.loc[:, "text"] = chunk.text.apply(lambda x:str(x.encode('ascii', 'xmlcharrefreplace')))

    # Handles language
    chunk = chunk[chunk.text.apply(lambda x: langdetect.detect(x) == "en")]
    language_len = len(chunk)

    # Handles tweets with very few words or very few characters
    chunk = chunk[chunk.text.apply(lambda x: len(x) > min_chars and len(x.split(" ")) > min_words)]
    length_len = len(chunk)

    # Creates .json file
    rows = [dict(row) for _, row in chunk.iterrows()]
    all_rows += rows

    # Prints the lengths of data frame along the filters
    if verbose:
        print("{0} ---> {1} ---> {2} ---> {3}".format(original_len, keywords_len, language_len, length_len))


df = pd.DataFrame(all_rows)

df = df.groupby("text").aggregate(lambda x: list(x)).reset_index()

df.loc[:, "selected"] = df["selected"].apply(lambda x: len(x))

rows = [dict(row) for _, row in df.iterrows()]

for num, tweets_list in enumerate(more_itertools.chunked(rows, 20)):
    with open("./data/questions/{0}.json".format(num), "w") as f:
        json.dump({"tweets": tweets_list}, f)

df.to_csv("data/tweets_processados_keywords.tsv")
