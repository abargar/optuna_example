import pandas as pd
import re
import string
from pathlib import Path
import json
from time import time

p = re.compile("[A-Za-z]+")
crap_words = set([])
with open("useless_words.txt", "r") as f:
    for line in f:
        crap_words.add(line.strip())


def pull_from_file(fname):
    f_recipes = []
    f = json.load(open(fname, "r"))
    for rid, recipe in f.items():
        if (recipe.get("title") is not None) and (
            recipe.get("ingredients") is not None
        ):
            ingredients = " ".join(recipe["ingredients"])
            f_recipes.append((rid, recipe["title"], ingredients))
    return f_recipes


def make_tokens(text):
    text = text.lower()
    only_words = [
        t.translate(str.maketrans("", "", string.punctuation))
        for t in text.split()
        if p.match(t)
    ]
    return [word for word in only_words if word not in crap_words]


# pull
files = [f for f in Path("data/raw").iterdir() if not f.name.endswith("zip")]
all_recipes = []
for f in files:
    f_recipes = pull_from_file(f)
    all_recipes.extend(f_recipes)
ingredients_df = pd.DataFrame(
    all_recipes, columns=["id", "title", "ingredients"]
).set_index("id")
ingredients_df.to_csv("all_ingredients.csv")

# tokenize
ingredients_df["tokens"] = ingredients_df["title"].str.cat(
    ingredients_df["ingredients"], " "
)
ingredients_df["tokens"] = ingredients_df["tokens"].map(make_tokens)
print(ingredients_df.head())
ingredients_df.to_parquet("data/tokenized_ingredients.parquet")
