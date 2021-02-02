import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim import corpora

model = LdaModel.load("ingredient_models/trial_61/61_lda")
corpus = corpora.MmCorpus("data/token_corpus.mm")
recipe_tokens = pd.read_parquet("data/tokenized_ingredients.parquet")[['title', 'tokens']]

recipe_topic_distributions = []
for bow, recipe in zip(corpus, recipe_tokens.itertuples()):
    record = {topic: prob for topic, prob in model.get_document_topics(bow)}
    record['recipe'] = recipe.title
    recipe_topic_distributions.append(record)

recipe_topic_df = pd.DataFrame.from_records(recipe_topic_distributions)
recipe_topic_df = recipe_topic_df.fillna(0)
recipe_topic_df = recipe_topic_df[["recipe", *range(10)]]
print(recipe_topic_df)
recipe_topic_df.to_csv("recipe_topic_df.csv")

with open("top_recipes.txt", 'w') as out:
    for topic in range(10):
        print(topic, file=out)
        print(df[["recipe", str(topic)]].sort_values(str(topic), ascending=False).head(), file=out)
        print('',file=out)