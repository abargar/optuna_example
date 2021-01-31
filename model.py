import gensim
from gensim.models import CoherenceModel
from gensim import corpora
import optuna
import pandas as pd
import logging
import csv
from pathlib import Path


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("optuna.log", mode="w"))

token_df = pd.read_parquet("data/tokenized_ingredients.parquet")
tokens_list = token_df.tokens.values
corpora_dict = corpora.Dictionary(tokens_list)
corpus = [corpora_dict.doc2bow(tokens) for tokens in tokens_list]

with open("model_results.csv", "w") as f:
    csvwriter = csv.DictWriter(
        f, fieldnames=["trialno", "coherence", "ntopics", "alpha", "eta"]
    )
    csvwriter.writeheader()


def compute_coherence(model, corpus, corpora_dict):
    coherence_model_lda = CoherenceModel(
        model=model,
        texts=corpus,
        corpus=None,
        dictionary=corpora_dict,
        coherence="c_v",
    )
    return coherence_model_lda.get_coherence()


def get_and_save_top_words(model, out_file):
    top_words_per_topic = []
    for t in range(model.num_topics):
        top_words_per_topic.extend([(t,) + x for x in model.show_topic(t, topn=50)])
    pd.DataFrame(top_words_per_topic, columns=["topic", "word", "p"]).to_parquet(
        path=out_file, index=False
    )


def objective(trial):
    alpha = trial.suggest_uniform("alpha", 0.01, 1)
    eta = trial.suggest_uniform("eta", 0.01, 1)
    num_topics = trial.suggest_uniform("num_topics", 10, 50)
    ideal_score = 0.8
    model = gensim.models.LdaMulticore(
        workers=7,
        corpus=corpus,
        id2word=corpora_dict,
        num_topics=num_topics,
        random_state=100,
        passes=3,
        alpha=alpha,
        eta=eta,
        per_word_topics=True,
    )
    coherence_score = compute_coherence(model, tokens_list, corpora_dict)
    print(f"Trial {trial.number} coherence score: {round(coherence_score,3)}")
    with open("model_results.csv", "a") as f:
        csvwriter = csv.DictWriter(
            f, fieldnames=["trialno", "coherence", "ntopics", "alpha", "eta"]
        )
        csvwriter.writerow(
            {
                "trialno": trial.number,
                "coherence": coherence_score,
                "ntopics": num_topics,
                "alpha": alpha,
                "eta": eta,
            }
        )
    if coherence_score >= 0.5:
        model_path = Path(f"models/trial_{trial.number}")
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path / f"{trial.number}_lda"))
        top_words_filename = model_path / f"trialno{trial.number}_top_words.parquet"
        get_and_save_top_words(model, top_words_filename)
    coherence_score_diff = abs(ideal_score - coherence_score)
    return coherence_score_diff


study = optuna.create_study()
study.optimize(objective, n_trials=100)
Path(f"models").mkdir(exist_ok=True)

logger.info(f"Best trial: {study.best_trial.number}")
logger.info(f"Best trial info: {study.best_trial}")
