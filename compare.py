import argparse
import os
import pickle
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from train import TextHandler
from catboost import CatBoostClassifier


def count_similarity(path_pair: list, model: CatBoostClassifier):
    frames = []
    origin_handler = TextHandler(Path(path_pair[0]))
    origin_df = origin_handler.make_df()

    plagiat_handler = TextHandler(Path(path_pair[1]))
    plagiat_df = plagiat_handler.make_df()

    pair = [origin_handler.code, plagiat_handler.code]
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectors = vectorizer.fit_transform(pair)

    origin_df['cosine_similarity'] = 1
    plagiat_df['cosine_similarity'] = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])

    origin_df['is_plagiat'] = 1
    plagiat_df['is_plagiat'] = 0

    frames.append(origin_df)
    frames.append(plagiat_df)

    final_df = pd.concat(frames)
    X = final_df.drop(['is_plagiat'], axis=1)
    return model.predict_proba(X)[:, 1][0]


def get_args() -> list[Any]:
    parser = argparse.ArgumentParser(description='PlagiarismChecker')
    parser.add_argument("input.txt", type=str, help="Input file")
    parser.add_argument("scores.txt", type=str, help='Output file')
    parser.add_argument("--model", type=str)
    return [item for item in vars(parser.parse_args()).values()]


def main():
    args = get_args()
    with open(args[2], 'rb') as file:
        model = pickle.load(file)
    with open(args[0], 'r', encoding="utf-8") as input_file, open(args[1], 'w', encoding="utf-8") as output_file:
        data = [pair.strip().split() for pair in input_file.readlines()]
        for file_path in data:
            output_file.write(f'{str(round(count_similarity(file_path, model=model), 2))}\n')


if __name__ == '__main__':
    main()
