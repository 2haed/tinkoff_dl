import argparse
import ast
import io
import os
import pathlib
import pickle
from pathlib import Path
import tokenize
from typing import Any
import numpy as np
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import builtins
from pathlib import Path
from typing import Any

from tabulate import tabulate


class InfoGatherer(ast.NodeVisitor):
    def __init__(self, code, filename):

        self.filename = filename
        self.class_counter: int = 0
        self.func_counter: int = 0
        self.import_counter: int = code.count(' import ')
        self.operator_counter: int = code.count('+') + code.count('-') + code.count('=')
        self.condition_counter: int = code.count(' if ') + code.count(' else ') + code.count('\nif ') + code.count(
            '\nelse ')
        self.cycle_counter: int = code.count(' for ') + code.count(' \nfor ')
        self.nesting_degree: int = 0
        words = set(code.split())
        self.avg_word_length: float = sum([len(word) for word in words]) / len(words)
        self.avg_density: float = np.mean([len(row) for row in code.split("\n")]) / len(code.split("\n"))
        self.global_var_name_length: int = 0
        self.func_name_length: int = 0
        self.class_name_length: int = 0

    def visit_Assign(self, node: ast.Assign) -> Any:
        self.global_var_name_length += len(node.targets[0].id)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.class_counter += 1
        self.class_name_length += len(node.name)

        for sub_node in ast.iter_child_nodes(node):
            if isinstance(sub_node, ast.FunctionDef):
                self.visit_FunctionDef(sub_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.func_counter += 1
        self.func_name_length += len(node.name)

        nesting = max(self.max_nesting_counter(node))
        if nesting > self.nesting_degree:
            self.nesting_degree = nesting

    def max_nesting_counter(self, node) -> list[int]:
        children = [n for n in ast.iter_child_nodes(node) if isinstance(n, (ast.If, ast.For, ast.While))]
        if children:
            nesting = [1] * len(children)
            for index, sub_node in enumerate(children):
                nesting[index] += max(self.max_nesting_counter(sub_node))
            return nesting
        else:
            return [0]

    def make_raw_df(self) -> pd.DataFrame:
        df = pd.Series({
            'class_counter': self.class_counter,
            'func_counter': self.func_counter,
            'import_counter': self.import_counter,
            'operator_counter': self.operator_counter,
            'condition_counter': self.condition_counter,
            'cycle_counter': self.cycle_counter,
            'nesting_degree': self.nesting_degree,
            'avg_word_length': self.avg_word_length,
            'avg_density': self.avg_density,
            'global_var_name_length': self.global_var_name_length,
            'func_name_length': self.func_name_length,
            'class_name_length': self.class_name_length,

        }).to_frame(name=self.filename).T
        return df


class TextHandler:
    def __init__(self, file_path: Path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = remove_comments_and_docstrings(file.read())
            self.tree = ast.parse(self.text)
            self.code = ast.unparse(self.tree)
            self.code_info = InfoGatherer(self.code, file.name)

    def make_df(self) -> pd.DataFrame:
        self.code_info.visit(self.tree)
        df = self.code_info.make_raw_df()
        return df


def make_dataframes(origin_path: str, f_plagiat_path: str = None, s_plagiat_path: str = None) -> pd.DataFrame:
    frames = []
    origin_handler = TextHandler(Path(origin_path))
    origin_df = origin_handler.make_df()

    f_plagiat_handler = TextHandler(Path(f_plagiat_path))
    f_plagiat_df = f_plagiat_handler.make_df()

    s_plagiat_handler = TextHandler(Path(s_plagiat_path))
    s_plagiat_df = f_plagiat_handler.make_df()

    pair = [origin_handler.code, f_plagiat_handler.code, s_plagiat_handler.code]
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectors = vectorizer.fit_transform(pair)

    origin_df['cosine_similarity'] = 1
    f_plagiat_df['cosine_similarity'] = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])
    s_plagiat_df['cosine_similarity'] = cosine_similarity(tfidf_vectors[0], tfidf_vectors[2])

    origin_df['is_plagiat'] = 1
    f_plagiat_df['is_plagiat'] = 0
    s_plagiat_df['is_plagiat'] = 0

    frames.append(origin_df)
    frames.append(f_plagiat_df)
    frames.append(s_plagiat_df)

    return pd.concat(frames)


def remove_comments_and_docstrings(text: str) -> str:
    io_obj = io.StringIO(text)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out


class Model:

    def __init__(self, files_dir: str, plagiat1_dir: str, pickle_model, df, plagiat2_dir: str = None):
        self.final_df: pd.DataFrame = df
        self.files_dir: str = files_dir
        self.plagiat1_dir: str = plagiat1_dir
        self.plagiat2_dir: str | None = plagiat2_dir
        self.pickle_model: str = pickle_model
        self.X: pd.DataFrame = self.final_df.drop(['is_plagiat'], axis=1)
        self.y: pd.Series = self.final_df.is_plagiat

    def train_model(self):
        randomized_search_model = CatBoostClassifier(random_state=42, task_type='GPU', verbose=False)
        randomized_search_result = randomized_search_model.randomized_search(
            {'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
             'l2_leaf_reg': [3, 5, 10, 15],
             'n_estimators': [2, 10, 50, 100, 200],
             'max_depth': [3, 4, 5, 7, 10],
             }, self.X, self.y, verbose=False)
        model = CatBoostClassifier(**randomized_search_result['params'], task_type='GPU', verbose=False)
        model.fit(self.X, self.y)
        with open(self.pickle_model, 'wb') as file:
            pickle.dump(model, file)
        return model


def parse_console() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PlagiarismChecker')
    parser.add_argument("files", type=str, help="directory with original codes")
    parser.add_argument("plagiat1", type=str)
    parser.add_argument("plagiat2", type=str)
    parser.add_argument("--model", type=str, help='model filename')
    return parser.parse_args()


def main():
    args = parse_console()

    final_frames = []
    for origin_file, first_file, second_file in zip(os.scandir(args.files), os.scandir(args.plagiat1),
                                                    os.scandir(args.plagiat2)):
        try:
            final_frames.append(
                make_dataframes(origin_path=origin_file, f_plagiat_path=first_file, s_plagiat_path=second_file))
        except Exception:
            continue
    final_df = pd.concat(final_frames)
    train_model = Model(files_dir=args.files, plagiat1_dir=args.plagiat1, pickle_model=args.model, df=final_df,
                        plagiat2_dir=args.plagiat2)
    train_model.train_model()


if __name__ == '__main__':
    main()
