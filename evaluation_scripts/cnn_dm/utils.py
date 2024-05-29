import os
import ujson
import random
import glob
from pathlib import Path
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def read_txt(filename):

    assert os.path.exists(filename)

    with open(filename, 'r') as reader:
        text = []
        abstract = []
        highlight_flag = False

        for line in reader:
            line = line.strip()
            if line == '@highlight':
                highlight_flag = True
                continue

            if highlight_flag:
                abstract.append(line)
            else:
                text.append(line)

        return ' '.join(text), ' '.join(abstract)


def read_json(filename):

    with open(filename, 'r') as reader:
        for row in reader:
            data = ujson.loads(row.strip())
            yield data


def get_file_name(filename):
    return str(Path(filename).stem)


def get_examples(num_examples=10):
    article_abstract_pairs = []
    list_of_files = [f for f in glob.glob(os.path.join('/Users/giovanni/Desktop/r3s_cnn_dm/train', '**/*.txt'), recursive=True)]
    selected_files = random.choices(list_of_files, k=num_examples)
    for filename in selected_files:
        article, abstract = read_txt(filename)
        article = ' '.join(article.split(' ')[:250])
        abstract = ' '.join(abstract.split(' ')[:100])
        article_abstract_pairs.append((article, abstract))

    return article_abstract_pairs


def compute_rouge(target, prediction):
    scores = scorer.score(target, prediction)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure
