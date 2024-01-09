import json
import os
import random
import re

from datasets import Dataset, DatasetDict, load_dataset


def read_BIO_file(path_to_bio_txt):
    with open(path_to_bio_txt, 'r') as file:
        bio_text = file.read()

    lines = bio_text.strip().split('\n')
    data = {'docID': [], 'text': [], 'labels': []}
    current_text = []
    current_labels = []
    docID = 1

    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            label, word = parts
            current_text.append(word)
            current_labels.append(label)
        else:
            data['text'].append(" ".join(current_text))
            data['labels'].append(current_labels)
            data['docID'].append(docID)
            current_text = []
            current_labels = []
            docID += 1

    return Dataset.from_dict(data)

def build_dataset_BIO_format_from_txt(mit_dataset_name):

    # download MIT datasets from https://groups.csail.mit.edu/sls/downloads/
    file_path = f'../../../datasets/MIT/{mit_dataset_name}'
    test_data = read_BIO_file(os.path.join(file_path, 'test.txt'))
    train_data = read_BIO_file(os.path.join(file_path, 'train.txt'))

    return DatasetDict({'train': train_data, 'validation': test_data, 'test': test_data})


# get statistics (number of occurrences per NE category) in train, validation and test folds
def get_ne_categories_statistics(dataset_dict):
    ne_categories_statistic = {}
    for split in dataset_dict.keys():
        if split != 'dataset_name':
            ne_categories_statistic[split] = {}
            for document in dataset_dict[split]:
                doc_labels = document["labels"]
                for lbl in doc_labels:
                    # counting number of occurrences per NE category (i.e. how many B- starting)
                    if lbl[0] == 'B':
                        if lbl not in ne_categories_statistic[split]:
                            ne_categories_statistic[split][lbl] = 1
                        else:
                            ne_categories_statistic[split][lbl] += 1

    ne_categories_statistic = {split: dict(sorted(ne_categories_statistic[split].items())) for split in ne_categories_statistic.keys()}
    return ne_categories_statistic

def build_dataset_MSEQA_format(mit_dataset_name):
    pass


if __name__ == '__main__':

    movie_dataset_from_txt = build_dataset_BIO_format_from_txt('movie')
    print(movie_dataset_from_txt)

    print(movie_dataset_from_txt['test'][1])
    ne_categories_statistics_from_txt = get_ne_categories_statistics(movie_dataset_from_txt)
    print(ne_categories_statistics_from_txt)
