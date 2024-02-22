"""
--- Data Handler for Cross-NER dataset, for approaching NER task through a Multi-Span Extractive QA system---

Dataset downloadable from github repo at https://github.com/zliucr/CrossNER
Contains 1 general domain NER dataset CoNLL2003, and 5 domain specific datasets:
- politics, science, literature, music, ai
Each domain specific dataset has its own defined set of NE categories.
"""

# importing packages
import os
from datasets import Dataset, DatasetDict


# read sentences with BIO labelling from txt file
def read_bio_file(path_to_bio_txt):

    with open(path_to_bio_txt, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    progressive_ID = 0
    sentences = []
    tokens = []
    labels = []

    for line in lines:
        line = line.strip()
        if not line:
            if tokens and labels:
                sentences.append({'id': progressive_ID, 'tokens': tokens, 'labels': labels})
                tokens = []
                labels = []
                progressive_ID += 1
        else:
            token, label = line.split()
            tokens.append(token)
            labels.append(label)

    return sentences


# build dataset from txt files with train, validation and test partitions
def build_dataset_from_txt(path_to_dataset):

    dataset_name = path_to_dataset.split("/")[-1]

    train_data = read_bio_file(os.path.join(path_to_dataset, 'train.txt'))
    validation_data = read_bio_file(os.path.join(path_to_dataset, 'dev.txt'))
    test_data = read_bio_file(os.path.join(path_to_dataset, 'test.txt'))

    train_dataset = Dataset.from_dict({
        "doc_id": [sample["id"] for sample in train_data],
        "tokens": [sample["tokens"] for sample in train_data],
        "labels": [sample["labels"] for sample in train_data]
    })
    validation_dataset = Dataset.from_dict({
        "doc_id": [sample["id"] for sample in validation_data],
        "tokens": [sample["tokens"] for sample in validation_data],
        "labels": [sample["labels"] for sample in validation_data]
    })
    test_dataset = Dataset.from_dict({
        "doc_id": [sample["id"] for sample in test_data],
        "tokens": [sample["tokens"] for sample in test_data],
        "labels": [sample["labels"] for sample in test_data]
    })

    dataset_dict = DatasetDict({'train': train_dataset,
                                'validation': validation_dataset,
                                'test': test_dataset
    })

    dataset_dict["dataset_name"] = dataset_name

    return dataset_dict


# get list of used BIO labels (unique)
def get_ne_categories_labels_unique(dataset_dict):
    ne_categories = {}
    for split in dataset_dict.keys():
        if split != 'dataset_name':
            for document in dataset_dict[split]:
                doc_labels = document["labels"]
                for lbl in doc_labels:
                    if lbl not in ne_categories:
                        ne_categories[lbl] = 0

    ne_categories_sorted = dict(sorted(ne_categories.items())).keys()
    return ne_categories_sorted


# get list of NE categories
def get_ne_categories_only(dataset_dict):
    ne_cat_lbls_unique = get_ne_categories_labels_unique(dataset_dict)
    return [lbl[2:] for lbl in ne_cat_lbls_unique if lbl[0] == 'B']


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
                            ne_categories_statistic[split][lbl] = 0
                        else:
                            ne_categories_statistic[split][lbl] += 1

    ne_categories_statistic = {split: dict(sorted(ne_categories_statistic[split].items())) for split in ne_categories_statistic.keys()}
    return ne_categories_statistic


# given split and index return doc_id, tokens and labels
def get_document_id_tokens_labels(dataset_dict, split, i):
    # adding split identifier to doc_id
    doc_id = split + ':' + str(dataset_dict[split][i]["doc_id"])
    return doc_id, dataset_dict[split][i]["tokens"], dataset_dict[split][i]["labels"]


# from BIO labeling to metadata (for each NE category a list of occurrences (text_span, character_start, character_end))
def get_doc_metadata_with_start_end_char_indexes(dataset_dict, doc_tokens, doc_labels):
    ne_categories = get_ne_categories_only(dataset_dict)
    doc_metadata = {ne: [] for ne in ne_categories}
    i = 0
    index = 0
    startIndex = index
    entity = ''  # entity being reconstructed
    while i < len(doc_labels):
        # if the token is labelled as part of an entity
        if doc_labels[i] != 'O':
            if entity == '':
                startIndex = index
            entity = entity + ' ' + doc_tokens[i]  # this will add an initial space (to be removed)
            # if next label is Other or the beginning of another entity
            # or end of document, the current entity is complete
            if (i < len(doc_labels) - 1 and doc_labels[i + 1][0] in ["O", "B"]) or (i == len(doc_labels) - 1):
                # add to metadata
                tagName = doc_labels[i].split("-")[-1]
                # adding also if same name but will have != start-end indices
                doc_metadata[tagName].append((entity[1:], startIndex, startIndex + len(entity[1:])))
                # cleaning for next entity
                entity = ''

        index = index + len(doc_tokens[i]) + 1
        i += 1

    return doc_metadata


# reading questions associated to each NE category, specific to the current dataset
def load_questions_from_txt(path_to_txt):
    questions = {}
    with open(path_to_txt, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        if line != '\n':
            tagName, associated_question = line.strip().split(":")
            questions[tagName] = associated_question
    return questions


# build DatasetDict object with docContext-question-goldAnswers
def build_dataset_QA_format(dataset_dict, path_to_questions_txt):
    # loading questions associated to each NE category
    questions = load_questions_from_txt(path_to_questions_txt)
    print(questions)

    newDataset_dict = {splitName: [] for splitName in dataset_dict.keys() if splitName != "dataset_name"}
    newDataset_Dataset = {splitName: None for splitName in dataset_dict.keys() if splitName != "dataset_name"}
    for splitName in dataset_dict.keys():
        if splitName != "dataset_name":
            for i in range(len(dataset_dict[splitName])):
                docID, tokens, labels = get_document_id_tokens_labels(dataset_dict, splitName, i)
                docMetadata = get_doc_metadata_with_start_end_char_indexes(dataset_dict, tokens, labels)
                question_number = 0
                for tagName in questions.keys():
                    question = questions[tagName]
                    # splitName:docID:questioNumberForThatDocument
                    doc_question_pairID = docID + ':' + str(question_number)
                    question_number += 1
                    # document context
                    context = ' '.join([str(elem) for elem in tokens])
                    # retrieving gold answers for this tagName
                    goldAnswers = docMetadata[tagName]
                    answers = {'answer_start': [], 'text': []}
                    for ga in goldAnswers:
                        answers['answer_start'].append(ga[1])
                        answers['text'].append(ga[0])
                    sample = {'doc_question_pairID': doc_question_pairID,
                              'document_context': context,
                              'tagName': tagName,
                              'question': question,
                              'answers': answers
                              }
                    newDataset_dict[splitName].append(sample)
            newDataset_Dataset[splitName] = Dataset.from_list(newDataset_dict[splitName])

    new_dataset_dict = DatasetDict(newDataset_Dataset)
    new_dataset_dict["dataset_name"] = dataset_dict["dataset_name"]

    return new_dataset_dict


if __name__ == '__main__':

    path_to_cross_NER_datasets = "../../datasets/CrossNER/ner_data"
    splits = ["train", "validation", "test"]

    # sentences = read_bio_file(os.path.join(path_to_cross_NER_datasets, 'music', 'dev.txt'))
    # print(sentences)

    dataset_name = "music"
    print(f"\nHandling data from dataset: {dataset_name}")
    music_dataset = build_dataset_from_txt(os.path.join(path_to_cross_NER_datasets, dataset_name))
    print("Dataset features: ")
    print(music_dataset["train"].features)

    print("dataset_name")
    print(music_dataset["dataset_name"])

    print("\nNE categories: ")
    ne_categories_music = get_ne_categories_labels_unique(music_dataset)
    print(len(ne_categories_music))
    print(ne_categories_music)

    print(get_ne_categories_only(music_dataset))

    print("\nSplits statistics (number of occurrences per NE category): ")
    ne_categories_statistics = get_ne_categories_statistics(music_dataset)
    for split in splits:
        print(split)
        print(ne_categories_statistics[split])

    print("\nOne document example: ")
    doc_ID, tokens, labels = get_document_id_tokens_labels(music_dataset, "train", 25)
    print(doc_ID)
    print(tokens)
    print(labels)
    # converting BIO labeling to QA metedata
    print(get_doc_metadata_with_start_end_char_indexes(music_dataset, tokens, labels))

    path_to_questions = os.path.join("./cross_ner_questions/", dataset_name + ".txt")
    questions = load_questions_from_txt(path_to_questions)
    print(questions)

    print(sorted(questions.keys()) == get_ne_categories_only(music_dataset))

    dataset_QA_format = build_dataset_QA_format(music_dataset, path_to_questions)
    print(dataset_QA_format)

    print(dataset_QA_format["train"][0])
    print(dataset_QA_format["train"][1])
    print(dataset_QA_format["train"][23])
