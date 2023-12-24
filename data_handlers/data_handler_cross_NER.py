"""
--- Data Handler for Cross-NER dataset, for approaching NER task through a Multi-Span Extractive QA system---

Dataset downloadable from github repo at https://github.com/zliucr/CrossNER
Contains 1 general domain NER dataset CoNLL2003, and 5 domain specific datasets:
- politics, science, literature, music, ai
Each domain specific dataset has its own defined set of NE categories.
"""


# importing packages
import json
import os
import random
import re

from datasets import Dataset, DatasetDict
try:
    from data_handlers.data_handler_pileNER import has_more_than_n_foreign_chars, has_too_many_newline, has_too_many_whitespaces, has_too_many_punctuations_and_digits, split_into_sentences, count_target_words
except:
    from data_handler_pileNER import has_more_than_n_foreign_chars, has_too_many_newline, has_too_many_whitespaces, has_too_many_punctuations_and_digits, split_into_sentences, count_target_words

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

    # dataset_dict["dataset_name"] = dataset_name

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
def build_dataset_MSEQA_format(dataset_dict, path_to_questions_txt):
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
    # new_dataset_dict["dataset_name"] = dataset_dict["dataset_name"]

    return new_dataset_dict

def get_one_sentence_from_sample(ne_type, document_context, ne_occurrences):
    # split in sentences according to punctuation .?!
    sentences = split_into_sentences(document_context)
    target_words = [ne[0] for ne in ne_occurrences]
    # print(target_words)
    # count the occurrences of target words in each sentence
    # to return the one with at least 1/highest number of occ.
    target_word_counts = []
    for sentence in sentences:
        occurrences_count, target_words_found = count_target_words(sentence, target_words)
        target_word_counts.append({"sentence": sentence,
                                   "target_words_in_it": target_words_found,
                                   "occurrences_count": occurrences_count
                                   })

    # sort sentences by decreasing occurrences_count
    target_word_counts = sorted(target_word_counts, key=lambda x: x['occurrences_count'], reverse=True)
    # returning the sentence with highest number of occurrences, but with some contraints
    sentence_to_ret = None
    i = 0
    while i < len(target_word_counts):
        if target_word_counts[i]['occurrences_count'] != 0:
            if 50 < len(target_word_counts[i]['sentence']) < 200:
                if not has_too_many_whitespaces(target_word_counts[i]['sentence'], 4):
                    if not has_too_many_newline(target_word_counts[i]['sentence'], 1):
                        if not has_more_than_n_foreign_chars(target_word_counts[i]['sentence'], 2):
                            if not has_too_many_punctuations_and_digits(target_word_counts[i]['sentence'], 10):
                                sentence_to_ret = target_word_counts[i]
                                break
            elif ne_type in ['conference', 'country', 'location', 'person', 'university']:
                sentence_to_ret = target_word_counts[i]
                break
        i += 1

    return sentence_to_ret

def get_n_sentences_per_ne_type(dataset_BIO_format, ne_types_list, n_sentences_per_ne=3):
    """ getting from training set n_sentences_per_ne as positive examples from which to let gpt infer NE definition """
    # field real_name contains NE name in natural language format, e.g. BUYING_COMPANY --> BUYING COMPANY
    # medicalcondition --> medical condition, since not always automatically deducible we let it empty and then manually add it in the json
    sentences_per_ne_type = {ne: {'real_name': "", 'Hints': "", "sentences": []} for ne in ne_types_list}
    trainDataset = dataset_BIO_format['train'].to_list()
    random.seed(4)
    random.shuffle(trainDataset)
    for ne_type in ne_types_list:
        i = 0
        while len(sentences_per_ne_type[ne_type]['sentences']) < n_sentences_per_ne and i < len(trainDataset):
            sample = trainDataset[i]
            doc_ID, tokens, labels = sample.values()
            doc_metadata = get_doc_metadata_with_start_end_char_indexes(dataset_BIO_format, tokens, labels)
            if doc_metadata[ne_type]:
                occurrences_for_this_ne = doc_metadata[ne_type]
                document_context = ' '.join([str(elem) for elem in tokens])
                sentence_target_words = get_one_sentence_from_sample(ne_type, document_context, occurrences_for_this_ne)
                if sentence_target_words is not None:
                    # removing duplicates in list of target words
                    sentence_target_words['target_words_in_it'] = list(set(sentence_target_words['target_words_in_it']))
                    sentences_per_ne_type[ne_type]['sentences'].append(sentence_target_words)
            i += 1

    not_enough_sentences = []
    for ne_type, values in sentences_per_ne_type.items():
        if len(values['sentences']) < n_sentences_per_ne:
            # raise ValueError(f"not enough sentences for {ne_type}")
            not_enough_sentences.append((ne_type, len(values['sentences'])))
    print(f"NE types with less than n_sentences_per_ne: {len(not_enough_sentences)}")
    print(not_enough_sentences)

    return sentences_per_ne_type


def generate_structured_prompt(exemplary_data):
    ne_name = exemplary_data['real_name']
    sentences = exemplary_data['sentences']
    # unpacking sentences
    ex_sentences_json = []
    for exsent in sentences:
        ex_sentences_json.append({'sentence': exsent['sentence'], 'entities': exsent['target_words_in_it']})

    # TODO: hints from CrossNER paper annotation guidelines
    hints = exemplary_data['Hints']

    prompt = f"Named Entity: \'{ne_name}\'. Examples: {ex_sentences_json}."
    if hints != "":
        prompt += f" Hints: {hints}\n"
    else:
        prompt += "\n"
    prompt += f"Instructions: 1. Provide a concise definition for the named entity \'{ne_name}\' in the context of NER. 2. Provide guidelines by specifying what entities should not be labeled as \'{ne_name}\' and include potential pitfalls to avoid. Go beyond generic terms and delve into nuanced scenarios. Be explicit about potential ambiguities and provide guidance on distinguishing \'{ne_name}\' from similar entities.\n"
    prompt += "Output in JSON format: {\"Definition\": \"\", \"Guidelines\": \"\"}."
    return prompt


def build_dataset_MSEQA_format_with_guidelines(path_to_crosNER_datasets, subdataset_name, path_to_ne_definitions_json):
    # datasetDict in BIO format
    dataset_BIO_format = build_dataset_from_txt(os.path.join(path_to_crosNER_datasets, subdataset_name))
    # ne_types_list
    ne_types_list = get_ne_categories_only(dataset_BIO_format)
    # definitions for each ne
    with open(path_to_ne_definitions_json, 'r') as file:
        subdataset_NEs_guidelines = json.load(file)

    if isinstance(subdataset_NEs_guidelines, list):
        subdataset_NEs_guidelines = {x['named_entity']: x for x in subdataset_NEs_guidelines}

    newDataset_dict = {splitName: [] for splitName in dataset_BIO_format.keys() if splitName != "dataset_name"}
    newDataset_Dataset = {splitName: None for splitName in dataset_BIO_format.keys() if splitName != "dataset_name"}
    for splitName in dataset_BIO_format.keys():
        if splitName != "dataset_name":
            for i in range(len(dataset_BIO_format[splitName])):
                docID, tokens, labels = get_document_id_tokens_labels(dataset_BIO_format, splitName, i)
                docMetadata = get_doc_metadata_with_start_end_char_indexes(dataset_BIO_format, tokens, labels)
                question_number = 0
                for tagName in ne_types_list:
                    # question
                    gpt_definition = subdataset_NEs_guidelines[tagName]['gpt_answer'].strip()
                    # print(gpt_definition.strip())
                    # gpt answer may have been truncated, ensure it ends by "} before evaluating to dict
                    if not gpt_definition.endswith("}"):
                        if not gpt_definition.endswith("\""):
                            gpt_definition += "\""
                        gpt_definition += "}"
                    # print(gpt_definition)
                    this_ne_guidelines = eval(gpt_definition)
                    # replacing ne types occurrences between single quotes to their UPPERCASE
                    tagName_in_guidelines = subdataset_NEs_guidelines[tagName]['real_name']
                    pattern = re.compile(rf'\'{re.escape(tagName_in_guidelines)}\'')
                    this_ne_guidelines = {k: pattern.sub(f'{tagName_in_guidelines.upper()}', v) for k, v in this_ne_guidelines.items()}

                    question = f"Your task is to extract the Named Entities of type {tagName_in_guidelines.upper()} from an input TEXT. "
                    question += "You are given a DEFINITION and some GUIDELINES.\n"
                    question += "DEFINITION: " + this_ne_guidelines['Definition'] + "\nGUIDELINES: " + this_ne_guidelines['Guidelines'] + "\n"
                    question += f"TEXT: "

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
    # new_dataset_dict["dataset_name"] = dataset_dict["dataset_name"]

    return new_dataset_dict


if __name__ == '__main__':

    path_to_cross_NER_datasets = "../../../datasets/CrossNER/ner_data"
    splits = ["train", "validation", "test"]

    # sentences = read_bio_file(os.path.join(path_to_cross_NER_datasets, 'music', 'dev.txt'))
    # print(sentences)

    dataset_name = "ai"
    print(f"\nHandling data from dataset: {dataset_name}")
    dataset_BIO_format = build_dataset_from_txt(os.path.join(path_to_cross_NER_datasets, dataset_name))
    print("Dataset features: ")
    print(dataset_BIO_format["train"].features)

    #print("dataset_name")
    #print(music_dataset["dataset_name"])

    print("\nNE categories: ")
    ne_categories = get_ne_categories_labels_unique(dataset_BIO_format)
    print(len(ne_categories))
    print(ne_categories)

    print(f"\n{dataset_name} NE categories:")
    ne_types_list = get_ne_categories_only(dataset_BIO_format)
    print(ne_types_list)

    print("\nSplits statistics (number of occurrences per NE category): ")
    ne_categories_statistics = get_ne_categories_statistics(dataset_BIO_format)
    for split in splits:
        print(split)
        print(ne_categories_statistics[split])

    print("\nOne document example: ")
    doc_ID, tokens, labels = get_document_id_tokens_labels(dataset_BIO_format, "train", 25)
    print(doc_ID)
    print(tokens)
    print(labels)
    # converting BIO labeling to QA metedata
    print(get_doc_metadata_with_start_end_char_indexes(dataset_BIO_format, tokens, labels))

    #sentences_per_ne_type = get_n_sentences_per_ne_type(dataset_BIO_format, ne_types_list, n_sentences_per_ne=3)
    #print(sentences_per_ne_type)
    #with open(f"./questions/crossNER/sentences_per_ne_type_{dataset_name}.json", 'w') as f:
    #json.dump(sentences_per_ne_type, f, indent=2)

    # dataset_MSEQA_format_with_guidelines = build_dataset_MSEQA_format_with_guidelines(path_to_cross_NER_datasets, dataset_name, f"./questions/crossNER/{dataset_name}_NE_definitions.json")
    #print(dataset_MSEQA_format_with_guidelines)
    #print(dataset_MSEQA_format_with_guidelines['train'][0])
    #print(dataset_MSEQA_format_with_guidelines['train'][1])
    #print(dataset_MSEQA_format_with_guidelines['train'][23])

    path_to_questions = os.path.join("./questions/crossNER/what_describes_questions", dataset_name + ".txt")
    questions = load_questions_from_txt(path_to_questions)
    print(questions)

    print(sorted(questions.keys()) == get_ne_categories_only(dataset_BIO_format))

    dataset_QA_format = build_dataset_MSEQA_format(dataset_BIO_format, path_to_questions)
    print(dataset_QA_format)

    print(dataset_QA_format["train"][0])
    print(dataset_QA_format["train"][1])
    print(dataset_QA_format["train"][23])
