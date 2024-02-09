"""
MSEQA_4_NER data_handler for Pile-NER dataset

Converts the dataset in MSEQA format with or w/o definition

- build_dataset_MSEQA_format() + remove_bad_ne_types() --> MSEQA dataset with question "What describes X in the text?"
- build_dataset_MSEQA_format_with_guidelines() --> MSEQA dataset with instruction+definition+guidelines as prefix to the context

- https://huggingface.co/datasets/Universal-NER/Pile-NER-type
- https://universal-ner.github.io/

Documents from PILE corpus annotated by GPT

NB: MSEQA dataset only considers 455-top frequent NEs,
and after some further deletions and NE mergings (lower casing + dict_of_merges)
--> we obtain a total of 423 different NEs
"""


import re
import ast
import json
import math
import random
import string
import numpy as np
from collections import OrderedDict
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets


# given a single UniversalNER conversation sample extract context (text passage) + (questions-gold_answers) list
def extract_context_quests_answers(conversation):
    # first element in the conversation list is the passage of text (context) provided by the human
    context = conversation.pop(0)
    if context["from"] == "human" and context["value"][:5] == "Text:":
        context = context["value"][len("Text: "):]
    else:
        raise ValueError("Invalid context or source in the conversation")

    # gpt confirms { "from": "gpt", "value": "I've read this text." }
    conversation.pop(0)

    # extracting list of questions for the context and to each q its associated list of answers
    quests_answers = []
    # reading 2 by 2
    for i in range(0, len(conversation), 2):
        if conversation[i]["from"] == "human" and conversation[i + 1]["from"] == "gpt":
            # extracting question from human
            question = conversation[i]["value"]
            # NE type being extracted
            start_char_ne_type = len("What describes ")
            end_char_ne_type = question.find("in the text?") - 1
            # extracting NE type and lowering so PERSON/Person/person are same
            ne_type = question[start_char_ne_type:end_char_ne_type].lower()
            # lower casing NE e.g. Person, PERSON to be both person
            question = "What describes " + ne_type + " in the text?"
            # other possible questions
            # question = "What refers to " + ne_type + " in the text?"

            # extracting answers from gpt
            ga_answers = {'answer_start': [], 'text': []}
            answers = ast.literal_eval(conversation[i + 1]["value"])
            for ans in answers:
                # finding target_word occurrences in the context
                # by returning a list of all starting positions in character
                def find_start_positions(text, target_word, find_subwords=False):
                    start_positions = []
                    # Find occurrences of the target_word
                    pattern = re.compile(r'\b' + re.escape(target_word) + r'\b')  # word bound delimiter
                    matches = pattern.finditer(text)
                    for match in matches:
                        start_positions.append({"answer_start": match.start(), "text": target_word})
                    # If find_subwords is True, find occurrences of subwords
                    if find_subwords:
                        sub_target_words = target_word.split()
                        for sub_word in sub_target_words:
                            # TODO: do not only for PERSON, only if > 3 chars, and if not in list of stopwords ["di", ...]
                            pattern = re.compile(r'\b' + re.escape(sub_word) + r'\b')
                            matches = pattern.finditer(text)
                            for match in matches:
                                start_position = match.start()
                                # Check if the subword appears as a standalone word in the context
                                # and not part of a target_word already in start_positions
                                is_standalone_subword = (
                                        re.search(r'\b' + re.escape(sub_word) + r'\b', text) is not None
                                        and not any(
                                    start_position >= occurrence["answer_start"] and
                                    start_position + len(sub_word) <= occurrence["answer_start"] + len(
                                        occurrence["text"]) and
                                    sub_word in occurrence["text"].split()
                                    for occurrence in start_positions
                                )
                                )

                                if is_standalone_subword:
                                    start_positions.append({"answer_start": start_position, "text": sub_word})

                    return start_positions

                # TODO: for now sub-word disabled
                if ne_type.lower() == "person":
                    start_positions = find_start_positions(context, ans, find_subwords=False)
                else:
                    start_positions = find_start_positions(context, ans, find_subwords=False)

                for sp in start_positions:
                    ga_answers['text'].append(sp["text"])
                    ga_answers['answer_start'].append(sp["answer_start"])

                if len(ga_answers['text']) != len(ga_answers['answer_start']):
                    raise ValueError("number of answer text not matching number of answer_start")

            quests_answers.append({"question": question, "ne_type": ne_type, "answers": ga_answers})

        else:
            raise ValueError("human-gpt non matched conversation")

    return {"context": context, "questions_answers": quests_answers}


def get_dataset_statistics():
    raw_dataset = load_dataset("Universal-NER/Pile-NER-type")

    context_lengths = []
    for raw_sample in raw_dataset['train']['conversations']:
        # extract context and list of question-goldAnswers associated to each context
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()

        context_length = len(context.split())
        context_lengths.append(context_length)

    return {
        'contexts_average_number_words': np.average(context_lengths),
        'contexts_min_number_words': np.min(context_lengths),
        'contexts_max_number_words': np.max(context_lengths)
    }


def build_dataset_MSEQA_format():
    # downloading raw dataset from huggingface repo
    # (has only "train" partition)
    raw_dataset = load_dataset("Universal-NER/Pile-NER-type")

    # populate list of {context-question-goldAnswers} elements
    context_question_list = []
    context_progressiveID = 0
    for raw_sample in raw_dataset['train']['conversations']:
        # extract context and list of question-goldAnswers associated to each context
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()
        question_progressiveID = 0
        # copy the context for each question associated to it
        for q_a in questions_answers_list:
            context_question_list.append(
                {"doc_question_pairID": str(context_progressiveID) + ":" + str(question_progressiveID),
                 "document_context": context,
                 "tagName": q_a["ne_type"],
                 "question": q_a["question"],
                 "answers": q_a["answers"]
                 })
            question_progressiveID += 1
        context_progressiveID += 1

    # 358181 context-question pairs
    # using 0.9 for training, 0.05 for validation, 0.05 for test
    train_ratio = 0.9
    validation_ratio = 0.05
    test_ratio = 0.05

    num_samples = len(context_question_list)
    num_train = int(train_ratio * num_samples)
    train_fold = context_question_list[:num_train]
    val_test_fold = context_question_list[num_train:]

    val_fold = val_test_fold[:math.floor(len(val_test_fold) / 2.0)]
    test_fold = val_test_fold[math.floor(len(val_test_fold) / 2.0):]

    # shuffling here after partitioning in fold so that same context is not both in train and val/test
    random.shuffle(train_fold)
    random.shuffle(val_fold)
    random.shuffle(test_fold)

    train_dataset = Dataset.from_list(train_fold)
    validation_dataset = Dataset.from_list(val_fold)
    test_dataset = Dataset.from_list(test_fold)

    dataset_MSEQA_format = DatasetDict({"train": train_dataset,
                                        "validation": validation_dataset,
                                        "test": test_dataset
                                        })
    return dataset_MSEQA_format


def remove_bad_ne_types(dataset_MSEQA_format):
    # get same NE types list for which we have GPT guidelines
    ne_types_list = get_ne_types_list(dataset_MSEQA_format, 100)
    print(len(ne_types_list))
    print(ne_types_list)
    # if the pileNER dataset is built by retaining only those NE which number of occurrences is > 100
    # the total number of NEs now should be 455
    # by plotting the dendrogram of the word embeddings using plot_word_emb.ipynb
    # we produce this mapping to a new list of NEs
    # by removing some bad NE categories or merging some
    # now the new list of NEs should be of length 423
    new_ne_type_list_mapping = {
        "misc": None,
        "miscellaneous": None,
        "other": None,
        "unknown": None,
        "general": None,
        "entity type not specified": None,
        "entity type": None,
        "entity": None,
        "text": None,
        "import": None,

        "bacteria": "bacterium",
        "biological": "biological entity",
        "cell": "cell type",
        "cellular component": "cell component",
        "governmental body": "government body",
        "movie": "film",
        "work": "work of art",
        "musical group": "music group",
        "org": "organization",

        "anatomical_structure": "anatomical structure",
        "anatomicalstructure": "anatomical structure",
        "biological_process": "biological process",
        "body_part": "body part",
        "gpe": "geopolitical entity",
        "gene/protein": "gene",
        "work_of_art": "work of art",
        "job_title": "job title",
        "organisation": "organization",
        "chemical_substance": "chemical substance",
        "medical_condition": "medical condition",
        "medicalcondition": "medical condition",

        "fieldterminology": None,
        "cryptocurrency": "cryptocurrency",
        "demonym": "demonym",
        "norp": "norp"
    }
    # new dataset with re-mapped Named Entities
    new_dataset_MSEQA_format_list = {split: [] for split in dataset_MSEQA_format.keys()}
    for split in dataset_MSEQA_format.keys():
        for sample in dataset_MSEQA_format[split]:
            ne_type = sample['tagName']
            old_ne_type = ne_type
            if ne_type in new_ne_type_list_mapping:
                ne_type = new_ne_type_list_mapping[ne_type]  # new NE name or None if to be removed
            # if has not been remove and the new mapping is in the list of NEs for which we have the gpt definition
            if ne_type is not None and ne_type in ne_types_list:
                # assign new NE type
                sample['tagName'] = ne_type
                # replacing the old ne type occurrence to their new UPPERCASE
                pattern = re.compile(re.escape(old_ne_type))
                sample['question'] = pattern.sub(ne_type.upper(), sample['question'])

                new_dataset_MSEQA_format_list[split].append(sample)

    return DatasetDict({split: Dataset.from_list(values) for split, values in new_dataset_MSEQA_format_list.items()})


def get_ne_types_list(dataset_MSEQA_format, min_num_samples_per_ne_type=100):
    """ list of NEs which number of answer spans (i.e. occurrences) across ALL splits is >= min_num_samples_per_ne_type """
    ne_types = {}
    for split in dataset_MSEQA_format.keys():
        for sample in dataset_MSEQA_format[split]:
            if sample["tagName"] in ne_types:
                ne_types[sample["tagName"]] += len(sample['answers']['text']) # number of occurrences
            else:
                ne_types[sample["tagName"]] = len(sample['answers']['text'])

    ne_types = [a[0] for a in sorted(ne_types.items(), key=lambda item: item[1], reverse=True) if
                a[1] >= min_num_samples_per_ne_type]

    #with open("./questions/ne_types_list.json", 'w') as f:
    #json.dump(ne_types, f, indent=2)

    return ne_types


""" --- functions to extract n sentences per NE type as examples to build definitions through GPT prompting --- """

def split_into_sentences(passage):
    # split the passage into sentences based on punctuation .?! while not splitting "Dr." or "Fig.1"
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s(?! \d)(?!\d)', passage)
    return [sentence for sentence in sentences if sentence.strip()]


def count_target_words(sentence, target_words):
    # count occurrences of the target words in the sentence
    # occurrences = [match.group() for match in re.finditer(r'\b(?:' + '|'.join(map(re.escape, target_words)) + r')\b', sentence, flags=re.IGNORECASE)]
    # return len(occurrences)
    matches = re.finditer(r'\b(?:' + '|'.join(map(re.escape, target_words)) + r')\b', sentence)

    # get the list of target words found in this sentence
    target_words_found = [match.group() for match in matches]

    # count the number of target words found in this sentence
    occurrences_count = len(target_words_found)

    return occurrences_count, target_words_found


def has_too_many_whitespaces(sentence, threshold=4):
    # count consecutive whitespaces
    consecutive_whitespaces = re.findall(r'\s+', sentence)

    # check if the count exceeds the threshold
    return any(len(whitespace) > threshold for whitespace in consecutive_whitespaces)


def has_too_many_newline(sentence, threshold=2):
    # count consecutive newline
    consecutive_newline = re.findall(r'\n+', sentence)

    # check if the count exceeds the threshold
    return any(len(whitespace) >= threshold for whitespace in consecutive_newline)


def has_more_than_n_foreign_chars(sentence, threshold=2):
    foreign_char_count = sum(1 for char in sentence if ord(char) > 127)
    return foreign_char_count > threshold

def has_too_many_punctuations_and_digits(sentence, threshold=5):
    # discard sentences like B [**\\#1**]{} (19\\#2) \\#3]{} \\#1\\#2
    # define the set of allowed punctuations
    allowed_punctuations = set(string.punctuation)
    # count the number of punctuations and digits in the sentence
    punctuation_count = sum(1 for char in sentence if char in allowed_punctuations or char.isdigit())
    return punctuation_count > threshold


def get_one_sentence_from_sample(sample):
    document_context = sample['document_context']
    answers = sample['answers']
    ne_type = sample['tagName']
    # split in sentences according to punctuation .?!
    sentences = split_into_sentences(document_context)
    target_words = answers['text']
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
            if 50 < len(target_word_counts[i]['sentence']) < 100:
                if not has_too_many_whitespaces(target_word_counts[i]['sentence'], 4):
                    if not has_too_many_newline(target_word_counts[i]['sentence'], 1):
                        if not has_more_than_n_foreign_chars(target_word_counts[i]['sentence'], 2):
                            if not has_too_many_punctuations_and_digits(target_word_counts[i]['sentence'], 10):
                                sentence_to_ret = target_word_counts[i]
                                break
            elif ne_type in ['namespace', 'import', 'keyword', 'surname', 'file name', 'header file', 'related art', 'boolean', 'struct', 'html attribute', 'protein domain', 'fieldterminology', 'constant', 'legal citation'] and len(target_word_counts[i]['sentence']) < 200:
                sentence_to_ret = target_word_counts[i]
                break

        i += 1

    return sentence_to_ret


def get_n_sentences_per_ne_type(dataset_MSEQA_format, ne_types_list, n_sentences_per_ne=3):
    # getting from training set n_sentences_per_ne as positive examples from which to let gpt infer NE definition
    sentences_per_ne_type = {ne: [] for ne in ne_types_list}
    trainDataset = dataset_MSEQA_format['train'].to_list()
    random.seed(4)
    random.shuffle(trainDataset)
    for ne_type in ne_types_list:
        i = 0
        while len(sentences_per_ne_type[ne_type]) < n_sentences_per_ne and i < len(trainDataset):
            sample = trainDataset[i]
            if sample['tagName'] == ne_type and len(sample['answers']['text']) != 0:
                sentence_target_words = get_one_sentence_from_sample(sample)
                if sentence_target_words is not None:
                    # removing duplicates in list of target words
                    sentence_target_words['target_words_in_it'] = list(set(sentence_target_words['target_words_in_it']))
                    sentences_per_ne_type[ne_type].append(sentence_target_words)
            i += 1

    not_enough_sentences = []
    for ne_type, sentences in sentences_per_ne_type.items():
        if len(sentences) < n_sentences_per_ne:
            # raise ValueError(f"not enough sentences for {ne_type}")
            not_enough_sentences.append((ne_type, len(sentences)))
    print(f"NE types with less than n_sentences_per_ne: {len(not_enough_sentences)}")
    print(not_enough_sentences)

    return sentences_per_ne_type


def generate_prompt(ne_type, example_sentences):
    # TODO: NOT used
    sentences_to_text = ""
    for sto in example_sentences:
        sentence = sto['sentence']
        target_words_in_it = sto['target_words_in_it']
        if len(target_words_in_it) == 1:
            sentences_to_text = sentences_to_text + "\"" + target_words_in_it[0] + "\""
            sentences_to_text += f" is \"{ne_type}\" in the sentence \"{sentence}\""
        else:
            for index, tw in enumerate(target_words_in_it):
                if index == len(target_words_in_it) - 1:
                    sentences_to_text = sentences_to_text + ' and ' + "\"" + tw + "\""
                elif index == len(target_words_in_it) - 2:
                    sentences_to_text = sentences_to_text + "\"" + tw + "\""
                else:
                    sentences_to_text = sentences_to_text + "\"" + tw + "\"" + ', '
            sentences_to_text += f" are \"{ne_type}\" in the sentence \"{sentence}\""
        sentences_to_text += '; '

    prompt = "Given a Named Entity and some exemplary sentences in input, provide, concisely, "
    prompt += "both a definition and warnings about what should not be labelled for this Named Entity. "
    prompt += "Here is the input: 'Named Entity': {}, 'examples of {}': {}".format(ne_type, ne_type, sentences_to_text)
    prompt += "Output in JSON format: {{\"Definition\": \"\", \"Do not label\": \"\"}}. Limit response to 50 tokens."

    return prompt


def generate_structured_prompt(ne_type, example_sentences):
    """ part of the prompt template to get NE definition from GPT, complete prompt conversation schema in the ipynb notebook """
    # unpacking sentences
    ex_sentences_json = []
    for exsent in example_sentences:
        ex_sentences_json.append({'sentence': exsent['sentence'], 'entities': exsent['target_words_in_it']})

    prompt = f"Named Entity: \'{ne_type}\'. Examples: {ex_sentences_json}.\n"
    prompt += f"Instructions: 1. Provide a concise definition for the named entity \'{ne_type}\' in the context of NER. 2. Provide guidelines by specifying what entities should not be labeled as \'{ne_type}\' and include potential pitfalls to avoid. Go beyond generic terms and delve into nuanced scenarios. Be explicit about potential ambiguities and provide guidance on distinguishing \'{ne_type}\' from similar entities.\n"
    prompt += "Output in JSON format: {\"Definition\": \"\", \"Guidelines\": \"\"}."
    return prompt


def build_dataset_MSEQA_format_with_guidelines(path_to_NE_guidelines_json):
    dataset_MSEQA_format = build_dataset_MSEQA_format()
    # dataset_MSEQA_format.save_to_disk("../../../datasets/pileNER_dataset_MSEQA_format")
    # dataset_MSEQA_format = DatasetDict.load_from_disk("../../../datasets/pileNER_dataset_MSEQA_format")

    ne_types_list = get_ne_types_list(dataset_MSEQA_format, 100)
    print(len(ne_types_list))
    print(ne_types_list)
    # if the pileNER dataset is built by retaining only those NE which number of occurrences is > 100
    # the total number of NEs now should be 455
    # by plotting the dendrogram of the word embeddings using plot_word_emb.ipynb
    # we produce this mapping to a new list of NEs
    # by removing some bad NE categories or merging some
    # now the new list of NEs should be of length 423
    new_ne_type_list_mapping = {
        "misc": None,
        "miscellaneous": None,
        "other": None,
        "unknown": None,
        "general": None,
        "entity type not specified": None,
        "entity type": None,
        "entity": None,
        "text": None,
        "import": None,

        "bacteria": "bacterium",
        "biological": "biological entity",
        "cell": "cell type",
        "cellular component": "cell component",
        "governmental body": "government body",
        "movie": "film",
        "work": "work of art",
        "musical group": "music group",
        "org": "organization",

        "anatomical_structure": "anatomical structure",
        "anatomicalstructure": "anatomical structure",
        "biological_process": "biological process",
        "body_part": "body part",
        "gpe": "geopolitical entity",
        "gene/protein": "gene",
        "work_of_art": "work of art",
        "job_title": "job title",
        "organisation": "organization",
        "chemical_substance": "chemical substance",
        "medical_condition": "medical condition",
        "medicalcondition": "medical condition",

        "fieldterminology": None,
        "cryptocurrency": "cryptocurrency",
        "demonym": "demonym",
        "norp": "norp"
    }

    # definition and guidelines for each NE in new_NE_type_list
    # obtained by prompting gpt, check prompt_tests.ipynb
    with open(path_to_NE_guidelines_json, 'r') as file:
        all_NEs_guidelines = json.load(file)

    # new dataset with re-mapped Named Entities and definition+guidelines+question
    new_dataset_MSEQA_format_list = {split: [] for split in dataset_MSEQA_format.keys()}
    for split in dataset_MSEQA_format.keys():
        for sample in dataset_MSEQA_format[split]:
            ne_type = sample['tagName']
            if ne_type in new_ne_type_list_mapping:
                ne_type = new_ne_type_list_mapping[ne_type]  # new NE name or None if to be removed
            # if has not been remove and the new mapping is in the list of NEs for which we have the gpt definition
            if ne_type is not None and ne_type in ne_types_list:
                # new NE type
                sample['tagName'] = ne_type
                # from string to dict
                gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
                # print(gpt_definition.strip())
                # gpt answer may have been truncated, ensure it ends by "} before evaluating to dict
                if not gpt_definition.endswith("}"):
                    if not gpt_definition.endswith("\""):
                        gpt_definition += "\""
                    gpt_definition += "}"
                # print(gpt_definition)
                this_ne_guidelines = eval(gpt_definition)
                # replacing ne types occurrences between single quotes to their UPPERCASE
                pattern = re.compile(rf'\'{re.escape(ne_type)}\'')
                this_ne_guidelines = {k: pattern.sub(f'{ne_type.upper()}', v) for k, v in this_ne_guidelines.items()}

                question = f"Your task is to extract the Named Entities of type {ne_type.upper()} from an input TEXT. "
                question += "You are given a DEFINITION and some GUIDELINES.\n"
                question += "DEFINITION: " + this_ne_guidelines['Definition'] + "\nGUIDELINES: " + this_ne_guidelines['Guidelines'] + "\n"
                question += f"TEXT: "
                sample['question'] = question

                new_dataset_MSEQA_format_list[split].append(sample)

    return DatasetDict({split: Dataset.from_list(values) for split, values in new_dataset_MSEQA_format_list.items()})


def add_negative_examples_to_MSEQA_dataset(dataset_MSEQA_format_w_guidelines, path_to_NE_guidelines_json):

    # loading gpt guidelines for each NE type in pileNER (only of top frequent NEs)
    with open(path_to_NE_guidelines_json, 'r') as file:
        all_NEs_guidelines = json.load(file)

    # everything is done in each split independently
    splits = ['train', 'validation', 'test']

    # for each Dataset split count how many positive samples (i.e. how many questions) of a ne_type exist
    # e.g. {'train': {'gene': 5, 'sports team': 4, 'norp': 3, 'disease': 9,...}
    # --> there are 5 MSEQA samples which question is about 'gene' NE type and have a non-empty answer
    number_positive_questions_per_ne_type = {split: {} for split in splits}
    for split in dataset_MSEQA_format_w_guidelines.keys():
        for sample in dataset_MSEQA_format_w_guidelines[split]:
            ne_type = sample['tagName']
            if ne_type in number_positive_questions_per_ne_type[split]:
                number_positive_questions_per_ne_type[split][ne_type] += 1
            else:
                number_positive_questions_per_ne_type[split][ne_type] = 1
    print("Number of positive questions per NE type:")
    print(number_positive_questions_per_ne_type)

    # list of ne_types per split
    # e.g. {'train': ['gene', 'sports team', 'norp', .. ]
    ne_types_list_per_split = {split: list(values.keys()) for split, values in number_positive_questions_per_ne_type.items()}
    print("List of positive NE types per split:")
    print(ne_types_list_per_split)
    # {'train': 423, 'validation': 416, 'test': 414}
    print("Number of positive NE types per split: {}".format({split: len(values) for split, values in ne_types_list_per_split.items()}))

    # for now are 0 (there are no new NEs never seen in train)
    # TODO: add novel NEs
    NEs_in_validation_but_not_train = list(set(ne_types_list_per_split['validation']) - set(ne_types_list_per_split['train']))
    print("{} new NEs in validation but not in train: {}".format(len(NEs_in_validation_but_not_train), NEs_in_validation_but_not_train))
    NEs_in_test_but_not_train = list(set(ne_types_list_per_split['test']) - set(ne_types_list_per_split['train']))
    print("{} new NEs in test but not in train: {}".format(len(NEs_in_test_but_not_train), NEs_in_test_but_not_train))

    # define a subset of valid ne_types for each passage of text from which to draw negative questions
    # first find the already existing NE types on each passage of text
    positive_NEs_per_passage = {split: {} for split in splits}
    for split in dataset_MSEQA_format_w_guidelines.keys():
        for sample in dataset_MSEQA_format_w_guidelines[split]:
            passage_id = sample['doc_question_pairID'].split(':')[0]
            if passage_id in positive_NEs_per_passage[split]:
                positive_NEs_per_passage[split][passage_id].append(sample['tagName'])
            else:
                positive_NEs_per_passage[split][passage_id] = [sample['tagName']]
    # now set for each passage the candidate ne_types (i.e. all_ne_types - positive_ne_types_for_this_passage)
    candidate_NEs_subset_per_passage = {split: {} for split in splits}

    # generating a negative sample for a generic NE as "concept" may confuse the model as many entities could be seen as "concept",
    # but labeled in the samples as other more specific NEs
    general_invalid_ne_types = ["concept", "location", "product", "technology", "object", "number", "attribute", "group", "process", "function",
                                "material", "type", "quantity", "data type", "data", "biological entity", "task", "resource", "biomolecule", "unit", "physical quantity", "information", "year", "acronym"
                                "group of people", "adjective", "string", "part", "landmark", "pronoun", "trait", "outcome", "financial", "verb", "keyword", "setting", "environment",
                                "item", "mechanism", "entertainment", "term", "noun"]
    for split in positive_NEs_per_passage.keys():
        for passage_id, positive_ne_list_for_this_passage in positive_NEs_per_passage[split].items():
            candidate_NEs_subset_per_passage[split][passage_id] = list((set(ne_types_list_per_split[split]) - set(general_invalid_ne_types)) - set(positive_ne_list_for_this_passage))

    n_passages_per_split = {split: len(candidate_NEs_subset_per_passage[split]) for split in splits}
    print("Number of passages per split: {}".format(n_passages_per_split))

    #print("Candidate_NEs_subset_per_passage: ")
    #print(candidate_NEs_subset_per_passage)

    negative_NEs_to_add = {split: [] for split in splits}
    for split in splits:
        total_positive_examples = sum(number_positive_questions_per_ne_type[split].values())
        for passage_id, candidate_NEs_as_negatives in candidate_NEs_subset_per_passage[split].items():
            # compute sampling probabilities for candidate NEs as negative questions for this passage
            sampling_probabilities = [number_positive_questions_per_ne_type['train'][ne] / total_positive_examples for ne in candidate_NEs_as_negatives]
            # number of negative questions to add per passage
            num_negative_questions_to_add_per_passage = 3
            sampled_negative_NEs = random.choices(candidate_NEs_as_negatives, weights=sampling_probabilities, k=num_negative_questions_to_add_per_passage)
            negative_NEs_to_add[split].append({"passage_id": passage_id, "negative_NEs_to_add": sampled_negative_NEs})

    #print(negative_NEs_to_add)

    document_contexts = {split: {} for split in splits}
    for split in splits:
        for sample in dataset_MSEQA_format_w_guidelines[split]:
            passage_id = sample['doc_question_pairID'].split(':')[0]
            if passage_id not in document_contexts[split]:
                document_contexts[split][passage_id] = sample['document_context']

    negative_samples_to_add_per_split = {split: [] for split in splits}
    # now it's type to add to the exising positive samples the negatives ones
    for split in splits:
        for passage_id_negative_NEs_to_add_on_this_passage in negative_NEs_to_add[split]:
            passage_id, negative_NEs_to_add_on_this_passage = passage_id_negative_NEs_to_add_on_this_passage.values()
            for neg_ne_id, negative_ne in enumerate(negative_NEs_to_add_on_this_passage):
                # constructing question with negative_ne
                gpt_definition = all_NEs_guidelines[negative_ne]['gpt_answer'].strip()
                # print(gpt_definition.strip())
                # gpt answer may have been truncated, ensure it ends by "} before evaluating to dict
                if not gpt_definition.endswith("}"):
                    if not gpt_definition.endswith("\""):
                        gpt_definition += "\""
                    gpt_definition += "}"
                # print(gpt_definition)
                this_ne_guidelines = eval(gpt_definition)
                # replacing ne types occurrences between single quotes to their UPPERCASE
                pattern = re.compile(rf'\'{re.escape(negative_ne)}\'')
                this_ne_guidelines = {k: pattern.sub(f'{negative_ne.upper()}', v) for k, v in this_ne_guidelines.items()}

                question = f"Your task is to extract the Named Entities of type {negative_ne.upper()} from an input TEXT. "
                question += "You are given a DEFINITION and some GUIDELINES.\n"
                question += "DEFINITION: " + this_ne_guidelines['Definition'] + "\nGUIDELINES: " + this_ne_guidelines['Guidelines'] + "\n"
                question += f"TEXT: "

                negative_sample = {"doc_question_pairID": str(passage_id) + ":" + str(neg_ne_id) + "-neg",
                                   "document_context": document_contexts[split][passage_id],
                                   "tagName": negative_ne,
                                   "question": question,
                                   "answers": {
                                            'answer_start': list(),
                                            'text': list()
                                        }
                                   }
                negative_samples_to_add_per_split[split].append(negative_sample)

    from datasets import Features, Value, Sequence

    # Define the features for each key
    features = Features({
        "doc_question_pairID": Value(dtype='string', id=None),
        "document_context": Value(dtype='string', id=None),
        "tagName": Value(dtype='string', id=None),
        "question": Value(dtype='string', id=None),
        "answers": {
            'answer_start': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
            'text': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
        }
    })

    for split in splits:
        positive_samples = dataset_MSEQA_format_w_guidelines[split]
        negative_sample = Dataset.from_list(negative_samples_to_add_per_split[split], features=features)
        concatenated_dataset = concatenate_datasets([positive_samples, negative_sample])
        shuffled_ds = concatenated_dataset.shuffle()
        dataset_MSEQA_format_w_guidelines[split] = shuffled_ds

    return dataset_MSEQA_format_w_guidelines


def convert_official_uniNER_eval_dataset_for_inference(dataset_name, path_to_dataset, with_definition=False, path_to_NE_guidelines_json=None):

    with open(path_to_dataset, 'r') as fh:
        uniNER_eval_samples = json.load(fh)

    all_NEs_guidelines = None
    if with_definition:
        with open(path_to_NE_guidelines_json, 'r') as file:
            all_NEs_guidelines = json.load(file)

    # converting list to dict for fast access
    if all_NEs_guidelines and isinstance(all_NEs_guidelines, list):
        all_NEs_guidelines = {x['named_entity']: x for x in all_NEs_guidelines}

    dataset_for_inference = []  # dataset being constructed
    for uniNER_sample in uniNER_eval_samples:

        context, questions_answers_list = extract_context_quests_answers(uniNER_sample['conversations']).values()

        if len(questions_answers_list) > 1:
            raise ValueError("Expected only 1 question")

        question, ne_type, answers = questions_answers_list[0].values()

        if with_definition:
            # some uniNER NEs are different from the original NEs
            try:
                gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
            except KeyError:
                if dataset_name in ['ai', 'literature', 'science', 'politics', 'music']:
                    ne_mapping = {
                        'organization': 'organisation',
                        'program language': 'programlang',
                        'literary genre': 'literarygenre',
                        'astronomical object': 'astronomicalobject',
                        'chemical element': 'chemicalelement',
                        'chemical compound': 'chemicalcompound',
                        'academic journal': 'academicjournal',
                        'political party': 'politicalparty',
                        'musical artist': 'musicalartist',
                        'musical instrument': 'musicalinstrument',
                        'music genre': 'musicgenre',
                    }
                elif dataset_name == 'movie':
                    ne_mapping = {
                        'character': 'CHARACTER',
                        'plot': 'PLOT',
                        'year': 'YEAR',
                        'director': 'DIRECTOR',
                        'rating': 'RATING',
                        'average ratings': 'RATINGS_AVERAGE',
                        'actor': 'ACTOR',
                        'genre': 'GENRE',
                        'song': 'SONG',
                        'trailer': 'TRAILER',
                        'review': 'REVIEW',
                        'title': 'TITLE'
                    }
                elif dataset_name == 'restaurant':
                    ne_mapping = {
                        'amenity': 'Amenity',
                        'location': 'Location',
                        'cuisine': 'Cuisine',
                        'restaurant name': 'Restaurant_Name',
                        'rating': 'Rating',
                        'hours': 'Hours',
                        'price': 'Price',
                        'dish': 'Dish'
                    }

                ne_type = ne_mapping[ne_type]
                gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()

            # gpt answer may have been truncated, ensure it ends by "} before evaluating to dict
            if not gpt_definition.endswith("}"):
                if not gpt_definition.endswith("\""):
                    gpt_definition += "\""
                gpt_definition += "}"

            this_ne_guidelines = eval(gpt_definition)
            # replacing ne types occurrences between single quotes to their UPPERCASE
            ne_type_in_natural_language = all_NEs_guidelines[ne_type]['real_name']
            pattern = re.compile(rf'\'{re.escape(ne_type_in_natural_language)}\'')
            this_ne_guidelines = {k: pattern.sub(f'{ne_type_in_natural_language.upper()}', v) for k, v in this_ne_guidelines.items()}

            question = f"Your task is to extract the Named Entities of type {ne_type_in_natural_language.upper()} from an input TEXT. "
            question += "You are given a DEFINITION and some GUIDELINES.\n"
            question += "DEFINITION: " + this_ne_guidelines['Definition'] + "\nGUIDELINES: " + this_ne_guidelines['Guidelines'] + "\n"
            question += f"TEXT: "

        inference_sample = {
            "doc_question_pairID": uniNER_sample['id'],
            "document_context": context,
            "tagName": ne_type,
            "question": question,
            "answers": answers
        }
        dataset_for_inference.append(inference_sample)

    return DatasetDict({"test": Dataset.from_list(dataset_for_inference)})


if __name__ == "__main__":

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def parser(text):
        try:
            match = re.match(r'\[(.*?)\]', text)
            if match:
                text = match.group()
            else:
                text = '[]'
            items = json.loads(text)
            formatted_items = []
            for item in items:
                if isinstance(item, list) or isinstance(item, tuple):
                    item = tuple([normalize_answer(element) for element in item])
                else:
                    item = normalize_answer(item)
                if item not in formatted_items:
                    formatted_items.append(item)
            return formatted_items
        except Exception:
            return []


    class NEREvaluator:
        def evaluate(self, preds: list, golds: list):
            n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
            for pred, gold in zip(preds, golds):
                gold_tuples = parser(gold)
                pred_tuples = parser(pred)
                for t in pred_tuples:
                    if t in gold_tuples:
                        n_correct += 1
                    n_pos_pred += 1
                n_pos_gold += len(gold_tuples)
            prec = n_correct / (n_pos_pred + 1e-10)
            recall = n_correct / (n_pos_gold + 1e-10)
            f1 = 2 * prec * recall / (prec + recall + 1e-10)
            return {
                'precision': prec,
                'recall': recall,
                'f1': f1,
            }

    gold_sample = "[\"naive Bayes classifier\", \"Gaussian mixture model\", \"variational autoencoders\"]"
    print(gold_sample)

    # Esempio d'uso:
    strings = ["naive Bayes classifier", "Gaussian mixture model", "variational autoencoders"]
    filename = "output.json"
    with open("./prova.json", "w") as f:
        f.write(str(strings))


    path_to_eval_dataset_uniNER = f'../../../datasets/eval_data_UniNER/test_data/mit-movie.json'
    with open(path_to_eval_dataset_uniNER, 'r') as fh:
        eval_dataset_uniNER = json.load(fh)
    golds = [example['conversations'][-1]['value'] for example in eval_dataset_uniNER]
    print(golds[0:100])
    print(type(golds[0]))
    print(type(golds[7]))

    with open('./movie_preds.json', 'r') as fh:
        preds = json.load(fh)

    preds = [json.dumps(x['pred_answers']) for x in preds]

    print(preds[0:100])
    print(type(preds[0]))
    print(type(preds[7]))

    eval_result = NEREvaluator().evaluate(preds, golds)

    print(f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}')


    """
    first_list = [{'id': 'CrossNER_literature_0'}, {'id': 'CrossNER_literature_1'}, {'id': 'CrossNER_literature_2'}]
    second_list = [{'doc_question_pairID': 'CrossNER_literature_2'}, {'doc_question_pairID': 'CrossNER_literature_1'}, {'doc_question_pairID': 'CrossNER_literature_0'}]

    # Sort the second list based on the order of IDs from the first list
    sorted_second_list = sorted(second_list,key=lambda x: [d['id'] for d in first_list].index(x['doc_question_pairID']))
    print(sorted_second_list)

    dataset_name = 'ai'
    data_path = f'../../../datasets/eval_data_UniNER/test_data/CrossNER_{dataset_name}.json'
    #data_path = f'../../../datasets/eval_data_UniNER/test_data/mit-{dataset_name}.json'
    with open(data_path, 'r') as fh:
        examples = json.load(fh)

    print(type(examples))
    print(len(examples))
    print(examples[0])

    dataset_for_inference_MSEQA = convert_official_uniNER_eval_dataset_for_inference(dataset_name, data_path, with_definition=True, path_to_NE_guidelines_json=f'./questions/crossNER/gpt_guidelines/{dataset_name}_NE_definitions.json')
    print(dataset_for_inference_MSEQA)
    print(dataset_for_inference_MSEQA['test'][0])
    print(dataset_for_inference_MSEQA['test'][1])
    print(dataset_for_inference_MSEQA['test'][10])

    #for sample in dataset_for_inference_MSEQA['test']:
        #print(sample)

    """

    """

    raw_dataset = load_dataset("Universal-NER/Pile-NER-type")
    print(raw_dataset)

    #pileNER_raw_statistics = get_dataset_statistics()
    #print(pileNER_raw_statistics)

    print(raw_dataset['train'][8840]['conversations'])
    
    context, quests_answers = extract_context_quests_answers(raw_dataset['train'][8840]['conversations']).values()
    print(context)

    print("\n")
    for qa in quests_answers:
        print(qa)
        for a_s, text in zip(qa['answers']['answer_start'], qa['answers']['text']):
            print(context[a_s:a_s+15])
            print(text)
            print("---")

    """

    """
    # uniNER_dataset_MSEQA_format = build_dataset_MSEQA_format()
    # uniNER_dataset_MSEQA_format.save_to_disk("../../../datasets/uniNER_dataset_MSEQA_format")
    #uniNER_dataset_MSEQA_format = DatasetDict.load_from_disk("../../../datasets/pileNER_dataset_MSEQA_format")
    #print(uniNER_dataset_MSEQA_format['train'][0])

    # for i in range(10):
    # print(uniNER_dataset_MSEQA_format['train'][i])

    ne_types = {split: {} for split in uniNER_dataset_MSEQA_format.keys()}
    for split in uniNER_dataset_MSEQA_format.keys():
        for sample in uniNER_dataset_MSEQA_format[split]:
            if sample["tagName"] in ne_types[split]:
                ne_types[split][sample["tagName"]] += len(sample['answers']['text'])
            else:
                ne_types[split][sample["tagName"]] = len(sample['answers']['text'])

    ne_types = {split:dict(sorted(values.items(), key=lambda item: item[1], reverse=True)).keys() for split, values in ne_types.items()}
    """

    with open("./questions/pileNER/ne_types_list.json", 'r') as file:
        ne_types_list = json.load(file)

    print("NE types which number of occurrences is > 100:")
    print(len(ne_types_list))
    print(ne_types_list)

    """
    for i in range(20):
        extracted_sentence = get_one_sentence_from_sample(uniNER_dataset_MSEQA_format['train'][i])
        print(len(extracted_sentence))
        print(extracted_sentence)
        print("-------------")
    """

    """
    sentences_per_ne_type = get_n_sentences_per_ne_type(uniNER_dataset_MSEQA_format, ne_types_list, n_sentences_per_ne=3)
    #for ne, sentences in sentences_per_ne_type.items():
    #print(ne, sentences)

    with open("./questions/sentences_per_ne_type.json", 'w') as f:
        json.dump(sentences_per_ne_type, f, indent=2)
    """

    with open("./questions/pileNER/sentences_per_ne_type.json", 'r') as file:
        sentences_per_ne_type = json.load(file)

    #print(sentences_per_ne_type)

    ne_type = 'location'
    ex_sentences = sentences_per_ne_type[ne_type]
    prompt = generate_structured_prompt(ne_type, ex_sentences)
    print("\n")
    print(prompt)

    print("\n")

    """
    dataset_MSEQA_format = build_dataset_MSEQA_format()
    print(dataset_MSEQA_format)
    dataset_MSEQA_format_removed_NEs = remove_bad_ne_types(dataset_MSEQA_format)
    print(dataset_MSEQA_format_removed_NEs)
    print(dataset_MSEQA_format_removed_NEs['train'][0])
    print(dataset_MSEQA_format_removed_NEs['train'][1])
    print(dataset_MSEQA_format_removed_NEs['train'][23])
    print(dataset_MSEQA_format_removed_NEs['train'][100])
    """

    """
    #dataset_MSEQA_format_with_guidelines = DatasetDict.load_from_disk("../../../datasets/dataset_MSEQA_format_with_guidelines")
    dataset_MSEQA_format_with_guidelines = build_dataset_MSEQA_format_with_guidelines("./questions/pileNER/all_423_NE_definitions.json")

    print(dataset_MSEQA_format_with_guidelines)
    print(dataset_MSEQA_format_with_guidelines['train'][400])
    print(dataset_MSEQA_format_with_guidelines['train'][443])
    print(dataset_MSEQA_format_with_guidelines['train'][432])
    print(dataset_MSEQA_format_with_guidelines['train'][732])

    #dataset_MSEQA_format_with_guidelines.save_to_disk("../../../datasets/dataset_MSEQA_format_with_guidelines")
    """

    """
    for split_name, split_dataset in dataset_MSEQA_format_with_guidelines.items():
        # Keep only the first num_samples_to_keep samples
        limited_dataset = split_dataset.select(range(1000))
        # Replace the original split with the limited dataset
        dataset_MSEQA_format_with_guidelines[split_name] = limited_dataset

    print(dataset_MSEQA_format_with_guidelines)
    """
    print("\n")

    """
    for sample in dataset_MSEQA_format_with_guidelines['train']:
        if sample['doc_question_pairID'].split(':')[0] == '12820':
            print(sample)
    """

    """
    dataset_MSEQA_format_with_guidelines_NEG_samples = add_negative_examples_to_MSEQA_dataset(dataset_MSEQA_format_with_guidelines, "./questions/pileNER/all_423_NE_definitions.json")
    print(dataset_MSEQA_format_with_guidelines_NEG_samples)

    for sample in dataset_MSEQA_format_with_guidelines_NEG_samples['train']:
        if sample['document_context'].endswith('-neg'):
            print(sample)
            break
    """
