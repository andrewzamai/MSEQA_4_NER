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

__package__ = "MSEQA_4_NER.data_handlers"

import os
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
    n_total_samples = 0
    n_negative_samples = 0
    for raw_sample in raw_dataset['train']['conversations']:
        # extract context and list of question-goldAnswers associated to each context
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()

        context_length = len(context.split())
        context_lengths.append(context_length)

        for q_ne_answers in questions_answers_list:
            n_total_samples += 1
            if q_ne_answers['answers']['text'] == []:
                n_negative_samples += 1

    fullPileNER_tagName_list = {}
    for raw_sample in raw_dataset['train']['conversations']:
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()
        for question in questions_answers_list:
            fullPileNER_tagName_list[question['ne_type']] = 1

    return {
        'contexts_average_number_words': np.average(context_lengths),
        'contexts_min_number_words': np.min(context_lengths),
        'contexts_max_number_words': np.max(context_lengths),
        'fullPileNER_tagName_list': [len(list(fullPileNER_tagName_list.keys())), list(fullPileNER_tagName_list.keys())],
        'number_total_QA_samples': n_total_samples,
        'number_negative_QA_samples': [n_negative_samples, f"{n_negative_samples/n_total_samples*100}%"]
    }


def get_statistics_for_QA_dataset(dataset_QA, input_column_name, instruction_column_name, output_column_name):
    """ get statistics for MSEQA/GenQA Dataset fold (e.g. train) """
    context_lengths = []
    n_total_samples = 0
    n_negative_samples = 0
    for sample in dataset_QA:
        # counting number words approximately
        context = sample[input_column_name]
        context_length = len(context.split())
        context_lengths.append(context_length)

        # counting number negative samples
        output = sample[output_column_name]
        answers_text = None
        # MSEQA case {'answer_start': [], 'text': []}
        if isinstance(output, dict):
            if 'text' in output:
                answers_text = output['text']
            else:
                raise Exception("Unexpected keys, expected 'text'")
        # GenQA case is a dumped JSON list
        elif isinstance(output, str):
            try:
                answers_text = json.loads(output)
            except:
                answers_text = []

        n_total_samples += 1
        if not answers_text:
            n_negative_samples += 1

    # list of unique NEs
    tagName_list = {}
    for sample in dataset_QA:
        tagName_list[sample['tagName']] = 1

    return {
        'contexts_average_number_words': math.ceil(np.average(context_lengths)),
        'contexts_min_number_words': np.min(context_lengths),
        'contexts_max_number_words': np.max(context_lengths),
        'fullPileNER_tagName_list': [len(list(tagName_list.keys())), list(tagName_list.keys())],
        'number_total_QA_samples': n_total_samples,
        'number_negative_QA_samples': [n_negative_samples, f"{round(n_negative_samples/n_total_samples*100, 2)}%"]
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
    random.seed(42) #42
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
    # print(len(ne_types_list))
    # print(ne_types_list)

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


def build_dataset_MSEQA_format_with_guidelines(path_to_NE_guidelines_json, dataset_MSEQA_format=None):
    # to simply convert an existing MSEQA FalseDef datasetDict to TrueDef
    if not dataset_MSEQA_format:
        dataset_MSEQA_format = build_dataset_MSEQA_format()
        min_num_samples_per_ne_type = 100
    else:
        # FalseDef assumed to be already with required samples only
        min_num_samples_per_ne_type = -1

    # dataset_MSEQA_format.save_to_disk("../../../datasets/pileNER_dataset_MSEQA_format")
    # dataset_MSEQA_format = DatasetDict.load_from_disk("../../../datasets/pileNER_dataset_MSEQA_format"
    ne_types_list = get_ne_types_list(dataset_MSEQA_format, min_num_samples_per_ne_type)
    #print(len(ne_types_list))
    #print(ne_types_list)

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
    # DEPRECATED (because I discovered that pileNER already has negative examples added)

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
    """
    converts a eval dataset provided by uniNER github for inference through a MSEQA model;
    i.e. applies Definition if with_definition=True, some NEs mapping and renames features as expected by MSEQA models
    """

    with open(path_to_dataset, 'r') as fh:
        uniNER_eval_samples = json.load(fh)

    with open(path_to_NE_guidelines_json, 'r') as f:
        all_NEs_guidelines = json.load(f)

    # converting list to dict for fast access
    if all_NEs_guidelines and isinstance(all_NEs_guidelines, list):
        all_NEs_guidelines = {x['named_entity']: x for x in all_NEs_guidelines}

    dataset_for_inference = []  # dataset being constructed
    for uniNER_sample in uniNER_eval_samples:

        context, questions_answers_list = extract_context_quests_answers(uniNER_sample['conversations']).values()

        if len(questions_answers_list) > 1:
            raise ValueError("Was expected only 1 question here!")

        question, ne_type, answers = questions_answers_list[0].values()

        # some uniNER NEs are different from the original NEs
        try:
            gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']
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
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']

        if with_definition:
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

        # else: question is already "What describes NE in the text?"

        inference_sample = {
            "doc_question_pairID": uniNER_sample['id'],
            "document_context": context,
            "tagName": ne_type,
            "question": question,
            "answers": answers
        }
        dataset_for_inference.append(inference_sample)

    return DatasetDict({"test": Dataset.from_list(dataset_for_inference)})


def convert_MSEQA_dataset_to_GenQA_format(dataset_MSEQA_format, with_definition=True, path_to_save_to='./unk_dataset_GenQA'):

    for split_name in dataset_MSEQA_format.keys():
        dataset_GenQA = []
        for MSEQA_sample in dataset_MSEQA_format[split_name]:
            genQA_sample = {
                "doc_question_pairID": MSEQA_sample['doc_question_pairID'],
                "tagName": MSEQA_sample['tagName'],
                # new column names as finetune_sft.py requires
                "input": MSEQA_sample['document_context'],
                "instruction": "",
                "output": ""
            }
            if with_definition:
                instruction = MSEQA_sample['question']
                instruction = instruction.replace("from an input TEXT", "from the text chunk you have read")
                instruction = instruction.replace("Your task is to extract", "Extract")
                instruction = instruction.replace("\nTEXT: ", "\nReturn a JSON list.")
                # instruction = instruction[:-len("\nTEXT: ")]
                genQA_sample['instruction'] = instruction
            else:
                # TODO: rephrase question What describes X in the text?
                genQA_sample['instruction'] = MSEQA_sample['question']

            # sorting the text answers by ascending starting positions to give the LLM a pattern: extract the occurences in the order they appear in the passage of text
            # this is because although the evaluation metrics are order independent the NTP loss penalizes order
            # we also delete duplicate occurrences thus obtaining a SET of gold_answers
            gold_answers_with_char_starts = MSEQA_sample['answers']
            # sort text answers by ascending start positions
            sorted_start_answers = sorted(zip(gold_answers_with_char_starts['answer_start'], gold_answers_with_char_starts['text']), key=lambda x: x[0])
            # retrieve only text answers
            sorted_answers_text_only = [item[1] for item in sorted_start_answers]
            # deleting any duplicate while preserving order (order within document context)
            sorted_textonly_gold_answers_wo_duplicates = list(OrderedDict.fromkeys(sorted_answers_text_only).keys())
            #genQA_sample["output"] = str(sorted_textonly_gold_answers_wo_duplicates)  # stringifying list
            genQA_sample["output"] = json.dumps(sorted_textonly_gold_answers_wo_duplicates)  # stringifying list

            dataset_GenQA.append(genQA_sample)

        dataset_GenQA = Dataset.from_list(dataset_GenQA)

        dataset_GenQA.to_json(os.path.join(path_to_save_to, split_name + '.jsonl'))

def convert_MSEQA_dataset_to_GenQA_format_SI(dataset_MSEQA_format, with_definition=True, path_to_save_to='./unk_dataset_GenQA'):

    for split_name in dataset_MSEQA_format.keys():
        dataset_GenQA = []
        for MSEQA_sample in dataset_MSEQA_format[split_name]:
            genQA_sample = {
                "doc_question_pairID": MSEQA_sample['doc_question_pairID'],
                "tagName": MSEQA_sample['tagName'],
                # new column names as finetune_sft.py requires
                "input": MSEQA_sample['document_context'],
                "instruction": "",
                "output": ""
            }
            if with_definition:
                instruction = MSEQA_sample['question']
                instruction = instruction.replace("from an input TEXT", "from the text chunk you have read")
                instruction = instruction.replace("Your task is to extract", "Extract")
                # instruction = instruction.replace("\nTEXT: ", "\nReturn a JSON list.")
                instruction = instruction.replace("\nTEXT: ", f"\nReturn a JSON list of instances of this Named Entity type. Return an empty list if no instances are present.")
                genQA_sample['instruction'] = instruction
            else:
                # TODO: rephrase question What describes X in the text?
                instruction_wo_guidelines = f"Extract the Named Entities of type {MSEQA_sample['tagName'].upper()} from the text chunk you have read."
                instruction_wo_guidelines += "\nReturn a JSON list of instances of this Named Entity type. Return an empty list if no instances are present."
                #genQA_sample['instruction'] = MSEQA_sample['question']
                genQA_sample['instruction'] = instruction_wo_guidelines

            # sorting the text answers by ascending starting positions to give the LLM a pattern: extract the occurences in the order they appear in the passage of text
            # this is because although the evaluation metrics are order independent the NTP loss penalizes order
            # we also delete duplicate occurrences thus obtaining a SET of gold_answers
            gold_answers_with_char_starts = MSEQA_sample['answers']
            # sort text answers by ascending start positions
            sorted_start_answers = sorted(zip(gold_answers_with_char_starts['answer_start'], gold_answers_with_char_starts['text']), key=lambda x: x[0])
            # retrieve only text answers
            sorted_answers_text_only = [item[1] for item in sorted_start_answers]
            # deleting any duplicate while preserving order (order within document context)
            sorted_textonly_gold_answers_wo_duplicates = list(OrderedDict.fromkeys(sorted_answers_text_only).keys())
            #genQA_sample["output"] = str(sorted_textonly_gold_answers_wo_duplicates)  # stringifying list
            genQA_sample["output"] = json.dumps(sorted_textonly_gold_answers_wo_duplicates)  # stringifying list

            dataset_GenQA.append(genQA_sample)

        dataset_GenQA = Dataset.from_list(dataset_GenQA)

        dataset_GenQA.to_json(os.path.join(path_to_save_to, split_name + '.jsonl'))


def convert_official_uniNER_eval_dataset_for_GenQA(dataset_name, path_to_dataset, with_definition=False, path_to_NE_guidelines_json=None):
    """
    Adapting UniNER eval datasets mit/crossNER for eval with Generative LLMs.
    Changing document_context column name to 'input' and answers to 'output'.
    Adding NE Guidelines as 'instruction'
    """

    with open(path_to_dataset, 'r') as fh:
        uniNER_eval_samples = json.load(fh)

    # we load guidelines also if with_def False to make NE mapping to canonical names (uniNER eval NEs are different)
    with open(path_to_NE_guidelines_json, 'r') as file:
        all_NEs_guidelines = json.load(file)

    # converting list to dict for fast access
    if all_NEs_guidelines and isinstance(all_NEs_guidelines, list):
        all_NEs_guidelines = {x['named_entity']: x for x in all_NEs_guidelines}

    dataset_GenQA = []  # dataset being constructed
    for uniNER_sample in uniNER_eval_samples:

        context, questions_answers_list = extract_context_quests_answers(uniNER_sample['conversations']).values()

        if len(questions_answers_list) > 1:
            raise ValueError("Expected only 1 question")

        question, ne_type, answers = questions_answers_list[0].values()

        # some uniNER NEs are different from the original NEs
        try:
            gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
            # NE name in natural languange form, e.g. ORG --> organization
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']
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
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']

        if with_definition:
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

            # adapting to reverse_INST template Prompter
            question = question.replace("from an input TEXT", "from the text chunk you have read")
            question = question.replace("Your task is to extract", "Extract")
            question = question.replace("\nTEXT: ", "\nReturn a JSON list.")

        genQA_sample = {
            "doc_question_pairID": uniNER_sample['id'],
            "input": context,
            "tagName": ne_type,  # real_name_ne if want to mask in evaluation given tagName
            "instruction": question,
            "output": uniNER_sample['conversations'][-1]['value']
        }
        dataset_GenQA.append(genQA_sample)

    return Dataset.from_list(dataset_GenQA)


def mask_named_entities_probability_proportional(dataset_split):
    # count how many samples (i.e. how many questions) of a ne_type exist
    # e.g. {'train': {'gene': 5, 'sports team': 4, 'norp': 3, 'disease': 9,...}
    # --> there are 5 MSEQA samples which question is about 'gene' NE type and have a empty/non-empty answer
    number_samples_per_ne_type = {}
    for sample in dataset_split:
        ne_type = sample['tagName']
        if ne_type in number_samples_per_ne_type:
            number_samples_per_ne_type[ne_type] += 1
        else:
            number_samples_per_ne_type[ne_type] = 1
    print("Number of samples per NE type:")
    print(sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True))

    total_num_samples = sum(x for x in number_samples_per_ne_type.values())
    tagName_probabilities = {tagName: x / total_num_samples for tagName, x in number_samples_per_ne_type.items()}
    #print(sorted(tagName_probabilities.items(), key=lambda x: x[1], reverse=True))

    # calculate scaling factor and apply maximum corruption probability threshold
    scaling_factor = max(tagName_probabilities.values())
    # scaling_factor = 0.5
    max_corruption_prob_threshold = 0.5  # Set a maximum corruption probability threshold
    # min_corruption_prob_threshold = 0.01

    # adjust the probabilities based on the scaling factor and the maximum corruption probability threshold
    scaled_tag_probabilities = {}
    for tagName, probability in tagName_probabilities.items():
        scaled_probability = min(probability / scaling_factor, max_corruption_prob_threshold)
        # scaled_probability = max(scaled_probability, min_corruption_prob_threshold)
        scaled_tag_probabilities[tagName] = scaled_probability

    #sorted_probabilities = sorted(scaled_tag_probabilities.items(), key=lambda x: x[1], reverse=True)

    #print(sorted_probabilities)

    # initialize tag_datasets dictionary
    tag_datasets = {tagName: {"original": [], "corrupted": []} for tagName in scaled_tag_probabilities}

    # Step 4: Split dataset into original and corrupted subsets for each tagName
    for sample in dataset_split:
        tagName = sample["tagName"]
        scaled_probability = scaled_tag_probabilities[tagName]
        # Randomly decide whether to include the sample in original or corrupted subset
        subset = "original" if random.random() < 0.8 else "corrupted"
        tag_datasets[tagName][subset].append(sample)

    corrupted_dataset = [sample for datasets in tag_datasets.values() for sample in datasets["corrupted"]]
    number_samples_per_ne_type = {}
    for sample in corrupted_dataset:
        ne_type = sample['tagName']
        if ne_type in number_samples_per_ne_type:
            number_samples_per_ne_type[ne_type] += 1
        else:
            number_samples_per_ne_type[ne_type] = 1
    print("Number of samples per NE type in corrupted dataset:")
    print(sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True))


def mask_named_entities(dataset_split, corruption_prob=0.2, masking_prob=0.8, default_mask='<unk>'):
    """
    works both on MSEQA and GenQA format dataset WITH guidelines
    for Llama2-7B use <unk> as mask, for DeBERTa use [UNK]
    """

    # count how many samples (i.e. how many questions) for a ne_type exist
    # e.g. {'train': {'gene': 5, 'sports team': 4, 'norp': 3, 'disease': 9,...}
    # --> there are 5 MSEQA samples which question is about 'gene' NE type and have a empty/non-empty answer
    number_samples_per_ne_type = {}
    for sample in dataset_split:
        ne_type = sample['tagName']
        if ne_type in number_samples_per_ne_type:
            number_samples_per_ne_type[ne_type] += 1
        else:
            number_samples_per_ne_type[ne_type] = 1
    print("Number of samples per NE type:")
    print(sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True))

    # initialize tag_datasets dictionary
    tag_datasets = {tagName: {"original": [], "corrupted": []} for tagName in number_samples_per_ne_type}

    # split dataset into original and corrupted subsets for each tagName
    for sample in dataset_split:
        tagName = sample["tagName"]
        # decide whether to include the sample in original or corrupted subset
        subset = "original" if random.random() < 1 - corruption_prob else "corrupted"
        tag_datasets[tagName][subset].append(sample)

    # corrupt samples in the corrupted subset
    for tagName, datasets in tag_datasets.items():
        for sample in datasets["corrupted"]:
            # Randomly decide whether to mask or replace Named Entity
            if random.random() < masking_prob:
                mask = default_mask
                # mask tagName occurrences in "instruction" with <unk>
                sample['doc_question_pairID'] = sample['doc_question_pairID'] + ':masked'
            else:
                # replace tagName occurrences in "instruction" with another random Named Entity
                mask = random.choice(list(tag_datasets.keys()))
                mask = mask.upper()
                sample['doc_question_pairID'] = sample['doc_question_pairID'] + ':switchedNE'

            pattern = re.compile(rf'{re.escape(tagName)}', flags=re.IGNORECASE)
            if 'instruction' in sample.keys():
                # GenQA instruction dataset feature
                sample['instruction'] = pattern.sub(mask, sample['instruction'])
            else:
                # MSEQA question dataset feature
                sample['question'] = pattern.sub(mask, sample['question'])

    number_samples_per_ne_type = {}
    for tagName, datasets in tag_datasets.items():
        for sample in datasets['corrupted']:
            ne_type = sample['tagName']
            if ne_type in number_samples_per_ne_type:
                number_samples_per_ne_type[ne_type] += 1
            else:
                number_samples_per_ne_type[ne_type] = 1
    print("Number of samples per NE type in corrupted dataset:")
    print(sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True))
    print(f"A total of {sum(number_samples_per_ne_type.values())} samples will be corrupted by masking or entity replacing")

    # combine corrupted samples with original dataset
    corrupted_subset = [sample for datasets in tag_datasets.values() for sample in datasets["corrupted"]]
    original_subset = [sample for datasets in tag_datasets.values() for sample in datasets["original"]]

    original_subset.extend(corrupted_subset)
    random.shuffle(original_subset)

    new_dataset = Dataset.from_list(original_subset)

    return new_dataset


def build_dataset_MSEQA_format_with_n_samples_per_NE(n_samples_per_NE=5):
    dataset_MSEQA_format = build_dataset_MSEQA_format()
    dataset_MSEQA_format = remove_bad_ne_types(dataset_MSEQA_format)
    n_samples_per_NE_MSEQA_dataset = {split: [] for split in dataset_MSEQA_format.keys()}
    n_samples_per_NE_MSEQA_dataset['test'] = dataset_MSEQA_format['test']
    for split in dataset_MSEQA_format.keys():
        if split != 'test':
            ne_list = {}
            for sample in dataset_MSEQA_format[split]:
                ne_type = sample['tagName']
                if ne_type in ne_list:
                    ne_list[ne_type] += 1
                else:
                    ne_list[ne_type] = 1
            ne_list = {ne: n_samples_per_NE if max_n_samples > n_samples_per_NE else max_n_samples for ne, max_n_samples in ne_list.items()}

            for sample in dataset_MSEQA_format[split]:
                if ne_list[sample['tagName']] > 0:
                    n_samples_per_NE_MSEQA_dataset[split].append(sample)
                    ne_list[sample['tagName']] -= 1

    return DatasetDict({split: Dataset.from_list(values) for split, values in n_samples_per_NE_MSEQA_dataset.items()})


def build_dataset_MSEQA_format_with_n_samples_per_NE_plus_negatives(n_samples_per_NE=5):
    dataset_MSEQA_format = build_dataset_MSEQA_format()
    dataset_MSEQA_format = remove_bad_ne_types(dataset_MSEQA_format)
    n_samples_per_NE_MSEQA_dataset = {split: [] for split in dataset_MSEQA_format.keys()}
    n_samples_per_NE_MSEQA_dataset['test'] = dataset_MSEQA_format['test']
    for split in dataset_MSEQA_format.keys():
        if split != 'test':
            ne_list = {}
            for sample in dataset_MSEQA_format[split]:
                ne_type = sample['tagName']
                if ne_type not in ne_list:
                    ne_list[ne_type] = {'yes_answer': 0, 'no_answer': 0}
                if not sample['answers']['text']:
                    ne_list[ne_type]['no_answer'] += 1
                else:
                    ne_list[ne_type]['yes_answer'] += 1

            ne_list = {ne: {'yes_answer': n_samples_per_NE if values['yes_answer'] > n_samples_per_NE else values['yes_answer'], 'no_answer': n_samples_per_NE if values['no_answer'] > n_samples_per_NE else values['no_answer']} for ne, values in ne_list.items()}

            for sample in dataset_MSEQA_format[split]:
                has_answer = 'yes_answer'
                if not sample['answers']['text']:
                    has_answer = 'no_answer'
                if ne_list[sample['tagName']][has_answer] > 0:
                    n_samples_per_NE_MSEQA_dataset[split].append(sample)
                    ne_list[sample['tagName']][has_answer] -= 1

            random.shuffle(n_samples_per_NE_MSEQA_dataset[split])

    return DatasetDict({split: Dataset.from_list(values) for split, values in n_samples_per_NE_MSEQA_dataset.items()})


def build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(n_pos_samples_per_NE, n_neg_samples_per_NE, removeTestDatasetsNEs=False, keep_only_top_tagNames=-1):
    """
    build MSEQA dataset (default is FalseDef) with N positive samples per NE and N negative samples per NE
    train fold with N + N samples per NE
    validation fold with ceil(N/4) + ceil(N/4) samples per NE
    test fold is copied unchanged
    """
    dataset_MSEQA_format = build_dataset_MSEQA_format()
    dataset_MSEQA_format = remove_bad_ne_types(dataset_MSEQA_format)

    if removeTestDatasetsNEs:
        dataset_MSEQA_format = remove_MIT_CrossNER_NEs_from_train(dataset_MSEQA_format)

    """
    number_samples_per_ne_type = {}
    for sample in dataset_MSEQA_format['train']:
        ne_type = sample['tagName']
        if ne_type in number_samples_per_ne_type:
            number_samples_per_ne_type[ne_type] += 1
        else:
            number_samples_per_ne_type[ne_type] = 1
    number_samples_per_ne_type = sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True)
    number_samples_per_ne_type_only_names = [x[0] for x in number_samples_per_ne_type]
    with open("../../../datasets/pileNER/top_391_NamedEntities.json", 'w') as f:
        json.dump(number_samples_per_ne_type_only_names, f, indent=2)
    print("DONE")
    """

    # if keep_only_top_tagNames==391 we consider it already filtered topNEs=391
    if keep_only_top_tagNames > -1 and keep_only_top_tagNames != 391:
        dataset_MSEQA_format = keep_only_top_N_tagNames(dataset_MSEQA_format, keep_only_top_tagNames)

    n_samples_per_NE_MSEQA_dataset = {split: [] for split in dataset_MSEQA_format.keys()}
    n_samples_per_NE_MSEQA_dataset['test'] = dataset_MSEQA_format['test']  # copy test fold unchanged
    for split in dataset_MSEQA_format.keys():
        # draw few samples only for train and validation
        if split != 'test':
            # count how many pos/neg samples we have per NE
            ne_list = {}
            for sample in dataset_MSEQA_format[split]:
                ne_type = sample['tagName']
                if ne_type not in ne_list:
                    ne_list[ne_type] = {'yes_answer': 0, 'no_answer': 0}
                if not sample['answers']['text']:
                    ne_list[ne_type]['no_answer'] += 1
                else:
                    ne_list[ne_type]['yes_answer'] += 1

            # if validation use 1/4 samples per NE
            if split == 'validation':
                n_pos_samples_per_NE = math.ceil(n_pos_samples_per_NE/4.0)
                n_neg_samples_per_NE = math.ceil(n_neg_samples_per_NE/4.0)
            ne_list = {ne: {'yes_answer': n_pos_samples_per_NE if values['yes_answer'] > n_pos_samples_per_NE else values['yes_answer'], 'no_answer': n_neg_samples_per_NE if values['no_answer'] > n_neg_samples_per_NE else values['no_answer']} for ne, values in ne_list.items()}

            for sample in dataset_MSEQA_format[split]:
                has_answer = 'yes_answer'
                if not sample['answers']['text']:
                    has_answer = 'no_answer'
                if ne_list[sample['tagName']][has_answer] > 0:
                    n_samples_per_NE_MSEQA_dataset[split].append(sample)
                    ne_list[sample['tagName']][has_answer] -= 1

            random.shuffle(n_samples_per_NE_MSEQA_dataset[split])

    return DatasetDict({split: Dataset.from_list(values) for split, values in n_samples_per_NE_MSEQA_dataset.items()})


def add_adversarial_negative_examples(dataset_MSEQA_format_FalseDef, path_to_adversarial_negative_examples_json):
    """ adding negative adversarial examples: 1 to train, 1 to validation"""
    with open(path_to_adversarial_negative_examples_json, 'r') as f:
        adversarial_examples_per_NE = json.load(f)
    # toadd_MSEQA_samples = []
    ne_progressiveID = 0
    for named_entity, values in adversarial_examples_per_NE.items():
        negative_sentences = values['negative_sentences']
        ne_progressiveID += 1
        if isinstance(negative_sentences, list):
            question_progressiveID = 0
            for i, sentence_explanation in enumerate(negative_sentences):
                # not always consistent key "sentence", we extract first value in each dict
                # sentence = sentence_explanation['sentence']
                sentence = list(sentence_explanation.values())[0]
                question_progressiveID += 1
                sample_MSEQA = {
                    "doc_question_pairID": str(ne_progressiveID) + ":" + str(question_progressiveID) + ":adversarial_negative",
                    "document_context": sentence,
                    "tagName": named_entity,
                    "question": f"What describes {named_entity.upper()} in the text?",
                    "answers": {'answer_start': [], 'text': []}
                }
                # if 3 negative sentences per NE, add 2 to train and 1 to validation
                if i == 0:
                    dataset_MSEQA_format_FalseDef['train'] = dataset_MSEQA_format_FalseDef['train'].add_item(sample_MSEQA)
                elif i == 1:
                    dataset_MSEQA_format_FalseDef['validation'] = dataset_MSEQA_format_FalseDef['validation'].add_item(sample_MSEQA)
        else:
            raise ValueError

    dataset_MSEQA_format_FalseDef['train'] = dataset_MSEQA_format_FalseDef['train'].shuffle()
    dataset_MSEQA_format_FalseDef['validation'] = dataset_MSEQA_format_FalseDef['validation'].shuffle()

    return dataset_MSEQA_format_FalseDef


def add_adversarial_positive_examples(dataset_GenQA_format_TrueDef, path_to_adversarial_positive_examples_json):
    """
    adding POSITIVE adversarial examples: 1 to train, 1 to validation for each NE
    for now working only on GenQA format (for MSEQA requires to compute characters starting positions)
    TrueDef since the adv examples are constructed specifically for these definitions
    """
    with open(path_to_adversarial_positive_examples_json, 'r') as f:
        adv_positive_examples_per_NE = json.load(f)

    # retrieving instructions per NE from the dataset
    instructions_per_NE = {}
    for sample in dataset_GenQA_format_TrueDef['train']:
        if sample['tagName'] not in instructions_per_NE:
            instructions_per_NE[sample['tagName']] = sample['instruction']

    ne_progressiveID = 0
    for named_entity, values in adv_positive_examples_per_NE.items():
        positive_adv_sentences = values['positive_adv_sentences']
        ne_progressiveID += 1
        if isinstance(positive_adv_sentences, list):
            question_progressiveID = 0
            for i, sentence_occurrences in enumerate(positive_adv_sentences):
                sentence = sentence_occurrences['sentence']
                question_progressiveID += 1
                sample_MSEQA = {
                    "doc_question_pairID": str(ne_progressiveID) + ":" + str(question_progressiveID) + ":adversarial_positive",
                    "input": sentence,
                    "tagName": named_entity,
                    "instruction": instructions_per_NE[named_entity],
                    "output": json.dumps(sentence_occurrences['positive occurrences'])
                }
                # if at least 2 positive sentences per NE, add 1 to train and 1 to validation
                if i == 0:
                    dataset_GenQA_format_TrueDef['train'] = dataset_GenQA_format_TrueDef['train'].add_item(sample_MSEQA)
                elif len(positive_adv_sentences) >= 2 and i == 1:
                    dataset_GenQA_format_TrueDef['validation'] = dataset_GenQA_format_TrueDef['validation'].add_item(sample_MSEQA)
        else:
            raise ValueError

    dataset_GenQA_format_TrueDef['train'] = dataset_GenQA_format_TrueDef['train'].shuffle()
    dataset_GenQA_format_TrueDef['validation'] = dataset_GenQA_format_TrueDef['validation'].shuffle()

    return dataset_GenQA_format_TrueDef


def remove_MIT_CrossNER_NEs_from_train(datasetDict_QA):
    """
    removing from train fold all NEs that are in MIT and CrossNER to have True Zero-shot setting in test
    person, location, country, organization, title, protein and all NEs with same NE but different definition are not removed
    """
    tagName_to_remove_list = ["actor", "character", "genre", "song", "year",  # MOVIE, title left
                              "dish", "restaurant",  # RESTAURANT
                              "algorithm", "field", "metric", "product", "programming language", "task", "university",  # AI
                              "award", "book", "event", "genre", "magazine",  # LITERATURE
                              "album", "award", "band", "artist", "instrument", "music genre", "genre", "song",  # MUSIC
                              "event", "political party",   # POLITICS
                              "journal", "object", "chemical compound", "chemical", "element", "enzyme", "event",  # SCIENCE
                              "company", "legal"]  # BUSTER
    tagName_to_remove_list = list(set(tagName_to_remove_list))

    datasetDict_QA['train'] = datasetDict_QA['train'].filter(lambda sample: sample['tagName'] not in tagName_to_remove_list)
    datasetDict_QA['validation'] = datasetDict_QA['validation'].filter(lambda sample: sample['tagName'] not in tagName_to_remove_list)

    return datasetDict_QA


def keep_only_top_N_tagNames(datasetDict_QA, top_N_tagNames):
    number_samples_per_ne_type = {}
    for sample in datasetDict_QA['train']:
        ne_type = sample['tagName']
        if ne_type in number_samples_per_ne_type:
            number_samples_per_ne_type[ne_type] += 1
        else:
            number_samples_per_ne_type[ne_type] = 1
    sorted_tagNames_list = [x[0] for x in sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True)]
    print(sorted_tagNames_list)
    valid_tagNames_list = sorted_tagNames_list[:top_N_tagNames]

    datasetDict_QA['train'] = datasetDict_QA['train'].filter(lambda sample: sample['tagName'] in valid_tagNames_list)
    datasetDict_QA['validation'] = datasetDict_QA['validation'].filter(lambda sample: sample['tagName'] in valid_tagNames_list)

    return datasetDict_QA


def convert_official_uniNER_eval_dataset_for_GenQA_same_instruction(dataset_name, path_to_dataset, with_definition=False, path_to_NE_guidelines_json=None):
    """
    Adapting UniNER eval datasets mit/crossNER for eval with Generative LLMs.
    Changing document_context column name to 'input' and answers to 'output'.
    Adding NE Guidelines as 'instruction'
    """

    with open(path_to_dataset, 'r') as fh:
        uniNER_eval_samples = json.load(fh)

    # we load guidelines also if with_def False to make NE mapping to canonical names (uniNER eval NEs are different)
    with open(path_to_NE_guidelines_json, 'r') as file:
        all_NEs_guidelines = json.load(file)

    # converting list to dict for fast access
    if all_NEs_guidelines and isinstance(all_NEs_guidelines, list):
        all_NEs_guidelines = {x['named_entity']: x for x in all_NEs_guidelines}

    dataset_GenQA = []  # dataset being constructed
    for uniNER_sample in uniNER_eval_samples:

        context, questions_answers_list = extract_context_quests_answers(uniNER_sample['conversations']).values()

        if len(questions_answers_list) > 1:
            raise ValueError("Expected only 1 question")

        question, ne_type, answers = questions_answers_list[0].values()

        # some uniNER NEs are different from the original NEs
        try:
            gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
            # NE name in natural languange form, e.g. ORG --> organization
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']
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
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']


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

        if with_definition:
            question = f"Your task is to extract the Named Entities of type {ne_type_in_natural_language.upper()} from an input TEXT. "
            question += "You are given a DEFINITION and some GUIDELINES.\n"
            question += "DEFINITION: " + this_ne_guidelines['Definition'] + "\nGUIDELINES: " + this_ne_guidelines['Guidelines'] + "\n"
            question += f"TEXT: "

            # adapting to reverse_INST template Prompter
            question = question.replace("from an input TEXT", "from the text chunk you have read")
            question = question.replace("Your task is to extract", "Extract")
            #question = question.replace("\nTEXT: ", "\nReturn a JSON list.")
            #question = question.replace("\nTEXT: ", "\nReturn a JSON list of strings for each occurrence you identify. Do not provide any further explanation.")
            #question = question.replace("\nTEXT: ", f"\nReturn a JSON list containing the occurrences you identify e.g. [\"{ne_type_in_natural_language.upper()}-1\", \"{ne_type_in_natural_language.upper()}-2\"]. Do not provide any further explanation or introduction to the answer.")
            #question = question.replace("\nTEXT: ", f"\nReturn a list containing the occurrences you identify e.g. [\"{ne_type_in_natural_language.upper()}-1\", \"{ne_type_in_natural_language.upper()}-2\"]. Return an empty list if no occurrences for this Named Entity type are present. Do not provide any further explanation or introduction to the answer.")
            question = question.replace("\nTEXT: ", "\nReturn a JSON list of instances of this Named Entity type. Return an empty list if no instances are present.")

        else:
            #question = f"Extract the Named Entities of type {ne_type_in_natural_language.upper()} from the text chunk you have read.\nReturn a JSON list."
            #question = f"Extract the Named Entities of type {ne_type_in_natural_language.upper()} from the text chunk you have read.\nReturn a JSON list of strings for each occurrence you identify. Do not provide any further explanation."
            #question = f"Extract the Named Entities of type {ne_type_in_natural_language.upper()} from the text chunk you have read.\n\nReturn a JSON list containing the occurrences you identify e.g. [\"{ne_type_in_natural_language.upper()}-1\", \"{ne_type_in_natural_language.upper()}-2\"]. Do not provide any further explanation or introduction to the answer."
            #question = f"Extract the Named Entities of type {ne_type_in_natural_language.upper()} from the text chunk you have read.\nReturn a list containing the occurrences you identify e.g. [\"{ne_type_in_natural_language.upper()}-1\", \"{ne_type_in_natural_language.upper()}-2\"]. Return an empty list if no occurrences for this Named Entity type are present. Do not provide any further explanation or introduction to the answer."
            question = f"Extract the Named Entities of type {ne_type_in_natural_language.upper()} from the text chunk you have read.\nReturn a JSON list of instances of this Named Entity type. Return an empty list if no instances are present."

        genQA_sample = {
            "doc_question_pairID": uniNER_sample['id'],
            "input": context,
            "tagName": ne_type,  # real_name_ne if want to mask in evaluation given tagName
            "instruction": question,
            "output": uniNER_sample['conversations'][-1]['value']
        }
        dataset_GenQA.append(genQA_sample)

    return Dataset.from_list(dataset_GenQA)


if __name__ == "__main__":

    #fullPileNER_MSEQA_FalseDef = build_dataset_MSEQA_format()
    #print(fullPileNER_MSEQA_FalseDef)

    dataset_MSEQA_format_with_n_samples_per_NE_FalseDef = build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(n_pos_samples_per_NE=20, n_neg_samples_per_NE=20, removeTestDatasetsNEs=True, keep_only_top_tagNames=391)
    print("\nMSEQA FalseDef train tagName list:")
    ne_list = {}
    for sample in dataset_MSEQA_format_with_n_samples_per_NE_FalseDef['train']:
        ne_type = sample['tagName']
        if ne_type in ne_list:
            ne_list[ne_type] += 1
        else:
            ne_list[ne_type] = 1
    print(sorted(ne_list.items(), key=lambda x: x[1], reverse=True))

    """
    DATASET_NAME = '5pos_5neg_perNE_TrueZeroShot_top50NEs'

    # N samples per NE (N positive + N negatives)
    dataset_MSEQA_format_with_n_samples_per_NE_FalseDef = build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(n_pos_samples_per_NE=5, n_neg_samples_per_NE=5, removeTestDatasetsNEs=True, keep_only_top_tagNames=50)
    dataset_MSEQA_format_with_n_samples_per_NE_FalseDef.save_to_disk(f"../../../datasets/pileNER/{DATASET_NAME}_MSEQA_FalseDef")
    # dataset_MSEQA_format_with_n_samples_per_NE_FalseDef = DatasetDict.load_from_disk('../../../datasets/pileNER/5_samples_per_NE_MSEQA_FalseDef_plus_negatives')
    print(dataset_MSEQA_format_with_n_samples_per_NE_FalseDef)

    print("\nMSEQA FalseDef train stats:")
    train_MSEQA_FalseDef_statistics = get_statistics_for_QA_dataset(dataset_MSEQA_format_with_n_samples_per_NE_FalseDef['train'], input_column_name='document_context', instruction_column_name='question', output_column_name='answers')
    for stat_name, stat_values in train_MSEQA_FalseDef_statistics.items():
        print(stat_name, stat_values)
    print("\nMSEQA FalseDef validation stats:")
    validation_MSEQA_FalseDef_statistics = get_statistics_for_QA_dataset(dataset_MSEQA_format_with_n_samples_per_NE_FalseDef['validation'], input_column_name='document_context', instruction_column_name='question', output_column_name='answers')
    for stat_name, stat_values in validation_MSEQA_FalseDef_statistics.items():
        print(stat_name, stat_values)

    # TO ADD NEGATIVE ADVERSARIAL EXAMPLES
    #dataset_MSEQA_format_with_n_samples_per_NE_FalseDef = add_adversarial_negative_examples(dataset_MSEQA_format_with_n_samples_per_NE_FalseDef, './questions/pileNER/ALL_pileNER_adv_examples.json')
    #print(dataset_MSEQA_format_with_n_samples_per_NE_FalseDef)

    print("\nMSEQA FalseDef train tagName list:")
    ne_list = {}
    for sample in dataset_MSEQA_format_with_n_samples_per_NE_FalseDef['train']:
        ne_type = sample['tagName']
        if ne_type in ne_list:
            ne_list[ne_type] += 1
        else:
            ne_list[ne_type] = 1
    print(sorted(ne_list.items(), key=lambda x: x[1], reverse=True))

    print("\nN samples on same document context:")
    doc_list = {}
    for sample in dataset_MSEQA_format_with_n_samples_per_NE_FalseDef['train']:
        docID = sample['doc_question_pairID'].split(":")[0]
        if docID in doc_list:
            doc_list[docID] += 1
        else:
            doc_list[docID] = 1
    print(sorted(doc_list.items(), key=lambda x: x[1], reverse=True))

    convert_MSEQA_dataset_to_GenQA_format(dataset_MSEQA_format_with_n_samples_per_NE_FalseDef, with_definition=False, path_to_save_to=f"../../../datasets/pileNER/{DATASET_NAME}_GenQA_FalseDef")

    dataset_MSEQA_format_with_n_samples_per_NE_TrueDef = build_dataset_MSEQA_format_with_guidelines("./questions/pileNER/all_423_NE_definitions.json", dataset_MSEQA_format_with_n_samples_per_NE_FalseDef)
    print(dataset_MSEQA_format_with_n_samples_per_NE_TrueDef)
    dataset_MSEQA_format_with_n_samples_per_NE_TrueDef.save_to_disk(f"../../../datasets/pileNER/{DATASET_NAME}_MSEQA_TrueDef")

    convert_MSEQA_dataset_to_GenQA_format(dataset_MSEQA_format_with_n_samples_per_NE_TrueDef, with_definition=True, path_to_save_to=f"../../../datasets/pileNER/{DATASET_NAME}_GenQA_TrueDef")
    
    """

    """
    # TO add ADVERSARIAL POSITIVE EXAMPLES
    
    dataset_GenQA_format_with_n_samples_per_NE_TrueDef_train = Dataset.from_json("../../../datasets/pileNER/5pos_5neg_per_NE_1NegAdv_GenQA_TrueDef/train.jsonl")
    print(dataset_GenQA_format_with_n_samples_per_NE_TrueDef_train)
    dataset_GenQA_format_with_n_samples_per_NE_TrueDef_validation = Dataset.from_json("../../../datasets/pileNER/5pos_5neg_per_NE_1NegAdv_GenQA_TrueDef/validation.jsonl")
    print(dataset_GenQA_format_with_n_samples_per_NE_TrueDef_validation)
    dataset_GenQA_format_with_n_samples_per_NE_TrueDef_test = Dataset.from_json("../../../datasets/pileNER/5pos_5neg_per_NE_1NegAdv_GenQA_TrueDef/test.jsonl")
    print(dataset_GenQA_format_with_n_samples_per_NE_TrueDef_test)

    datasetDict_GenQA_format_with_n_samples_per_NE_TrueDef = DatasetDict({
        "train": dataset_GenQA_format_with_n_samples_per_NE_TrueDef_train,
        "validation": dataset_GenQA_format_with_n_samples_per_NE_TrueDef_validation,
        "test": dataset_GenQA_format_with_n_samples_per_NE_TrueDef_test
    })

    print(datasetDict_GenQA_format_with_n_samples_per_NE_TrueDef)

    datasetDict_GenQA_format_with_n_samples_per_NE_TrueDef_PLUS_adv_pos = add_adversarial_positive_examples(datasetDict_GenQA_format_with_n_samples_per_NE_TrueDef, "./questions/pileNER/ALL_pileNER_positive_adversarial_examples.json")
    print(datasetDict_GenQA_format_with_n_samples_per_NE_TrueDef_PLUS_adv_pos)
    path_to_save_to = "../../../datasets/pileNER/5pos_5neg_1posADV_1negADV_GenQA_TrueDef"
    for split_name, dataset in datasetDict_GenQA_format_with_n_samples_per_NE_TrueDef_PLUS_adv_pos.items():
        dataset.to_json(os.path.join(path_to_save_to, split_name + '.jsonl'))
    """

    """
    pileNER_statistics = get_dataset_statistics()
    fullPileNER_tagName_list = pileNER_statistics['fullPileNER_tagName_list']
    with open(os.path.join('../../../datasets', "fullPileNER_tagName_list.json"), 'w') as f:
        json.dump(fullPileNER_tagName_list, f, indent=2)
    """

    """ 
    pileNER_MSEQA_TrueDef = DatasetDict.load_from_disk("../../../datasets/pileNER_MSEQA_format_TrueDef")
    print(pileNER_MSEQA_TrueDef)

    # for DeBERTa model use '[UNK]' as mask
    pileNER_MSEQA_TrueDef_enhanced_split = mask_named_entities(pileNER_MSEQA_TrueDef['validation'], corruption_prob=0.2, masking_prob=0.8, default_mask='[UNK]')
    pileNER_MSEQA_TrueDef_enhanced_split.to_json(os.path.join("../../../datasets/pileNER_MSEQA_TrueDef_enhanced", 'validation' + '.jsonl'))
    """

    #print("\n\n\n")
    """
    pileNER_train_GenQA_TrueDef = load_dataset("../../../datasets/pileNER_GenQA_format_TrueDef")['validation']
    print(pileNER_train_GenQA_TrueDef)

    print(pileNER_train_GenQA_TrueDef[0])

    pileNER_train_GenQA_TrueDef_enhanced = mask_named_entities(pileNER_train_GenQA_TrueDef)
    print(pileNER_train_GenQA_TrueDef_enhanced)

    pileNER_train_GenQA_TrueDef_enhanced.to_json(os.path.join("../../../datasets/pileNER_GenQA_format_TrueDef_enhanced", 'validation' + '.jsonl'))
    """

    # (to count how many samples per NE in validation with 5000 samples)
    """
    pileNER_train_GenQA_TrueDef_enhanced = load_dataset("../../../datasets/pileNER_GenQA_format_TrueDef_enhanced")

    number_samples_per_ne_type = {}
    for sample in pileNER_train_GenQA_TrueDef_enhanced['validation']:
        ne_type = sample['tagName']
        if ne_type in number_samples_per_ne_type:
            number_samples_per_ne_type[ne_type] += 1
        else:
            number_samples_per_ne_type[ne_type] = 1
    print("Number of samples per NE type:")
    print(sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True))

    number_samples_per_ne_type = {}
    for i, sample in enumerate(pileNER_train_GenQA_TrueDef_enhanced['validation']):
        ne_type = sample['tagName']
        if ne_type in number_samples_per_ne_type:
            number_samples_per_ne_type[ne_type] += 1
        else:
            number_samples_per_ne_type[ne_type] = 1
        if i >= 5000:
            break
    print("Number of samples per NE type:")
    print(sorted(number_samples_per_ne_type.items(), key=lambda x: x[1], reverse=True))

    
    """

    """
    dataset_MSEQA_format_FalseDef = build_dataset_MSEQA_format()
    dataset_MSEQA_format_FalseDef = remove_bad_ne_types(dataset_MSEQA_format_FalseDef)

    print(dataset_MSEQA_format_FalseDef)
    print(dataset_MSEQA_format_FalseDef['train'][400])
    print(dataset_MSEQA_format_FalseDef['train'][443])
    print(dataset_MSEQA_format_FalseDef['train'][432])
    print(dataset_MSEQA_format_FalseDef['train'][732])

    convert_MSEQA_dataset_to_GenQA_format(dataset_MSEQA_format_FalseDef, with_definition=False, path_to_save_to="../../../datasets/pileNER_GenQA_format_FalseDef")
    """

    """
    dataset_MSEQA_format_with_guidelines = DatasetDict.load_from_disk("../../../datasets/pileNER_MSEQA_format_with_guidelines")
    print(dataset_MSEQA_format_with_guidelines)
    print(dataset_MSEQA_format_with_guidelines['train'][400])
    print(dataset_MSEQA_format_with_guidelines['train'][443])
    print(dataset_MSEQA_format_with_guidelines['train'][432])
    print(dataset_MSEQA_format_with_guidelines['train'][732])

    convert_MSEQA_dataset_to_GenQA_format(dataset_MSEQA_format_with_guidelines, with_definition=True, path_to_save_to="../../../datasets/pileNER_GenQA_format_with_guidelines")
    """


    """

    dataset_name = 'restaurant'
    #data_path = f'../../../datasets/eval_data_UniNER/test_data/CrossNER_{dataset_name}.json'
    data_path = f'../../../datasets/eval_data_UniNER/test_data/mit-{dataset_name}.json'
    with open(data_path, 'r') as fh:
        examples = json.load(fh)

    print(type(examples))
    print(len(examples))
    print(examples[0])

    dataset_for_inference_MSEQA = convert_official_uniNER_eval_dataset_for_GenQA(dataset_name,
                                                                                     data_path,
                                                                                     with_definition=True,
                                                                                     path_to_NE_guidelines_json=f'./questions/MIT/gpt_guidelines/{dataset_name}_NE_definitions.json')
    print(dataset_for_inference_MSEQA)
    # print(len(dataset_for_inference_MSEQA))
    print(dataset_for_inference_MSEQA[0])
    print(dataset_for_inference_MSEQA[1])
    print(dataset_for_inference_MSEQA[10])

    """
    # dataset_for_inference_MSEQA.to_json('./ai_GenQA_for_inference.json')

    """
    def normalize_answer(s):
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
    #print("\n")

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
