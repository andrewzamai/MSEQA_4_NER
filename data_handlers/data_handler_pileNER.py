"""
--- Pile-NER dataset gpt annotated ---
https://huggingface.co/datasets/Universal-NER/Pile-NER-type
https://universal-ner.github.io/
"""

import ast
import json
import re
import math
import random
import string
from collections import OrderedDict
from datasets import Dataset, DatasetDict, load_dataset


# given an UniversalNER conversation sample extract context (text passage) + (questions-gold_answers) list
def extract_context_quests_answers(conversation):
    # first element in the conversation list is the passage of text (context) provided by the human
    context = conversation.pop(0)
    if context["from"] == "human" and context["value"][:5] == "Text:":
        context = context["value"][6:]
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


def remove_outlier_ne_types(dataset_QA_format, min_num_samples_per_ne_type=10):
    # new dataset with removed outliers
    filtered_dataset_MSEQA_format_list = {split: [] for split in dataset_QA_format.keys()}
    for split in dataset_QA_format.keys():
        # counting number of QA samples per NE type
        ne_types = {}
        for sample in dataset_QA_format[split]:
            if sample["tagName"] not in ne_types:
                ne_types[sample["tagName"]] = 1
            else:
                ne_types[sample["tagName"]] += 1

        # filter ne_types with count > min_num_samples_per_ne_type
        filtered_items = {key: value for key, value in ne_types.items() if value >= min_num_samples_per_ne_type}
        # sorting by decreasing count value
        sorted_ordered_dict = OrderedDict(sorted(filtered_items.items(), key=lambda item: item[1], reverse=True))

        for sample in dataset_QA_format[split]:
            if sample["tagName"] in sorted_ordered_dict:
                filtered_dataset_MSEQA_format_list[split].append(sample)

        # creating smaller dataset for debugging
        """
        for split in filtered_dataset_MSEQA_format_list.keys():
            if split != 'train':
                filtered_dataset_MSEQA_format_list[split] = filtered_dataset_MSEQA_format_list[split][:200]
            else:
                filtered_dataset_MSEQA_format_list[split] = filtered_dataset_MSEQA_format_list[split][:1000]
        """
    return DatasetDict({split: Dataset.from_list(values) for split, values in filtered_dataset_MSEQA_format_list.items()})


def get_ne_types_list(dataset_MSEQA_format, min_num_samples_per_ne_type=100):
    ne_types = {}
    for split in dataset_MSEQA_format.keys():
        for sample in dataset_MSEQA_format[split]:
            if sample["tagName"] in ne_types:
                ne_types[sample["tagName"]] += len(sample['answers']['text'])
            else:
                ne_types[sample["tagName"]] = len(sample['answers']['text'])

    ne_types = [a[0] for a in sorted(ne_types.items(), key=lambda item: item[1], reverse=True) if
                a[1] >= min_num_samples_per_ne_type]

    with open("./questions/ne_types_list.json", 'w') as f:
        json.dump(ne_types, f, indent=2)

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


def generate_prompt_json(ne_type, example_sentences):
    ex_sentences_json = []
    for exsent in example_sentences:
        ex_sentences_json.append({'sentence': exsent['sentence'], 'entities in it': exsent['target_words_in_it']})

    prompt = "Given a Named Entity and some exemplary sentences in input, provide, concisely, "
    prompt += "both a definition and warnings about what should not be labelled for this Named Entity. "
    prompt += "Here is the input: 'Named Entity': \'{}\', 'examples': {}".format(ne_type, ex_sentences_json)
    prompt += " Output in JSON format: {{\"Definition\": \"\", \"Do not label\": \"\"}}. Limit response to 50 tokens."

    return prompt


def generate_structured_prompt(ne_type, example_sentences):
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
    dataset_MSEQA_format = remove_outlier_ne_types(dataset_MSEQA_format, min_num_samples_per_ne_type=100)

    new_ne_type_list = {
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




if __name__ == "__main__":
    """
    raw_dataset = load_dataset("Universal-NER/Pile-NER-type")
    print(raw_dataset)

    #print(raw_dataset['train'][0]['conversations'])
    
    context, quests_answers = extract_context_quests_answers(raw_dataset['train'][23]['conversations']).values()
    print(context)

    print("\n")
    for qa in quests_answers:
        print(qa)
        for a_s, text in zip(qa['answers']['answer_start'], qa['answers']['text']):
            print(context[a_s:a_s+15])
            print(text)
            print("---")
    """
    # uniNER_dataset_MSEQA_format = build_dataset_MSEQA_format()
    # uniNER_dataset_MSEQA_format.save_to_disk("../../../datasets/uniNER_dataset_MSEQA_format")
    uniNER_dataset_MSEQA_format = DatasetDict.load_from_disk("../../../datasets/uniNER_dataset_MSEQA_format")

    # for i in range(10):
    # print(uniNER_dataset_MSEQA_format['train'][i])

    """
    ne_types = {split: {} for split in uniNER_dataset_MSEQA_format.keys()}
    for split in uniNER_dataset_MSEQA_format.keys():
        for sample in uniNER_dataset_MSEQA_format[split]:
            if sample["tagName"] in ne_types[split]:
                ne_types[split][sample["tagName"]] += len(sample['answers']['text'])
            else:
                ne_types[split][sample["tagName"]] = len(sample['answers']['text'])

    ne_types = {split:dict(sorted(values.items(), key=lambda item: item[1], reverse=True)).keys() for split, values in ne_types.items()}
    """

    with open("./questions/ne_types_list.json", 'r') as file:
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

    with open("./questions/sentences_per_ne_type.json", 'r') as file:
        sentences_per_ne_type = json.load(file)

    #print(sentences_per_ne_type)

    ne_type = 'location'
    ex_sentences = sentences_per_ne_type[ne_type]
    prompt = generate_structured_prompt(ne_type, ex_sentences)
    print("\n")
    print(prompt)
