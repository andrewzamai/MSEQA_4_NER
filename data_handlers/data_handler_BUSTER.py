"""
--- MSEQA for NER - BUSTER dataset handler ---

Dataset loading, document retrieval from ID, from BIO labeling to Multi-Span Extractive QA format,
ground truth metadata retrieval (non-positional labels) from BUSTER dataset (BIO positional labeling), etc ...

"""


from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import random
import json
import os
import re

try:
    from data_handlers.data_handler_pileNER import has_more_than_n_foreign_chars, has_too_many_newline, has_too_many_whitespaces, has_too_many_punctuations_and_digits, split_into_sentences, count_target_words
except:
    from data_handler_pileNER import has_more_than_n_foreign_chars, has_too_many_newline, has_too_many_whitespaces, has_too_many_punctuations_and_digits, split_into_sentences, count_target_words


def createDatasetsFromKFoldsPermutations(pathToKFoldDir):
    permutations = [{'train': ['FOLD_1', 'FOLD_2', 'FOLD_3'], 'validation': 'FOLD_4', 'test': 'FOLD_5', 'name': '123_4_5'},
                    {'train': ['FOLD_5', 'FOLD_1', 'FOLD_2'], 'validation': 'FOLD_3', 'test': 'FOLD_4', 'name': '512_3_4'},
                    {'train': ['FOLD_4', 'FOLD_5', 'FOLD_1'], 'validation': 'FOLD_2', 'test': 'FOLD_3', 'name': '451_2_3'},
                    {'train': ['FOLD_3', 'FOLD_4', 'FOLD_5'], 'validation': 'FOLD_1', 'test': 'FOLD_2', 'name': '345_1_2'},
                    {'train': ['FOLD_2', 'FOLD_3', 'FOLD_4'], 'validation': 'FOLD_5', 'test': 'FOLD_1', 'name': '234_5_1'}
                    ]

    for perm in permutations:
        data_files = {"train_0": os.path.join(pathToKFoldDir, perm['train'][0] + '.json'),
                      "train_1": os.path.join(pathToKFoldDir, perm['train'][1] + '.json'),
                      "train_2": os.path.join(pathToKFoldDir, perm['train'][2] + '.json'),
                      "validation": os.path.join(pathToKFoldDir, perm['validation'] + '.json'),
                      "test": os.path.join(pathToKFoldDir, perm['test'] + '.json')
                      }
        dataset_dict = load_dataset("json", data_files=data_files)
        merged_train_folds = concatenate_datasets([dataset_dict['train_0'], dataset_dict['train_1'], dataset_dict['train_2']], axis=0)
        del dataset_dict['train_0']
        del dataset_dict['train_1']
        del dataset_dict['train_2']
        dataset_dict['train'] = merged_train_folds

        print(dataset_dict.keys())
        print(len(dataset_dict['train']))
        print(len(dataset_dict['validation']))
        print(len(dataset_dict['test']))
        print("\n")

        dataset_dict.save_to_disk('./BUSTER_K_FOLDS', perm['name'] + '.json')


#  load BUSTER dataset from json files, requires as parameter path to folder where json files are
def loadDataset(pathToDir):
    data_files = {"train": os.path.join(pathToDir, "train.json"),
                  "test": os.path.join(pathToDir, "test.json"),
                  "validation": os.path.join(pathToDir, "validation.json")} 
    # , "silver": os.path.join(pathToDir, "silver.json")}
    return load_dataset("json", data_files=data_files)


# get document at position i in the split (train/validation/test/silver)
# returns "split:document_id" ad new docID, tokensList, labelsList
def getDocumentTokensLabels(raw_dataset, split, i):
    docID = split + ':' + raw_dataset[split][i]["document_id"]  # new docID
    return docID, raw_dataset[split][i]["tokens"], raw_dataset[split][i]["labels"]


# retrieves document metadata,
# but also start-end indexes (in characters count and not token count)
def getDocMetadataWithStartEndCharIndexes(documentTokens, documentLabels):
    docMetadata = dict(Parties={"BUYING_COMPANY": [], "SELLING_COMPANY": [], "ACQUIRED_COMPANY": []},
                       Advisors={"CONSULTANT": [], "LEGAL_CONSULTING_COMPANY": [], "GENERIC_CONSULTING_COMPANY": []},
                       Generic_Info={"ANNUAL_REVENUES": []})
    i = 0
    index = 0
    startIndex = index
    entity = ''  # entity being reconstructed
    while i < len(documentLabels):
        # if the token is labelled as part of an entity
        if documentLabels[i] != 'O':
            if entity == '':
                startIndex = index
            entity = entity + ' ' + documentTokens[i]  # this will add an initial space (to be removed)
            # if next label is Other or the beginning of another entity
            # or end of document, the current entity is complete
            if (i < len(documentLabels) - 1 and documentLabels[i + 1][0] in ["O", "B"]) or (i == len(documentLabels) - 1):
                # add to metadata
                tagFamily, tagName = documentLabels[i].split(".")
                # adding also if same name but will have != start-end indices
                docMetadata[tagFamily[2:]][tagName].append((entity[1:], startIndex, startIndex + len(entity[1:])))
                # cleaning for next entity
                entity = ''

        index = index + len(documentTokens[i]) + 1
        i += 1

    return docMetadata


# since folds can be later shuffled, we retrieve documents from their document_id within their fold name
def retrieveDocFromID(raw_dataset, docID):
    # split on ':' to retrieve fold name and document_id
    foldName, id = docID.split(':')
    document = filter(lambda x: x["document_id"] == id, raw_dataset[foldName])
    if document is not None:
        return list(document)[0]
    else:
        print("Document not found!")


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


# a list of questions for each class is stored in a 'questions.txt' file
# a txt with 1 single question for each class is instead stored in singleQuestionPerClass.txt
def loadQuestionsDictFromTxt(pathToTxt):
    questions = dict()
    with open(pathToTxt, "r") as f:
        newClass = True
        for line in f:
            if newClass:
                tagFamily, tagName = line.strip().split(':')
                if tagFamily not in questions.keys():
                    questions[tagFamily] = dict()
                if tagName not in questions[tagFamily].keys():
                    questions[tagFamily][tagName] = ""
                newClass = False
                continue
            if line.strip() != '---':
                questions[tagFamily][tagName] = line.strip()
            else:
                newClass = True
    return questions


# build DatasetDict object with docContext-question-goldAnswers
def build_dataset_MSEQA_format(pathToBUSTERDir, pathToQuestionsTxt):
    raw_BUSTER_dataset = loadDataset(pathToBUSTERDir)
    questions = loadQuestionsDictFromTxt(pathToQuestionsTxt)
    print(questions)

    newDataset_dict = {splitName: [] for splitName in raw_BUSTER_dataset.keys()}
    newDataset_Dataset = {splitName: None for splitName in raw_BUSTER_dataset.keys()}
    for splitName in raw_BUSTER_dataset.keys():
        for i in range(len(raw_BUSTER_dataset[splitName])):
            docID, tokens, labels = getDocumentTokensLabels(raw_BUSTER_dataset, splitName, i)
            docMetadata = getDocMetadataWithStartEndCharIndexes(tokens, labels)
            question_number = 0
            for tagFamily in questions.keys():
                for tagName in questions[tagFamily].keys():
                    question = questions[tagFamily][tagName]  # only 1 question per tagName
                    # splitName:docID:questioNumberForThatDocument
                    doc_question_pairID = docID + ':' + str(question_number)
                    question_number += 1
                    # document context 
                    context = ' '.join([str(elem) for elem in tokens])
                    # retrieving gold answers for this tagName
                    goldAnswers = docMetadata[tagFamily][tagName]
                    answers = {'answer_start': [], 'text': []}
                    for ga in goldAnswers:
                        answers['answer_start'].append(ga[1])
                        answers['text'].append(ga[0])
                    sample = {'doc_question_pairID': doc_question_pairID, 
                              'document_context': context,
                              'tagFamily': tagFamily,
                              'tagName': tagName,
                              'question': question,
                              'answers': answers
                              }
                    newDataset_dict[splitName].append(sample)
        newDataset_Dataset[splitName] = Dataset.from_list(newDataset_dict[splitName])

    return DatasetDict(newDataset_Dataset)


""" --- Generating definition+guidelines for each NE category by prompting gpt-3.5-turbo ---"""

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
            elif ne_type in []:
                sentence_to_ret = target_word_counts[i]
                break
        i += 1

    return sentence_to_ret


def get_n_sentences_per_ne_type(dataset_BIO_format, ne_types_list, n_sentences_per_ne=3):
    """ getting from training set n_sentences_per_ne as positive examples from which gpt will infer NE definition """

    # field 'real_name' contains the NE name in natural language format, e.g. BUYING_COMPANY --> BUYING COMPANY, medicalcondition --> medical condition
    # field 'Hints' may or may not contain additional hints about the NE category to be provided to GPT when infering the NE definition
    # field 'sentences' the n_sentences_per_ne as examples
    sentences_per_ne_type = {ne: {'real_name': "", 'Hints': "", 'sentences': []} for ne in ne_types_list}

    for ne_type in ne_types_list:
        # ne_type is tagFamily.tagName --> Parties.BUYING_COMPANY
        tagFamily, tagName = ne_type.split('.')
        i = 0
        while len(sentences_per_ne_type[ne_type]['sentences']) < n_sentences_per_ne and i < len(dataset_BIO_format['train']):
            doc_ID, tokens, labels = getDocumentTokensLabels(dataset_BIO_format, 'train', i)
            doc_metadata = getDocMetadataWithStartEndCharIndexes(tokens, labels)
            if doc_metadata[tagFamily][tagName]:
                occurrences_for_this_ne = doc_metadata[tagFamily][tagName]
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


def build_dataset_MSEQA_format_with_guidelines(path_to_BUSTER_dataset, path_to_ne_definitions_json):
    # datasetDict in BIO format
    dataset_BIO_format = loadDataset(path_to_BUSTER_dataset)
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
                docID, tokens, labels = getDocumentTokensLabels(dataset_BIO_format, splitName, i)
                docMetadata = getDocMetadataWithStartEndCharIndexes(tokens, labels)
                question_number = 0
                for tagFam_tagName in ne_types_list:
                    # question
                    gpt_definition = subdataset_NEs_guidelines[tagFam_tagName]['gpt_answer'].strip()
                    # print(gpt_definition.strip())
                    # gpt answer may have been truncated, ensure it ends by "} before evaluating to dict
                    if not gpt_definition.endswith("}"):
                        if not gpt_definition.endswith("\""):
                            gpt_definition += "\""
                        gpt_definition += "}"
                    # print(gpt_definition)
                    this_ne_guidelines = eval(gpt_definition)
                    # replacing ne types occurrences between single quotes to their UPPERCASE
                    tagName_in_guidelines = subdataset_NEs_guidelines[tagFam_tagName]['real_name']
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
                    tagFamily, tagName = tagFam_tagName.split('.')
                    goldAnswers = docMetadata[tagFamily][tagName]
                    answers = {'answer_start': [], 'text': []}
                    for ga in goldAnswers:
                        answers['answer_start'].append(ga[1])
                        answers['text'].append(ga[0])
                    sample = {'doc_question_pairID': doc_question_pairID,
                              'document_context': context,
                              'tagName': tagFam_tagName,
                              'question': question,
                              'answers': answers
                              }
                    newDataset_dict[splitName].append(sample)
            newDataset_Dataset[splitName] = Dataset.from_list(newDataset_dict[splitName])

    new_dataset_dict = DatasetDict(newDataset_Dataset)
    # new_dataset_dict["dataset_name"] = dataset_dict["dataset_name"]

    return new_dataset_dict


if __name__ == "__main__":

    BUSTER_BIO = loadDataset('../../../datasets/BUSTER/FULL_KFOLDS/123_4_5')
    print(BUSTER_BIO)

    ne_types_list = get_ne_categories_only(BUSTER_BIO)
    print(ne_types_list)

    #sentences_per_ne_type = get_n_sentences_per_ne_type(BUSTER_BIO, ne_types_list, n_sentences_per_ne=3)
    #print(sentences_per_ne_type)
    #with open(f"./questions/BUSTER/sentences_per_ne_type_BUSTER.json", 'w') as f:
        #json.dump(sentences_per_ne_type, f, indent=2)

    # BUSTER_MSEQA = build_dataset_MSEQA_format('../../../datasets/BUSTER/FULL_KFOLDS/123_4_5', './questions/BUSTER/BUSTER_describes.txt')
    # print(BUSTER_MSEQA)

    dataset_MSEQA_format_with_guidelines = build_dataset_MSEQA_format_with_guidelines('../../../datasets/BUSTER/FULL_KFOLDS/123_4_5', './questions/BUSTER/BUSTER_NE_definitions.json')
    print(dataset_MSEQA_format_with_guidelines)
    print(dataset_MSEQA_format_with_guidelines['train'][0])
    print(dataset_MSEQA_format_with_guidelines['train'][1])
    print(dataset_MSEQA_format_with_guidelines['train'][23])
    print(dataset_MSEQA_format_with_guidelines['train'][53])
    print(dataset_MSEQA_format_with_guidelines['train'][64])




