from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
import transformers
import numpy as np
import pickle
import torch
import sys
import os

# my libraries
import data_handler_cross_NER  # to load dataset in BIO format and convert it into QA format for NER


def rename_ids(examples):
    examples['doc_question_pairID'] = strIDs_to_floatIDs[examples['doc_question_pairID'].split(':')[0]][examples['doc_question_pairID']]
    return examples


def prepare_features_for_training(examples):
    # concatenate the question;document_context and tokenize (adding also RoBERTa special tokens)
    # overflows will be automatically treated by using a sliding window approach
    # questions are concatenated to the left of the document_context

    # setting padding=longest, padding to the longest sequence in the batch
    tokenized_examples = tokenizer(
        examples["question"],
        examples["document_context"],
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # not padding here
    )

    # Since one document might produce several passages if it has a long context, we need a map from passages to its corresponding doc-question sample
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The offset mappings will give us a map from token to character positions in the original context.
    # This will help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # in multispan EQA for each sample we may have multiple start_positions & end_positions
    # therefore will not be a single int for each sample but a List[int]
    # we cannot have variable length Lists when padding, nor we want to impose a maximum number of answers per passage
    # we encode spans as multi-1-hot-vectors
    tokenized_examples["start_positions"] = [np.zeros(len(offset_mapping[i]), dtype=int) for i in range(len(offset_mapping))]
    tokenized_examples["end_positions"] = [np.zeros(len(offset_mapping[i]), dtype=int) for i in range(len(offset_mapping))]

    # which are passage tokens and which are question/special tokens
    tokenized_examples["sequence_ids"] = [[] for i in range(len(offset_mapping))]

    # in passage_id we save the doc_question_pairID that generated it to later collect back passages answers to doc level
    tokenized_examples["passage_id"] = []

    # new offset_mappings with [-1, -1] if not passage token (added to pad to MAX_SEQ_LENGTH)
    tokenized_examples["offset_mapping"] = [[] for i in range(len(offset_mapping))]

    for i, offsets in enumerate(offset_mapping):
        # giving to passageID the ID of the doc-question pair that generated it
        sample_index = sample_mapping[i]
        tokenized_examples["passage_id"].append(examples["doc_question_pairID"][sample_index])

        # Labeling impossible answers with the index of the CLS token
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)  # i is batch index
        # creating mask with 1 marking valid CLS and passage tokens
        sequence_ids = np.where(np.array(sequence_ids) == 1, sequence_ids, 0)  # 0 if 0 or None (special tokens and padding tokens to MAX_SEQ_LENGTH)
        sequence_ids[0] = 1  # CLS token will be used for not_answerable questions then its token must be treated as passage token
        tokenized_examples["sequence_ids"][i] = sequence_ids

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else (-1, -1))
            for k, o in enumerate(offset_mapping[i])
        ]

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers at document level are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"][i][cls_index] = 1
            tokenized_examples["end_positions"][i][cls_index] = 1
        else:
            atLeastOneAnswer = False
            for answer_start_char, answer_text in zip(answers["answer_start"], answers["text"]):
                # sequence_ids hides the sequence_ids modified to act as mask for question tokens and passage tokens
                # retrieve not modified back
                sequence_ids = tokenized_examples.sequence_ids(i)

                # Start/end character index of the answer in the text.
                start_char = answer_start_char
                end_char = start_char + len(answer_text)

                # moving start token index to the start of the passage
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # moving end token index to the end of the passage
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"][i][cls_index] = 1
                    tokenized_examples["end_positions"][i][cls_index] = 1
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"][i][token_start_index - 1] = 1

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"][i][token_end_index + 1] = 1

                    atLeastOneAnswer = True  # there is at least one answer in this passage

            # it may be that some doc level answer was not in the passage and triggered the CLS position to be 1
            # we set it back to 0
            if atLeastOneAnswer:
                tokenized_examples["start_positions"][i][cls_index] = 0
                tokenized_examples["end_positions"][i][cls_index] = 0

    return tokenized_examples


if __name__ == "__main__":

    # underlying LM
    pretrained_model_relying_on = "roberta-base"

    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_relying_on)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    MODEL_CONTEXT_WINDOW = tokenizer.model_max_length
    print("Model: {} has context window size of {}".format(pretrained_model_relying_on, MODEL_CONTEXT_WINDOW))

    MAX_SEQ_LENGTH = 256  # question + context + special # 128 for conll2003, 256 for others
    assert MAX_SEQ_LENGTH <= MODEL_CONTEXT_WINDOW, ("MAX SEQ LENGTH must be smallerEqual than model context window")
    MAX_QUERY_LENGTH = 48
    DOC_STRIDE = 64  # overlap between 2 consecutive passages from same document, 32 for conll2003
    assert DOC_STRIDE < (MAX_SEQ_LENGTH - MAX_QUERY_LENGTH), ("DOC_STRIDE must be smaller, otherwise parts of the doc will be skipped")

    # name of the dataset to convert to QA and tokenize
    # path_to_cross_NER_datasets = "../../datasets/CrossNER/ner_data"
    path_to_cross_NER_datasets = "./datasets/CrossNER/ner_data"
    dataset_name = "music"

    # loading dataset in BIO format
    dataset_BIO_format = data_handler_cross_NER.build_dataset_from_txt(os.path.join(path_to_cross_NER_datasets, dataset_name))

    # converting to QA for NER format (building document;question;gold_answers dataset)
    # path_to_questions = os.path.join("./cross_ner_questions_simpler/", dataset_name + ".txt")
    path_to_questions = os.path.join("./src/data_handlers/cross_ner_questions_improved/", dataset_name + ".txt")
    dataset_QA_format = data_handler_cross_NER.build_dataset_QA_format(dataset_BIO_format, path_to_questions)

    print("Dataset converted to QA format: ")
    print(dataset_QA_format)

    print("some samples:")
    print(dataset_QA_format["train"][0])
    print(dataset_QA_format["train"][1])
    print(dataset_QA_format["train"][23])

    dataset_name = dataset_QA_format.pop("dataset_name")

    # Shuffling question-document samples to not have all questions for a document grouped
    dataset_QA_format = dataset_QA_format.shuffle()

    # dict for str <--> float ID renaming
    strIDs_to_floatIDs = {splitName: {} for splitName in dataset_QA_format.keys()}
    i = 0.5
    for splitName in dataset_QA_format.keys():
        for sample in dataset_QA_format[splitName]:
            strIDs_to_floatIDs[splitName][sample['doc_question_pairID']] = i
            i += 1
    # inverting dict
    floatIDs_to_strIDs = {splitName: {} for splitName in strIDs_to_floatIDs.keys()}
    for splitName in strIDs_to_floatIDs.keys():
        floatIDs_to_strIDs[splitName] = {v: k for k, v in strIDs_to_floatIDs[splitName].items()}

    # RENAMING str IDs to Float IDs
    dataset_QA_format["train"] = dataset_QA_format["train"].map(rename_ids, batched=False)
    dataset_QA_format["validation"] = dataset_QA_format["validation"].map(rename_ids, batched=False)
    dataset_QA_format["test"] = dataset_QA_format["test"].map(rename_ids, batched=False)

    # saving dataset_doc_quest_ans
    # dirToSaveTo = "../../datasets/CrossNER_QA_format_simpler"
    dirToSaveTo = "./datasets/CrossNER_QA_format"
    #os.makedirs(os.path.join(dirToSaveTo, dataset_name + '_wrong_questions'))
    os.makedirs(os.path.join(dirToSaveTo, dataset_name + '_describes'))
    with open(os.path.join(dirToSaveTo, dataset_name + '_describes', 'dataset_doc_quest_ans.pickle'), 'wb') as handle:
        pickle.dump(dataset_QA_format, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # building tokenized datasets
    tokenized_datasets = dataset_QA_format.map(prepare_features_for_training, batched=True, remove_columns=dataset_QA_format["train"].column_names)
    print(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.shuffle()

    """
    # counting passages per question
    questions = data_handler_cross_NER.load_questions_from_txt(path_to_questions)
    answer_noanswer_per_question = {question: {"answerable": 0, "not_answerable": 0} for tn, question in questions.items()}
    for sample in tokenized_datasets['train']:
        # retrieve question from input_ids using sequence_ids to understand where question starts and ends
        passageStart = sample['sequence_ids'][1:].index(1)
        questionStart = 1
        questionEnd = passageStart - 2

        question = tokenizer.decode(sample['input_ids'][questionStart:questionEnd + 1]).split()
        # if question mark ? is not attached to last word in the formulated questions in tex
        question = ' '.join([str(elem) if str(elem)[-1] != '?' else str(elem)[:-1] for elem in question])

        if question + ' ?' in questions.keys():
            question = question + ' ?'
        else:
            question = question + '?'

        # if passage with not answerable question
        if sample['start_positions'][0] == 1 and sample['end_positions'][0] == 1:
            answer_noanswer_per_question[question]['not_answerable'] += 1
        else:
            number_of_answer_spans = np.count_nonzero(sample["start_positions"])
            answer_noanswer_per_question[question]['answerable'] += number_of_answer_spans

    print(answer_noanswer_per_question)
    """

    # saving tokenized_datasets
    with open(os.path.join(dirToSaveTo, dataset_name + '_describes', 'tokenized_datasets.pickle'), 'wb') as handle:
        pickle.dump(tokenized_datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # saving strIDs_to_floatIDs
    with open(os.path.join(dirToSaveTo, dataset_name + '_describes', 'strIDs_to_floatIDs.pickle'), 'wb') as handle:
        pickle.dump(strIDs_to_floatIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)
