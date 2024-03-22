""" Data handler for converting BUSTER dataset to input format expected by GNER models https://github.com/yyDing1/GNER """

__package__ = "SOTA_MODELS.GNER"

from MSEQA_4_NER.data_handlers import data_handler_BUSTER
from datasets import Dataset


def convert_test_dataset_for_GNER_inference(BUSTER_BIO_CHUNKED):
    """ DEPRECATED: convert test dataset only from BUSTER already CHUNCKED docs --> still too long documents, they exceed model's max output length of 640 """
    label_list = data_handler_BUSTER.get_ne_categories_only_natural_language_format(BUSTER_BIO_CHUNKED)
    label_list_to_str = ', '.join(label_list)

    GNER_samples = []
    n_chunks_per_document = {}
    for BIO_sample in BUSTER_BIO_CHUNKED['test']:
        document_id = BIO_sample['document_id']
        if document_id not in n_chunks_per_document:
            n_chunks_per_document[document_id] = 1
        else:
            n_chunks_per_document[document_id] += 1

        document_input = ' '.join(BIO_sample['tokens'])

        instruction = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n\n"
        instruction += f"Use the specific entity tags: {label_list_to_str} and O.\nDataset: BUSTER.\nSentence: {document_input}."

        gold_labels_per_token = [data_handler_BUSTER.convert_tagName_in_natural_language_format(label) for label in BIO_sample['labels']]
        GNER_sample = {
            "task": "NER",
            "dataset": "BUSTER",
            "split": "test",
            "label_list": label_list,
            "negative_boundary": None,
            "instance": {
                "id": document_id,
                "subpart": n_chunks_per_document[document_id],
                "words": BIO_sample['tokens'],
                "labels": gold_labels_per_token,
                "instruction_inputs": instruction,
            },
            "prediction": ""
        }

        GNER_samples.append(GNER_sample)

    return Dataset.from_list(GNER_samples)


def split_into_sentence_chunks(tokens, labels):
    """ splits each BUSTER documents in sentences according to '.' """
    chunks = []
    current_sentence_tokens = []
    current_sentence_labels = []

    for tok, label in zip(tokens, labels):
        if tok != ".":
            current_sentence_tokens.append(tok)
            current_sentence_labels.append(label)
        else:
            chunks.append({"chunk_tokens": current_sentence_tokens, "chunk_labels": current_sentence_labels})
            current_sentence_tokens = []
            current_sentence_labels = []

    if current_sentence_tokens:
        chunks.append({"chunk_tokens": current_sentence_tokens, "chunk_labels": current_sentence_labels})

    return chunks

def convert_test_dataset_for_GNER_inference_splitting_into_sentences(BUSTER_BIO):
    """ DEPRECATED: converts test dataset only from BUSTER by splitting documents in sentences according to dot .
    --> gives too short sentences and regex would be required to not split spans like Rosetta Inc . or other exceptions
    """
    label_list = data_handler_BUSTER.get_ne_categories_only_natural_language_format(BUSTER_BIO)
    label_list_to_str = ', '.join(label_list)

    GNER_samples = []
    n_chunks_per_document = {x: 0 for x in BUSTER_BIO['test']['document_id']}
    for BIO_sample in BUSTER_BIO['test']:
        document_id = BIO_sample['document_id']
        document_tokens = BIO_sample['tokens']
        document_labels = BIO_sample['labels']

        chunks = split_into_sentence_chunks(document_tokens, document_labels)
        for chunk_id, chunk in enumerate(chunks):
            chunk_tokens = chunk['chunk_tokens']
            chunk_labels = chunk['chunk_labels']

            n_chunks_per_document[document_id] += 1

            chunk_input = ' '.join(chunk_tokens)

            instruction = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n\n"
            instruction += f"Use the specific entity tags: {label_list_to_str} and O.\nDataset: BUSTER.\nSentence: {chunk_input}."

            gold_labels_per_token = [data_handler_BUSTER.convert_tagName_in_natural_language_format(label) for label in chunk_labels]

            GNER_sample = {
                "task": "NER",
                "dataset": "BUSTER",
                "split": "test",
                "label_list": label_list,
                "negative_boundary": None,
                "instance": {
                    "id": document_id,
                    "subpart": chunk_id,
                    "words": chunk_tokens,
                    "labels": gold_labels_per_token,
                    "instruction_inputs": instruction,
                },
                "prediction": ""
            }

            GNER_samples.append(GNER_sample)

    return Dataset.from_list(GNER_samples)


def chunk_document_w_sliding_window(document_tokens, document_labels, window_size=150, overlap=30):
    """ splits a long BUSTER document in chunks of length=window_size, with an overlap b/t two consecutive windows of 'overlap' words """
    chunks = []
    start = 0
    end = window_size
    while start < len(document_tokens):
        chunk_tokens = document_tokens[start:end]
        chunk_labels = document_labels[start:end]
        chunks.append({"chunk_tokens": chunk_tokens, "chunk_labels": chunk_labels})
        start += window_size - overlap
        end += window_size - overlap

    return chunks


def convert_test_dataset_for_GNER_inference_sliding_window_chunking(BUSTER_BIO, window_size=150, overlap=15):
    """ splits a long BUSTER document in chunks of length=window_size, with an overlap b/t two consecutive windows of 'overlap' words """
    label_list = data_handler_BUSTER.get_ne_categories_only_natural_language_format(BUSTER_BIO)
    label_list_to_str = ', '.join(label_list)

    GNER_samples = []
    n_chunks_per_document = {x: 0 for x in BUSTER_BIO['test']['document_id']}
    for BIO_sample in BUSTER_BIO['test']:
        document_id = BIO_sample['document_id']
        document_tokens = BIO_sample['tokens']
        document_labels = BIO_sample['labels']

        chunks = chunk_document_w_sliding_window(document_tokens, document_labels, window_size, overlap)
        for chunk_id, chunk in enumerate(chunks):
            chunk_tokens = chunk['chunk_tokens']
            chunk_labels = chunk['chunk_labels']

            n_chunks_per_document[document_id] += 1

            chunk_input = ' '.join(chunk_tokens)

            instruction = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n\n"
            instruction += f"Use the specific entity tags: {label_list_to_str} and O.\nDataset: BUSTER.\nSentence: {chunk_input}."

            # gold_labels_per_token = [data_handler_BUSTER.convert_tagName_in_natural_language_format(label) for label in chunk_labels]
            gold_labels_per_token = []
            for label in chunk_labels:
                if label != 'O':
                    label_prefix, label_tagFamTagName = label.split('-')  # splitting B-tagFamily.TagName
                    # converting tagName in Natural language format as label_list
                    label_NL = data_handler_BUSTER.convert_tagName_in_natural_language_format(label_tagFamTagName)
                    label = label_prefix + '-' + label_NL
                    gold_labels_per_token.append(label)
                else:
                    gold_labels_per_token.append(label)

            GNER_sample = {
                "task": "NER",
                "dataset": "BUSTER",
                "split": "test",
                "label_list": label_list,
                "negative_boundary": None,
                "instance": {
                    "id": document_id,
                    "subpart": chunk_id,
                    "words": chunk_tokens,
                    "labels": gold_labels_per_token,
                    "instruction_inputs": instruction,
                },
                "prediction": ""
            }

            GNER_samples.append(GNER_sample)

    return Dataset.from_list(GNER_samples)


if __name__ == '__main__':

    BUSTER_BIO = data_handler_BUSTER.loadDataset('../../../datasets/BUSTER/FULL_KFOLDS/123_4_5')
    #BUSTER_BIO_CHUNKED = data_handler_BUSTER.loadDataset('../../../datasets/BUSTER/CHUNKED_KFOLDS/123_4_5')
    print(BUSTER_BIO)

    BUSTER_BIO_statistics = data_handler_BUSTER.get_dataset_statistics(BUSTER_BIO)
    print(BUSTER_BIO_statistics)

    #ne_types_list = data_handler_BUSTER.get_ne_categories_only_natural_language_format(BUSTER_BIO_CHUNKED)
    ne_types_list = ['generic consulting company', 'legal consulting company', 'annual revenues', 'acquired company', 'buying company', 'selling company']
    print(ne_types_list)

    #label_list_to_str = ', '.join(ne_types_list)
    #print(label_list_to_str)

    #BUSTER_GNER_test = convert_test_dataset_for_GNER_inference(BUSTER_BIO_CHUNKED)

    #BUSTER_GNER_test_sentences = convert_test_dataset_for_GNER_inference_splitting_into_sentences(BUSTER_BIO)

    BUSTER_GNER_test_sliding_window = convert_test_dataset_for_GNER_inference_sliding_window_chunking(BUSTER_BIO, window_size=100, overlap=15)

    #BUSTER_GNER_test.to_json("./BUSTER_GNER_test.jsonl")
    #BUSTER_GNER_test_sentences.to_json("./BUSTER_GNER_test_sentences.jsonl")

    #BUSTER_GNER_test_sliding_window.to_json("./BUSTER_GNER_test_sliding_window.jsonl")
    print(BUSTER_GNER_test_sliding_window)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")

    first_sample_input = BUSTER_GNER_test_sliding_window[0]['instance']['instruction_inputs']
    print(first_sample_input)

    # INSTRUCTION PROMPT is itself 173 tokens!
    tokenized_input = tokenizer.encode(first_sample_input)
    print(tokenized_input)
    print(len(tokenized_input))



