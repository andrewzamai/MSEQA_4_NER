__package__ = "SOTA_MODELS.GoLLIE"

import json
from collections import defaultdict
from typing import Type, Any
from datasets import Dataset

from MSEQA_4_NER.data_handlers import data_handler_BUSTER


def instantiate_class(class_name: Type, span: str) -> Any:
    """Instantiate a BUSTER Entity class with the given span."""
    return class_name(span)


def getGoldSpans(documentTokens, documentLabels):
    docMetadata = defaultdict(list)
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
                docMetadata[tagName].append(entity[1:])
                # cleaning for next entity
                entity = ''

        index = index + len(documentTokens[i]) + 1
        i += 1

    # tagName mapping to class names
    tagName_to_class_mapping = {
        "BUYING_COMPANY": BuyingCompany,
        "ACQUIRED_COMPANY": AcquiredCompany,
        "SELLING_COMPANY": SellingCompany,
        "GENERIC_CONSULTING_COMPANY": GenericConsultingCompany,
        "LEGAL_CONSULTING_COMPANY": LegalConsultingCompany,
        "ANNUAL_REVENUES": AnnualRevenues
    }

    gold_spans = []
    for tagName, this_tagName_spans in docMetadata.items():
        class_name = tagName_to_class_mapping[tagName]
        for span in this_tagName_spans:
            gold_spans.append(instantiate_class(class_name, span))

    return gold_spans

def convert_BUSTER_sample_for_GoLLIE(BUSTER_BIO_sample, prompt_template, guidelines):

    MAX_INPUT_LENGTH = 260

    def chunk_document_w_sliding_window(BUSTER_BIO_sample, window_size=300, overlap=15):
        """ splits a long BUSTER document in chunks of length=window_size, with an overlap b/t two consecutive windows of 'overlap' words """
        chunks = []
        start = 0
        end = window_size
        while start < len(BUSTER_BIO_sample['tokens']):
            chunk_tokens = BUSTER_BIO_sample['tokens'][start:end]
            chunk_labels = BUSTER_BIO_sample['labels'][start:end]
            chunks.append((chunk_tokens, chunk_labels))
            start += window_size - overlap
            end += window_size - overlap
        if len(chunks[-1][0]) < 20:
            chunks = chunks[:-1]
        return chunks

    this_document_chunks = []
    for chunk_tokens, chunk_labels in chunk_document_w_sliding_window(BUSTER_BIO_sample, window_size=MAX_INPUT_LENGTH, overlap=15):
        document_input = ' '.join(chunk_tokens)
        goldSpans = getGoldSpans(chunk_tokens, chunk_labels)
        sample_prompt = prompt_template.render(guidelines=guidelines, text=document_input, annotations=goldSpans)

        black_mode = black.Mode()
        sample_prompt_black_formatted = black.format_str(sample_prompt, mode=black_mode)

        prompt_only, _ = sample_prompt_black_formatted.split("result =")
        prompt_only = prompt_only + "result ="

        this_document_chunks.append({
                'guidelines_input_results': sample_prompt_black_formatted,
                'prompt_only': prompt_only,
                'goldSpans': str(goldSpans),
                'prediction': ""

        })

    return this_document_chunks

def convert_BUSTER_test_dataset_for_GoLLIE(BUSTER_BIO, prompt_template, guidelines):
    BUSTER_test_GoLLIE = []
    for BUSTER_BIO_sample in BUSTER_BIO['test']:
        sample_GoLLIE = convert_BUSTER_sample_for_GoLLIE(BUSTER_BIO_sample, prompt_template, guidelines)
        #BUSTER_test_GoLLIE.append(sample_GoLLIE)
        BUSTER_test_GoLLIE.extend(sample_GoLLIE)
    return Dataset.from_list(BUSTER_test_GoLLIE)


if __name__ == '__main__':

    import inspect
    import black
    from jinja2 import Template
    from BUSTER_guidelines_GoLLIE import *
    #from BUSTER_guidelines_GoLLIE_noexamples import *

    BUSTER_guidelines = [inspect.getsource(definition) for definition in ENTITY_DEFINITIONS]
    print(BUSTER_guidelines)

    with open("../../GoLLIE/templates/prompt.txt", "rt") as f:
        gollie_prompt_template = Template(f.read())

    BUSTER_BIO = data_handler_BUSTER.loadDataset('../../../datasets/BUSTER/FULL_KFOLDS/123_4_5')
    BUSTER_sample_GoLLIE = convert_BUSTER_sample_for_GoLLIE(BUSTER_BIO['test'][1], prompt_template=gollie_prompt_template, guidelines=BUSTER_guidelines)
    print(BUSTER_sample_GoLLIE)

    BUSTER_test_GoLLIE = convert_BUSTER_test_dataset_for_GoLLIE(BUSTER_BIO, prompt_template=gollie_prompt_template, guidelines=BUSTER_guidelines)
    print(BUSTER_test_GoLLIE)
    print(BUSTER_test_GoLLIE[0])

    # BUSTER_test_GoLLIE.to_json('./BUSTER_test_GoLLIE.jsonl')
    BUSTER_test_GoLLIE.to_json('./BUSTER_test_GoLLIE_maxLength260.jsonl')

    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HiTZ/GoLLIE-7B")

    tokenized_input = tokenizer.encode(formated_text)
    print(tokenized_input)
    print(len(tokenized_input))
    """

    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HiTZ/GoLLIE-7B")
    average = 0
    for i, sample in enumerate(BUSTER_test_GoLLIE):
        tokenized_input = tokenizer.encode(sample['prompt_only'])
        # print(len(tokenized_input))
        if len(tokenized_input) > 3000:
            print(len(tokenized_input))
        average += len(tokenized_input)
    print(average/754)
    """

