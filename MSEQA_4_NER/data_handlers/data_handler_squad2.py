from datasets import Dataset, DatasetDict, load_dataset


def build_dataset_MSEQA_format():
    # downloading squad2 dataset from huggingface repo (no test fold provided, we duplicate val into fold)
    squad2_dataset_raw = load_dataset('squad_v2')
    # building dataset with required MSEQA format features
    squad2_MSEQA_format = {"train": [], "validation": [], "test": []}
    for split in squad2_dataset_raw.keys():
        for sample in squad2_dataset_raw[split]:
            # in training there will be always 1 only answer or [] if non-answerable
            # but in validation multiple answers are given as gold answers
            # here we keep at most 1 to use our classic metrics_EQA_MS
            # using instead official evaluation script to test against multiple gold answers for a softer match
            if sample['answers']['text']:
                answers = {'text': [sample['answers']['text'][0]], 'answer_start': [sample['answers']['answer_start'][0]]}
            else:
                answers = {'text': [], 'answer_start': []}

            # there is a sample with a question with over 25000 tokens, skip it
            if len(sample['question']) > 1000:
                continue
            squad2_MSEQA_format[split].append(
                {"doc_question_pairID": sample['id'],  # no docID:questID but only 1 single ID
                 "document_context": sample['context'],
                 "tagName": None,  # no NE type
                 "question": sample['question'],
                 "answers": answers
                 })

    dataset_dict_MSEQA_format = DatasetDict({"train": Dataset.from_list(squad2_MSEQA_format["train"]),
                                             "validation": Dataset.from_list(squad2_MSEQA_format["validation"]),
                                             "test": Dataset.from_list(squad2_MSEQA_format["validation"])  # copying val also as test
                                             })
    return dataset_dict_MSEQA_format


if __name__ == '__main__':
    dataset_dict_MSEQA_format = build_dataset_MSEQA_format()
    print(dataset_dict_MSEQA_format)
    print(dataset_dict_MSEQA_format['train'])
    print(dataset_dict_MSEQA_format['validation'])
    print(dataset_dict_MSEQA_format['test'])

    print(dataset_dict_MSEQA_format['train'][0])
    print(dataset_dict_MSEQA_format['validation'][23])
