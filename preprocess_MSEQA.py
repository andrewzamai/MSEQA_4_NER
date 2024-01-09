import torch

def tokenize_and_preprocess(examples_MSEQA_format, tokenizer, max_seq_length, doc_stride):
    # concatenate the question;document_context and tokenize (adding also tokenizer special tokens)
    # overflows will be automatically treated by using a sliding window approach with stride=doc_stride
    # questions are concatenated to the left of the document_context, so truncation="only_second" used

    # setting padding=longest, padding to the longest sequence in the batch
    # if concatenated input does not fit in max_seq_length stride approach applied
    tokenized_examples = tokenizer(
        examples_MSEQA_format["question"],
        examples_MSEQA_format["document_context"],
        truncation='only_second',  # longest_first
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # not padding here
    )

    # Since one document might produce several passages if it has a long context,
    # we need a map from passages to its corresponding doc-question sample
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The offset mappings will give us a map from token to character positions in the original context.
    # This will help us compute the start_positions and end_positions
    # and going back from token to character positions
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # in multi-span EQA for each sample we may have multiple start_positions & end_positions
    # we encode gold start/end positions through k-hot-vectors
    tokenized_examples["start_positions"] = [torch.zeros(len(offset_mapping[i]), dtype=torch.int32) for i in range(len(offset_mapping))]
    tokenized_examples["end_positions"] = [torch.zeros(len(offset_mapping[i]), dtype=torch.int32) for i in range(len(offset_mapping))]

    # which are passage tokens and which are question/special tokens
    tokenized_examples["sequence_ids"] = [[] for i in range(len(offset_mapping))]

    # in passage_id we save the doc_question_pairID that generated it to later collect back passages answers to doc level
    tokenized_examples["passage_id"] = []

    # new offset_mappings with [-1, -1] if not passage token (added to pad to MAX_SEQ_LENGTH)
    tokenized_examples["offset_mapping"] = [[] for i in range(len(offset_mapping))]

    for i, offsets in enumerate(offset_mapping):
        # giving to passageID the ID of the doc-question pair that generated it
        sample_index = sample_mapping[i]
        tokenized_examples["passage_id"].append(examples_MSEQA_format["doc_question_pairID"][sample_index])

        # Labeling impossible answers with the index of the CLS token
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # creating mask with 1 marking valid CLS and passage tokens
        # cannot infer None typer when doing torch.tensor(), so we do it before with list comprehension
        sequence_ids = [x if x == 1 else 0 for x in tokenized_examples.sequence_ids(i)]  # i is batch index
        sequence_ids[
            0] = 1  # CLS token will be used for not_answerable questions then its token must be treated as passage token
        tokenized_examples["sequence_ids"][i] = torch.tensor(sequence_ids)

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else (-1, -1))
            for k, o in enumerate(offset_mapping[i])
        ]

        # sample_index = sample_mapping[i]
        answers = examples_MSEQA_format["answers"][sample_index]
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

    # not padding here

    return {
        'input_ids': tokenized_examples['input_ids'],
        'attention_mask': tokenized_examples['attention_mask'],
        'start_positions': tokenized_examples['start_positions'],
        'end_positions': tokenized_examples['end_positions'],
        'sequence_ids': tokenized_examples['sequence_ids'],
        'passage_id': tokenized_examples['passage_id'],
        'offset_mapping': tokenized_examples['offset_mapping']
    }
