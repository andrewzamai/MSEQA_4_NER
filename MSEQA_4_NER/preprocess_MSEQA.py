__package__ = "MSEQA_4_NER"

import torch

def tokenize_and_preprocess(examples_MSEQA_format, tokenizer, max_seq_length, doc_stride):
    # concatenate the question;document_context and tokenize (adding also tokenizer special tokens)
    # overflows will be automatically treated by using a sliding window approach with stride=doc_stride
    # questions are concatenated to the left of the document_context, so truncation="only_second" used

    tokenized_examples = tokenizer(
        examples_MSEQA_format["question"],
        examples_MSEQA_format["document_context"],
        truncation='only_second',
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # not padding here, but when collating
    )

    # Since one document might produce several passages if it has a long context,
    # we need a map from passages to its corresponding doc-question sample
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The offset mappings will give us a map from token to character positions in the original context.
    # This will help us compute the start_positions and end_positions
    # and going back from token to character positions
    offset_mapping = tokenized_examples.pop("offset_mapping")

    num_passages = len(offset_mapping)

    # in multi-span EQA for each sample we may have multiple start_positions & end_positions
    # we encode gold start/end positions through k-hot-vectors
    tokenized_examples["start_positions"] = [torch.zeros(len(offset_mapping[i]), dtype=torch.int8) for i in range(num_passages)]
    tokenized_examples["end_positions"] = [torch.zeros(len(offset_mapping[i]), dtype=torch.int8) for i in range(num_passages)]

    # which are passage tokens and which are question/special tokens
    tokenized_examples["sequence_ids"] = [[] for i in range(num_passages)]

    # in passage_id we save the doc_question_pairID that generated it to later collect back passages answers to doc level
    tokenized_examples["passage_id"] = []

    # new offset_mappings with [-1, -1] if not passage token (added to pad to MAX_SEQ_LENGTH)
    tokenized_examples["offset_mapping"] = [[] for i in range(num_passages)]

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
        sequence_ids[0] = 1  # CLS token will be used for not_answerable questions then its token must be treated as passage token
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


def tokenize_and_preprocess_T5(examples_MSEQA_format, tokenizer, max_seq_length, doc_stride):
    # since T5 does not have CLS token (Extractive QA models use it for no-answers)
    # we prefix each question with <extra_id_0>, one of the T5 special tokens for MLM
    # <extra_id_0> question </s> context </s>
    cls_token_workaround = '<extra_id_0> '

    # concatenate the question;document_context and tokenize (adding also tokenizer special tokens)
    # overflows will be automatically treated by using a sliding window approach with stride=doc_stride
    # questions are concatenated to the left of the document_context, so truncation="only_second" used
    tokenized_examples = tokenizer(
        [cls_token_workaround + q for q in examples_MSEQA_format["question"]],  # prefixing with cls_token_workaround
        examples_MSEQA_format["document_context"],
        truncation='only_second',
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # not padding here, but when collating
    )

    # Since one document might produce several passages if it has a long context,
    # we need a map from passages to its corresponding doc-question sample
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The offset mappings will give us a map from token to character positions in the original context.
    # This will help us compute the start_positions and end_positions
    # and going back from token to character positions
    offset_mapping = tokenized_examples.pop("offset_mapping")

    num_passages = len(offset_mapping)

    # in multi-span EQA for each sample we may have multiple start_positions & end_positions
    # we encode gold start/end positions through k-hot-vectors
    # changed from dtype=torch.int32 to torch.int8
    tokenized_examples["start_positions"] = [torch.zeros(len(offset_mapping[i]), dtype=torch.int8) for i in range(num_passages)]
    tokenized_examples["end_positions"] = [torch.zeros(len(offset_mapping[i]), dtype=torch.int8) for i in range(num_passages)]

    # which are passage tokens and which are question/special tokens
    tokenized_examples["sequence_ids"] = [[] for i in range(num_passages)]

    # in passage_id we save the doc_question_pairID that generated it to later collect back passages answers to doc level
    tokenized_examples["passage_id"] = []

    # new offset_mappings with [-1, -1] if not passage token (added to pad to MAX_SEQ_LENGTH)
    tokenized_examples["offset_mapping"] = [[] for i in range(num_passages)]

    for i, offsets in enumerate(offset_mapping):
        # giving to passageID the ID of the doc-question pair that generated it
        sample_index = sample_mapping[i]
        tokenized_examples["passage_id"].append(examples_MSEQA_format["doc_question_pairID"][sample_index])

        # Labeling impossible answers with the index of the CLS token
        input_ids = tokenized_examples["input_ids"][i]
        # cls_index = input_ids.index(tokenizer.cls_token_id)
        # TODO: T5 does not have CLS index but nevertheless we use first token <extra_id_0> for no-answers
        cls_index = 0

        # creating mask with 1 marking valid CLS and passage tokens
        # cannot infer None typer when doing torch.tensor(), so we do it before with list comprehension
        sequence_ids = [x if x == 1 else 0 for x in tokenized_examples.sequence_ids(i)]  # i is batch index
        sequence_ids[0] = 1  # CLS token will be used for not_answerable questions then its token must be treated as passage token
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


def tokenize_and_preprocess_LLAMA(examples_MSEQA_format, tokenizer, max_seq_length, doc_stride):
    # TODO: set empty answer if chunck does not contain answer !!!
    # TODO: INST or something if LLAMA2
    # TODO: ... if second part of chunked text

    """ <s> ### Instruction:\n{instruction}\n\n <s> ### Input:\n{input}\n\n <s> ### Response:\n </s> """

    prompt_template = {
        "instruction": "### Instruction:\n{instruction}\n\n### Input:\n",
        "input": "{input}",
        "response": "\n\n### Response:\n"
    }

    """
    prompt_template = {
        "instruction": "### Instruction:\n{instruction}\n\n",
        "input": "### Input:\n{input}",
        "response": "\n\n### Response:\n"
    }
    """

    tokenized_instruction_context = tokenizer(
        [prompt_template['instruction'].format(instruction=x[:-len("TEXT:\n")]) for x in examples_MSEQA_format['question']],  # removing ending "TEXT:"
        [prompt_template['input'].format(input=x) for x in examples_MSEQA_format['document_context']],
        truncation='only_second',
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,  # don't care about offset mapping with a generative model
        padding=False,  # not padding here
    )

    # popping middle <s> inserted by tokenizer between instruction and context
    offset_mapping = tokenized_instruction_context.pop("offset_mapping")
    print(offset_mapping)
    for i, tok_input_ids in enumerate(tokenized_instruction_context['input_ids']):
        # finding middle <s> which corresponds to ID=1
        middle_s_index = tok_input_ids.index(tokenizer.bos_token_id, 1)
        tokenized_instruction_context['input_ids'][i].pop(middle_s_index)
        tokenized_instruction_context['attention_mask'][i].pop(middle_s_index)
        offset_mapping[i] = offset_mapping[i][middle_s_index:]

    print(offset_mapping)

    # if the passage is the second part of a chunked doc --> prefix with ...

    # tokenize answers and right-concatenate to the instruction+passage_of_text already encoded
    # sorting the text answers by ascending start positions to give the LLM a pattern: extract the occurences in the order they appear in the passage of text
    # although the evaluation metrics are order independent the NTP loss is penalizes order
    # we delete duplicate occurrences thus obtaining a SET of gold_answers
    gold_answers_with_char_start_perDoc = examples_MSEQA_format['answers']
    # working in batch
    sorted_textonly_gold_answers_wo_duplicates_perDoc = []
    for ga_w_c_s_perDoc in gold_answers_with_char_start_perDoc:
        # sort text answers by ascending start positions
        sorted_start_answers = sorted(zip(ga_w_c_s_perDoc['answer_start'], ga_w_c_s_perDoc['text']), key=lambda x: x[0])
        # retrieve only text answers
        sorted_answers_text_only = [item[1] for item in sorted_start_answers]
        # deleting any duplicate while preserving order (order within document context)
        sorted_textonly_gold_answers_wo_duplicates = list(OrderedDict.fromkeys(sorted_answers_text_only).keys())
        # converting to string and appending
        #sorted_textonly_gold_answers_wo_duplicates_perDoc.append(prompt_template['response'] + str(sorted_textonly_gold_answers_wo_duplicates))
        sorted_textonly_gold_answers_wo_duplicates_perDoc.append(sorted_textonly_gold_answers_wo_duplicates)

    print(sorted_textonly_gold_answers_wo_duplicates_perDoc)

    # total number of passages the documents have been chunked in
    num_passages = len(tokenized_instruction_context['input_ids'])

    # Since one document might produce several passages if it has a long context,
    # we need a map from passages to its corresponding doc-question sample
    sample_mapping = tokenized_instruction_context.pop("overflow_to_sample_mapping")



    # in passage_id we save the doc_question_pairID that generated it to later collect back passages answers to doc level
    tokenized_instruction_context["passage_id"] = []

    gold_answers_per_passage = []
    for i in range(num_passages):
        # giving to passageID the ID of the doc-question pair that generated it
        sample_index = sample_mapping[i]
        tokenized_instruction_context["passage_id"].append(examples_MSEQA_format["doc_question_pairID"][sample_index])

        sorted_text_only_answers_for_this_doc = sorted_textonly_gold_answers_wo_duplicates_perDoc[sample_index]
        # since the document may have been chunked in passages not all gold answers may be present in this passage of text
        # now we need to pop the gold answers that are not in this chunk of text
        # TODO: return also offsetmappings, after finging middle <s> scale offset by -len(istruction)? probably no
        # check if start, end char intervals within passage offset mapping

        passage_input_ids = tokenized_instruction_context['input_ids'][i]
        this_passage_gold_answers = []
        for i, ga_text in enumerate(sorted_text_only_answers_for_this_doc):
            ga_text_input_ids = tokenizer.encode(ga_text.strip(), add_special_tokens=False)
            #print(f"{ga_text}: {ga_text_input_ids}")
            #print([tokenizer._convert_id_to_token(x) for x in ga_text_input_ids])
            #print(passage_input_ids)

            passage_contains_ga = False
            # Iterate through input_ids using a sliding window
            max_index = len(passage_input_ids) - len(ga_text_input_ids)
            for j in range(max_index + 1):
                if passage_input_ids[j:j + len(ga_text_input_ids)] == ga_text_input_ids:
                    passage_contains_ga = True

            if passage_contains_ga:
                this_passage_gold_answers.append(ga_text)

        gold_answers_per_passage.append(this_passage_gold_answers)

        """   
        tokenized_answers = tokenizer(
            this_passage_gold_answers,
            truncation=True,
            max_length=max_seq_length,
            return_overflowing_tokens=True
        )

        print(tokenized_answers)

        this_passage_tokenized_gold_answers_input_ids = tokenized_answers['input_ids'][sample_index][1:]
        this_passage_tokenized_gold_answers_attention_mask = tokenized_answers['attention_mask'][sample_index][1:]

        tokenized_instruction_context['input_ids'][i].extend(this_passage_tokenized_gold_answers_input_ids)
        tokenized_instruction_context['attention_mask'][i].extend(this_passage_tokenized_gold_answers_attention_mask)

        tokenized_instruction_context['input_ids'][i].append(tokenizer.eos_token_id)
        tokenized_instruction_context['attention_mask'][i].append(1)
        """
    print(gold_answers_per_passage)

    # not padding here
    # TODO: add also "labels" key?

    # TODO: add ... if chuncked text
    # TODO: empty answer if chunck does not contain answer!!!
    return {
        'input_ids': tokenized_instruction_context['input_ids'],
        'attention_mask': tokenized_instruction_context['attention_mask'],
        'passage_id': tokenized_instruction_context['passage_id']
    }


if __name__ == '__main__':

    from datasets import DatasetDict, Dataset
    # dataset_MSEQA_format_with_guidelines = DatasetDict.load_from_disk("../../datasets/dataset_MSEQA_format_with_guidelines")

    from transformers import LlamaTokenizerFast
    from typing import Union
    from collections import OrderedDict
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    from transformers import AutoTokenizer

    print(tokenizer.encode("<unk>"))

    #print(tokenizer("..."))
    #print(tokenizer._convert_id_to_token(2023))

    print(tokenizer.encode("STDIN_FILENO"))
    print([tokenizer._convert_id_to_token(x) for x in tokenizer.encode("STDIN_FILENO")])
    print(tokenizer.encode("(STDIN_FILENO)"))
    print([tokenizer._convert_id_to_token(x) for x in tokenizer.encode("(STDIN_FILENO)")])

    print(29922, 949, 29898, 1254, 29928, 1177, 29918, 7724, 6632)
    print([tokenizer._convert_id_to_token(x) for x in [29922, 949, 29898, 1254, 29928, 1177, 29918, 7724, 6632]])

    print(tokenizer("affection"))
    print(tokenizer.encode("affection", add_special_tokens=False))
    print([tokenizer._convert_id_to_token(x) for x in tokenizer("affection")['input_ids']])
    print(tokenizer(" affection"))
    print(tokenizer.encode(" affection", add_special_tokens=False))
    print([tokenizer._convert_id_to_token(x) for x in tokenizer(" affection")['input_ids']])
    print(tokenizer("affection "))
    print([tokenizer._convert_id_to_token(x) for x in tokenizer("affection ")['input_ids']])
    print(tokenizer("ciao affection ciao"))
    print([tokenizer._convert_id_to_token(x) for x in tokenizer("ciao affection ciao")['input_ids']])

    """
    prompt_template = {
        "instruction": "### Instruction:\n{instruction}\n\n",
        "input": "### Input:\n{input}\n\n",
        "response": "### Response:\n"
    }

    for i in range(1):
        MSEQA_sample = dataset_MSEQA_format_with_guidelines['train'][i]
        #print(MSEQA_sample)

        answers = MSEQA_sample['answers']
        print(answers)

        # sort text answers by ascending start positions
        sorted_start_answers = sorted(zip(answers['answer_start'], answers['text']), key=lambda x: x[0])
        sorted_answers_text_only = [item[1] for item in sorted_start_answers]
        # deleting any duplicate while preserving order (order within document context)
        sorted_textonly_gold_answers_wo_duplicates = list(OrderedDict.fromkeys(sorted_answers_text_only).keys())
        print(sorted_textonly_gold_answers_wo_duplicates)

    #llama_sample = generate_prompt(MSEQA_sample['question'], MSEQA_sample['document_context'])
    #print(llama_sample)


    tokenized_examples = tokenizer(
        prompt_template['instruction'].format(instruction=MSEQA_sample['question'][:-len("TEXT:\n")]),
        prompt_template['input'].format(input=MSEQA_sample['document_context']),
        truncation='only_second',  # longest_first
        max_length=380, # TODO: leave space for response
        stride=50,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # not padding here
    )

    print(type(tokenizer))

    print(len(tokenized_examples['input_ids']))
    for i in range(len(tokenized_examples['input_ids'])):
        print(tokenized_examples['input_ids'][i])
        tokens = [tokenizer._convert_id_to_token(x) for x in tokenized_examples['input_ids'][i]]
        print(tokens)
        print(tokenized_examples['offset_mapping'][i])
        print(tokenized_examples.sequence_ids(i))

        print("\n")

    # TODO: convert list to string before encoding
    # add attention mask 111 when concatenating to right of input text
    tokenized_gold_answers = tokenizer.encode(str(sorted_textonly_gold_answers_wo_duplicates)) # TODO: convert list to string

    print(tokenized_gold_answers)
    tokens = [tokenizer._convert_id_to_token(x) for x in tokenized_gold_answers]
    print(tokens)

    print("\n")
    print([tokenizer.all_special_tokens])
    
    """

    small_train_MSEQA = Dataset.from_dict(dataset_MSEQA_format_with_guidelines['train'][0:10])
    print(small_train_MSEQA)
    #for sample in small_train_MSEQA:
    #print(sample)

    small_train_MSEQA_tokenized = small_train_MSEQA.map(
        lambda examples_batch: tokenize_and_preprocess_LLAMA(examples_batch, tokenizer, max_seq_length=380, doc_stride=50),
        batched=True,
        remove_columns=small_train_MSEQA.column_names,
    )

    print(small_train_MSEQA_tokenized)

    print(small_train_MSEQA_tokenized[1])
    print([tokenizer._convert_id_to_token(x) for x in small_train_MSEQA_tokenized[1]['input_ids']])

    print("\nExample of chunked/encoded and decoded back sample: \n")
    print(tokenizer.decode(small_train_MSEQA_tokenized[1]['input_ids']))

    #for sample in small_train_MSEQA_tokenized:
    #print(len(sample['input_ids']) == len(sample['attention_mask']))


    """
    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('t5-small')
    #tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')
    #tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    #question = '<extra_id_0> This is the question.'
    question = 'This is the question.'
    context = 'and this is the passage of text'

    tokenized_examples = tokenizer(
        question,
        context,
        truncation='only_second',  # longest_first
        max_length=380,
        stride=100,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,  # not padding here
    )

    print(type(tokenizer))

    print(tokenized_examples)
    print(tokenized_examples['input_ids'][0])
    tokens = [tokenizer._convert_id_to_token(x) for x in tokenized_examples['input_ids'][0]]
    print(tokens)
    print(tokenized_examples['offset_mapping'][0])



    print(tokenized_examples['input_ids'][0].index(tokenizer.cls_token_id))

    # print(tokenizer.cls_token_id)  # DOES not have CLS token
    tokens = [tokenizer._convert_id_to_token(x) for x in tokenized_examples['input_ids'][0]]
    print(tokens)
    print(tokenized_examples.sequence_ids(0))

    print([tokenizer.all_special_tokens])
    #print(tokenizer.get_sentinel_token_ids())
    #print([tokenizer._convert_id_to_token(x) for x in tokenizer.get_sentinel_token_ids()])


    """

    print("\n\nIDs to token:")
    print(tokenizer._convert_id_to_token(5159)) #▁[]
    print(tokenizer._convert_id_to_token(2636)) # []

