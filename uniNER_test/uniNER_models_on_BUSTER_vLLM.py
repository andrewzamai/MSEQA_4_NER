"""
Evaluating universal-NER models for zero-shot NER on BUSTER dataset

- Using provided uniNER official evaluation script

- Using vLLM library for faster inference

UniNER's authors provide the crossNER/MIT test datasets already converted to QA format
./datasets/eval_data_UniNER/CrossNER_ai.json

We use convert_official_uniNER_eval_dataset_for_GenQA for:
 - replacing question with definition if with_definition=True
 - format to input expected by SFT_finetuning preprocess and tokenizer function
"""

__package__ = "uniNER_test"

import re
from collections import defaultdict

import torch
# use vllm_pip_container.sif
# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams

from datasets import Dataset, DatasetDict, load_dataset
import json
import sys
import os

# copy of uniNER official eval script from their github
from SFT_finetuning.evaluating import uniNER_official_eval_script

# my libraries
from MSEQA_4_NER.data_handlers import data_handler_pileNER, data_handler_BUSTER

from SFT_finetuning.commons.initialization import get_HF_access_token
from SFT_finetuning.commons.preprocessing import truncate_input

from .prompter import Prompter  # prompter for uniNER models


# TODO: use their Prompter
def get_prompt(context, ne_to_extract):
    # from list of words to string if needed
    if isinstance(context, list):
        context = ' '.join(context)
    if '_' in ne_to_extract:
        ne_to_extract = ne_to_extract.lower().split('_')
        ne_to_extract = ' '.join(ne_to_extract)

    prompt = f"A virtual assistant answers questions from a user based on the provided text.\nUSER: Text: {context}"
    prompt += f"\nASSISTANT: I’ve read this text.\nUSER: What describes {ne_to_extract} in the text?\nASSISTANT:"

    return prompt


def load_or_build_dataset_GenQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition):
    """
    universal-ner github provides the crossNER and MIT NER-datasets already in a conversation-QA format (eval_dataset_uniNER folder);
    here we convert the dataset to our usual features and replace "question" with the NE definition if with_definition=True
    """
    print("Loading train/validation/test Datasets in MS-EQA format...")
    print(" ...converting uniNER Datasets in GenQA format for inference")
    sys.stdout.flush()

    if datasets_cluster_name == 'pileNER':
        if with_definition:
            # already existing pileNER test fold in GenQA format
            path_to_pileNER_test_GenQA_format = './datasets/pileNER/GenQA_format_TrueDef/test.jsonl'
            return load_dataset("json", data_files=path_to_pileNER_test_GenQA_format)['train']
        else:
            path_to_pileNER_test_GenQA_format = './datasets/pileNER/GenQA_format_FalseDef/test.jsonl'
            return load_dataset("json", data_files=path_to_pileNER_test_GenQA_format)['train']

    if datasets_cluster_name == 'BUSTER':
        if not os.path.exists(f"./datasets/BUSTER/GenQA_format_{with_definition}Def"):
            if with_definition:
                path_to_BUSTER_MSEQA = f"./datasets/BUSTER/MSEQA_format_guidelines/BUSTER"
            else:
                path_to_BUSTER_MSEQA = f"./datasets/BUSTER/MSEQA_format_no_def/BUSTER"
            dataset_MSEQA_format = DatasetDict.load_from_disk(path_to_BUSTER_MSEQA)
            data_handler.convert_MSEQA_dataset_to_GenQA_format(dataset_MSEQA_format, with_definition, f"./datasets/BUSTER/GenQA_format_{with_definition}Def")
        return load_dataset("json", data_files=f"./datasets/BUSTER/GenQA_format_{with_definition}Def/test.jsonl")['train']  # since saved as single Dataset

    if datasets_cluster_name == 'crossNER':
        path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json"
    else:
        path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
    path_to_guidelines_folder = f"./src/MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"

    # load definitions also if with_def False to map NEs to their canonical names
    path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')
    return data_handler.convert_official_uniNER_eval_dataset_for_GenQA(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines)


if __name__ == '__main__':

    HF_ACCESS_TOKEN = get_HF_access_token('./.env')

    print("ZERO-SHOT NER EVALUATIONS of UniNER-paper models:\n")

    to_eval_on = [
        # converting from uniNER eval datasets using function inside data_handler_pileNER
        {'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER']},
        #{'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science']},
        #{'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant']},
        #{'datasets_cluster_name': 'pileNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['pileNER']},
    ]

    # NB: all uniNER models are trained without Definition
    WITH_DEFINITION = False
    print(f"\nWith definition: {WITH_DEFINITION}")

    #model_path_or_name = "Universal-NER/UniNER-7B-type"
    # model_path_or_name = "Universal-NER/UniNER-7B-definition"
    model_path_or_name = "Universal-NER/UniNER-7B-type-sup"
    #model_path_or_name = "Universal-NER/UniNER-7B-all"
    print(f"LLM model: {model_path_or_name}")

    max_new_tokens = 256
    print(f"max_new_tokens {max_new_tokens}")

    vllm_model = LLM(model=model_path_or_name, download_dir='./hf_cache_dir', tensor_parallel_size=1, dtype="bfloat16")

    tokenizer = vllm_model.get_tokenizer()

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])
    print(sampling_params)

    # prompter = Prompter('reverse_INST', template_path='./src/SFT_finetuning/templates', eos_text='')
    prompter = Prompter("ie_as_qa")

    for data in to_eval_on:

        for subdataset_name in data['subdataset_names']:

            print(f"\n\nEvaluating model named '{model_path_or_name.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            cutoff_len = 768  # 768
            if subdataset_name == 'BUSTER':
                cutoff_len = 768  # 1528
            print(f"cutoff_len: {cutoff_len}")

            dataset_GenQA_format = load_or_build_dataset_GenQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], WITH_DEFINITION)

            indices_per_tagName = {}
            for i, sample in enumerate(dataset_GenQA_format):
                tagName = sample['tagName']
                if tagName not in indices_per_tagName:
                    indices_per_tagName[tagName] = []
                indices_per_tagName[tagName].append(i)

            # retrieving gold answers (saved in ouput during dataset conversion from uniNER eval datatasets)
            all_gold_answers = dataset_GenQA_format['output']

            #instructions = dataset_GenQA_format['instruction']
            inputs = dataset_GenQA_format['input']

            instructions = []
            for sample in dataset_GenQA_format:
                tagName = sample['tagName']
                if '_' in tagName:
                    tagName = tagName.lower().split('_')
                    tagName = ' '.join(tagName)
                instruction = f"What describes {tagName} in the text?"
                sample['instruction'] = instruction
                instructions.append(instruction)

            if data['datasets_cluster_name'] != 'BUSTER':
                batch_instruction_input_pairs = [
                    (instruction,
                     truncate_input({"input": context, "instruction": instruction}, tokenizer, prompter, cutoff_len))
                    for context, instruction in zip(inputs, instructions)
                ]

            else:
                def chunk_document_w_sliding_window(document_input, window_size=300, overlap=15):
                    """ splits a long BUSTER document in chunks of length=window_size, with an overlap b/t two consecutive windows of 'overlap' words """
                    chunks = []
                    start = 0
                    end = window_size
                    while start < len(document_input):
                        chunk_inputs = document_input[start:end]
                        chunks.append(chunk_inputs)
                        start += window_size - overlap
                        end += window_size - overlap
                    if len(chunks[-1].split(' ')) < 20:
                        chunks = chunks[:-1]
                    return chunks


                batch_instruction_input_pairs = []
                # for each sample ID a list of indices of its chunks
                chunks_per_sample = defaultdict(list)
                chunk_id = 0
                for i, sample in enumerate(dataset_GenQA_format):
                    document_input = sample['input']
                    #instruction = sample['instruction']
                    instruction = instructions[i]
                    chunks = chunk_document_w_sliding_window(document_input, window_size=900, overlap=15)
                    for chunk_input in chunks:
                        chunks_per_sample[sample['doc_question_pairID']].append(chunk_id)
                        batch_instruction_input_pairs.append((instruction, chunk_input))
                        chunk_id += 1

                        # print(chunk_input)
                        # print("\n\n")
                        # sys.stdout.flush()
                    # print("\n\n------------------------------------------\n\n")

                sys.stdout.flush()

                print(f"Number of samples num_NE x n_chunks: {len(batch_instruction_input_pairs)}")

            prompts = [prompter.generate_prompt(instruction, input) for instruction, input in batch_instruction_input_pairs]
            print(prompts[0])

            responses = vllm_model.generate(prompts, sampling_params)

            # should be already ordered by the vLLM engine
            responses_corret_order = []
            response_set = {response.prompt: response for response in responses}
            for prompt in prompts:
                assert prompt in response_set
                responses_corret_order.append(response_set[prompt])
            responses = responses_corret_order
            all_pred_answers = [output.outputs[0].text.strip() for output in responses]

            if data['datasets_cluster_name'] == 'BUSTER':
                # aggregate predictions from chunks to document level
                all_pred_answers_aggregated = []
                # for sample_ID, chunks_indices in chunks_per_sample.items():
                for sample in dataset_GenQA_format:
                    sampleID = sample['doc_question_pairID']
                    chunks_indices = chunks_per_sample[sampleID]
                    document_level_preds = set()
                    for idx in chunks_indices:
                        this_chunk_preds = all_pred_answers[idx]
                        try:
                            this_chunk_preds = json.loads(this_chunk_preds)
                        except:
                            this_chunk_preds = []
                        for pred in this_chunk_preds:
                            document_level_preds.add(pred)
                    document_level_preds = json.dumps(list(document_level_preds))
                    all_pred_answers_aggregated.append(document_level_preds)
                all_pred_answers = all_pred_answers_aggregated

            print("\ngold_answers")
            print(all_gold_answers[0:10])
            print("pred_answers")
            print(all_pred_answers[0:10])
            eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(all_pred_answers, all_gold_answers)
            precision = round(eval_result["precision"]*100, 2)
            recall = round(eval_result["recall"]*100, 2)
            f1 = round(eval_result["f1"]*100, 2)
            print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(subdataset_name, precision, recall, f1))

            print("\nMetrics per NE category (100%):\n")
            for tagName, indices_for_this_tagName in indices_per_tagName.items():
                this_tagName_golds = [gold_ans for idx, gold_ans in enumerate(all_gold_answers) if idx in indices_for_this_tagName]
                this_tagName_preds = [pred_ans for idx, pred_ans in enumerate(all_pred_answers) if idx in indices_for_this_tagName]
                eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(this_tagName_preds, this_tagName_golds)
                # eval json dumps to list before counting support
                #support = sum(len(eval(sublist)) for sublist in this_tagName_golds)
                support = eval_result['support']

                print("{} --> support: {}".format(tagName, support))
                precision = round(eval_result["precision"] * 100, 2)
                recall = round(eval_result["recall"] * 100, 2)
                f1 = round(eval_result["f1"] * 100, 2)
                print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, precision, recall, f1))
                print("------------------------------------------------------- ")

            preds_to_save = []
            for i, sample in enumerate(dataset_GenQA_format):
                preds_to_save.append({
                    'doc_question_pairID': sample['doc_question_pairID'],
                    'tagName': sample['tagName'],
                    'gold_answers': all_gold_answers[i],
                    'pred_answers': all_pred_answers[i]
                })

            """
            path_to_save_predictions = './eval_predictions'
            with open(os.path.join(path_to_save_predictions, subdataset_name + '.json'), 'w', encoding='utf-8') as f:
                json.dump(preds_to_save, f, ensure_ascii=False, indent=2)
            """
            print("\n")

    print("\nDONE :)")
    sys.stdout.flush()
