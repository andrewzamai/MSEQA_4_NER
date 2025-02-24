"""
Evaluating pileNER-finetuned Llama-2-7B for zero-shot NER on CrossNER/MIT/BUSTER datasets

- Using provided uniNER official evaluation script

- Using vLLM library for faster inference

UniNER's authors provide the crossNER/MIT test datasets already converted to QA format
./datasets/eval_data_UniNER/CrossNER_ai.json

Importantly these provided datasets exclude MISCELLANEOUS class

We use convert_official_uniNER_eval_dataset_for_GenQA for:
 - replacing question with definition if with_definition=True
 - format to input expected by SFT_finetuning preprocess and tokenizer function
"""

__package__ = "SFT_finetuning.evaluating"

import shutil

# use vllm_pip_container.sif
# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams

from datasets import Dataset, DatasetDict, load_dataset
from collections import defaultdict
import numpy as np
import argparse
import json
import sys
import os
import re

# copy of uniNER official eval script from their github
import uniNER_official_eval_script

# my libraries
from MSEQA_4_NER.data_handlers import data_handler_pileNER, data_handler_BUSTER
from ..commons.initialization import get_HF_access_token
from ..commons.preprocessing import truncate_input
from ..commons.prompter import Prompter


def load_or_build_dataset_GenQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition):
    """
    universal-ner github provides the crossNER and MIT NER-datasets already in a conversation-QA format (eval_dataset_uniNER folder);
    here we convert the dataset to our usual features and replace "question" with the NE definition if with_definition=True
    """
    print("\nLoading train/validation/test Datasets in QA format...")
    print(" ...converting uniNER Datasets in our GenQA format for inference\n")
    sys.stdout.flush()

    if datasets_cluster_name == 'pileNER':
        if with_definition:
            # already existing pileNER test fold in GenQA format
            path_to_pileNER_test_GenQA_format = './datasets/pileNER/GenQA_format_TrueDef/test.jsonl'
            return load_dataset("json", data_files=path_to_pileNER_test_GenQA_format)['train']
        else:
            path_to_pileNER_test_GenQA_format = './datasets/pileNER/GenQA_format_FalseDef/test.jsonl'
            return load_dataset("json", data_files=path_to_pileNER_test_GenQA_format)['train']

    elif datasets_cluster_name == 'BUSTER':
        if with_definition:
            path_to_BUSTER_MSEQA = f"./datasets/BUSTER/MSEQA_format_guidelines/BUSTER"
        else:
            path_to_BUSTER_MSEQA = f"./datasets/BUSTER/MSEQA_format_no_def/BUSTER"
        dataset_MSEQA_format = DatasetDict.load_from_disk(path_to_BUSTER_MSEQA)
        # TODO: changed to same instruction
        return data_handler.convert_MSEQA_dataset_to_GenQA_format_SI(dataset_MSEQA_format, with_definition, path_to_save_to=f"./datasets/BUSTER/GenQA_format_{with_definition}Def", only_test=True)['test']

    else:
        if datasets_cluster_name == 'crossNER':
            path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json"
        else:
            path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
        path_to_guidelines_folder = f"./src/MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"
        # load definitions also if with_def False to map NEs to their canonical names
        path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')
        # TODO: changed to same instruction
        return data_handler.convert_official_uniNER_eval_dataset_for_GenQA_same_instruction(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''Llama eval parser same instructions''')
    # adding arguments
    parser.add_argument('--with_guidelines', action='store_true', help='Whether to use guidelines')
    parser.add_argument('number_NEs', type=int, help='Number of NEs')
    parser.add_argument('number_pos_samples_per_NE', type=int, help='Number of positive samples per NE')
    parser.add_argument('number_neg_samples_per_NE', type=int, help='Number of negative samples per NE')
    # parsing arguments
    args = parser.parse_args()

    HF_ACCESS_TOKEN = get_HF_access_token('./.env')

    print("CrossNER/MIT/BUSTER ZERO-SHOT NER EVALUATIONS with UniNER official eval script and datasets (w/o misc):\n")

    to_eval_on = [
        # converting from uniNER eval datasets using function inside data_handler_pileNER
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science']},
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant']},
        {'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER']},
        # {'datasets_cluster_name': 'pileNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['pileNER']},
    ]

    WITH_DEFINITION = args.with_guidelines
    print(f"\nWith definition: {WITH_DEFINITION}")

    partial_evaluate = False
    print(f"\npartial_evaluate: {partial_evaluate}")

    model_path_or_name = f"./merged_models/llama2_7B_{args.number_pos_samples_per_NE}pos_{args.number_neg_samples_per_NE}neg_perNE_top{args.number_NEs}NEs_{args.with_guidelines}Def-SI-C"
    print(f"LLM model: {model_path_or_name}")

    max_new_tokens = 128
    print(f"max_new_tokens {max_new_tokens}")

    vllm_model = LLM(model=model_path_or_name, download_dir='./hf_cache_dir')

    tokenizer = vllm_model.get_tokenizer()

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])

    """
    # beam search generation
    sampling_params = SamplingParams(
        n=1,  # number of output sequences to return for the given prompt,
        best_of=4,  # from these `best_of` sequences, the top `n` are returned, treated as the beam width when `use_beam_search` is True
        use_beam_search=True,
        early_stopping='never',  # stopping condition for beam search
        temperature=0,
        top_p=1,
        top_k=-1
    )
    """

    print(sampling_params)

    prompter = Prompter('reverse_INST', template_path='./src/SFT_finetuning/templates', eos_text='')

    for data in to_eval_on:

        for subdataset_name in data['subdataset_names']:

            print(f"\n\nEvaluating model named '{model_path_or_name.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            cutoff_len = 768  # 768
            if subdataset_name == 'BUSTER':
                cutoff_len = 768  # 1528
            print(f"cutoff_len: {cutoff_len}")

            dataset_GenQA_format = load_or_build_dataset_GenQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], WITH_DEFINITION)
            print(dataset_GenQA_format)
            print(dataset_GenQA_format[0])
            sys.stdout.flush()

            # TODO: remove after debugging
            # dataset_GenQA_format = Dataset.from_list(dataset_GenQA_format.to_list()[0:20])

            indices_per_tagName = {}
            for i, sample in enumerate(dataset_GenQA_format):
                tagName = sample['tagName']
                if tagName not in indices_per_tagName:
                    indices_per_tagName[tagName] = []
                indices_per_tagName[tagName].append(i)

            # retrieving gold answers (saved in ouput during dataset conversion from uniNER eval datatasets)
            all_gold_answers = dataset_GenQA_format['output']

            # masking tagName if necessary
            """
            instructions = []
            for sample in dataset_MSEQA_format:
                tagName = sample['tagName']
                pattern = re.compile(rf'{re.escape(tagName)}', flags=re.IGNORECASE)
                sample['instruction'] = pattern.sub('<unk>', sample['instruction'])
                instructions.append(sample['instruction'])
            """
            instructions = dataset_GenQA_format['instruction']
            print(instructions[0])
            sys.stdout.flush()

            inputs = dataset_GenQA_format['input']

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
                for sample in dataset_GenQA_format:
                    document_input = sample['input']
                    instruction = sample['instruction']
                    chunks = chunk_document_w_sliding_window(document_input, window_size=900, overlap=15)
                    for chunk_input in chunks:
                        chunks_per_sample[sample['doc_question_pairID']].append(chunk_id)
                        batch_instruction_input_pairs.append((instruction, chunk_input))
                        chunk_id += 1

                        #print(chunk_input)
                        #print("\n\n")
                        #sys.stdout.flush()
                    #print("\n\n------------------------------------------\n\n")

                sys.stdout.flush()

                print(f"Number of samples num_NE x n_chunks: {len(batch_instruction_input_pairs)}")

            prompts = [prompter.generate_prompt(instruction, input) for instruction, input in batch_instruction_input_pairs]

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
                            # add only if text prediction and not evaluates to other types e.g. dict
                            if isinstance(pred, str):
                                document_level_preds.add(pred)
                    document_level_preds = json.dumps(list(document_level_preds))
                    all_pred_answers_aggregated.append(document_level_preds)
                all_pred_answers = all_pred_answers_aggregated

            print("\ngold_answers")
            print(all_gold_answers[0:10])
            print("pred_answers")
            print(all_pred_answers[0:10])
            if partial_evaluate:
                eval_result = uniNER_official_eval_script.NEREvaluator().partial_evaluate(all_pred_answers, all_gold_answers)
            else:
                eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(all_pred_answers, all_gold_answers)
            precision = round(eval_result["precision"]*100, 2)
            recall = round(eval_result["recall"]*100, 2)
            f1 = round(eval_result["f1"]*100, 2)
            print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(subdataset_name, precision, recall, f1))

            print("\nMetrics per NE category (100%):\n")
            this_dataset_metrics = {}
            for tagName, indices_for_this_tagName in indices_per_tagName.items():
                this_tagName_golds = [gold_ans for idx, gold_ans in enumerate(all_gold_answers) if idx in indices_for_this_tagName]
                this_tagName_preds = [pred_ans for idx, pred_ans in enumerate(all_pred_answers) if idx in indices_for_this_tagName]
                if partial_evaluate:
                    eval_result = uniNER_official_eval_script.NEREvaluator().partial_evaluate(this_tagName_preds, this_tagName_golds)
                else:
                    eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(this_tagName_preds, this_tagName_golds)
                # eval json dumps to list before counting support
                # CANNOT count here support as the gold answers are not reduced to SET yet
                # support = sum(len(eval(sublist)) for sublist in this_tagName_golds)

                print("{} --> support: {}".format(tagName, eval_result['support']))
                print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(tagName, eval_result['TP'], eval_result['FN'], eval_result['FP'], -1))
                precision = round(eval_result["precision"] * 100, 2)
                recall = round(eval_result["recall"] * 100, 2)
                f1 = round(eval_result["f1"] * 100, 2)
                print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, precision, recall, f1))
                print("------------------------------------------------------- ")
                this_dataset_metrics[tagName] = {
                    'support': eval_result['support'],
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # computing MACRO scores
            this_dataset_precisions = [this_dataset_metrics[tagName]['precision'] for tagName in this_dataset_metrics]
            this_dataset_recalls = [this_dataset_metrics[tagName]['recall'] for tagName in this_dataset_metrics]
            this_dataset_f1s = [this_dataset_metrics[tagName]['f1'] for tagName in this_dataset_metrics]
            this_dataset_supports = [this_dataset_metrics[tagName]['support'] for tagName in this_dataset_metrics]
            print(
                "\n{} ==> MACRO-Precision: {:.2f} +- {:.2f}, MACRO-Recall: {:.2f} +- {:.2f}, MACRO-F1: {:.2f} +- {:.2f}".format(
                    subdataset_name,
                    np.average(this_dataset_precisions),
                    np.std(this_dataset_precisions),
                    np.average(this_dataset_recalls),
                    np.std(this_dataset_recalls),
                    np.average(this_dataset_f1s),
                    np.std(this_dataset_f1s)))

            this_dataset_supports_sum = sum(this_dataset_supports)
            this_dataset_precisions_weighted = [this_dataset_metrics[tagName]['precision'] * (this_dataset_metrics[tagName]['support']/this_dataset_supports_sum) for tagName in this_dataset_metrics]
            this_dataset_recalls_weighted = [this_dataset_metrics[tagName]['recall'] * (this_dataset_metrics[tagName]['support']/this_dataset_supports_sum) for tagName in this_dataset_metrics]
            this_dataset_f1s_weighted = [this_dataset_metrics[tagName]['f1'] * (this_dataset_metrics[tagName]['support']/this_dataset_supports_sum) for tagName in this_dataset_metrics]
            print(
                "\n{} ==> Weighted-Precision: {:.2f}, Weighted-Recall: {:.2f}, Weighted-F1: {:.2f}".format(
                    subdataset_name,
                    np.sum(this_dataset_precisions_weighted),
                    np.sum(this_dataset_recalls_weighted),
                    np.sum(this_dataset_f1s_weighted)))

            preds_to_save = []
            for i, sample in enumerate(dataset_GenQA_format):
                preds_to_save.append({
                    'doc_question_pairID': sample['doc_question_pairID'],
                    'tagName': sample['tagName'],
                    'gold_answers': all_gold_answers[i],
                    'pred_answers': all_pred_answers[i]
                })

            path_to_save_predictions = os.path.join("./predictions", model_path_or_name.split('/')[-1])
            if not os.path.exists(path_to_save_predictions):
                os.makedirs(path_to_save_predictions)
            with open(os.path.join(path_to_save_predictions, subdataset_name + '.json'), 'w', encoding='utf-8') as f:
                json.dump(preds_to_save, f, ensure_ascii=False, indent=2)
            print("\n")

    print("\nDONE :)")

    #TODO: DELETING MODEL!
    print("Assuming model is on HF, deleting model!!!")
    if 'andrewzamai' in model_path_or_name:
        model_path_or_name = os.path.join('./hf_cache_dir', 'models--andrewzamai--' + model_path_or_name.split("/")[-1])

    shutil.rmtree(model_path_or_name)

    sys.stdout.flush()
