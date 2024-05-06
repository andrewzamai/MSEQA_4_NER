__package__ = "SFT_finetuning.evaluating"

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
from MSEQA_4_NER.data_handlers import data_handler_pileNER
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

    if datasets_cluster_name == 'crossNER':
        path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json"
    else:
        path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
    path_to_guidelines_folder = f"./src/MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"
    # load definitions also if with_def False to map NEs to their canonical names
    path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')
    # TODO: using same instruction for both w and w/o guidelines, using only 10 samples per dataset
    return data_handler.convert_official_uniNER_eval_dataset_for_GenQA_same_instruction(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines)


if __name__ == '__main__':

    HF_ACCESS_TOKEN = get_HF_access_token('./.env')

    print("Testing Prompts on CrossNER/MIT:\n")

    to_eval_on = [
        # converting from uniNER eval datasets using function inside data_handler_pileNER
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science']},
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant']},
        # {'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER']},
        # {'datasets_cluster_name': 'pileNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['pileNER']},
    ]

    partial_evaluate = False
    print(f"\npartial_evaluate: {partial_evaluate}")

    model_path_or_name = "meta-llama/Llama-2-7b-chat-hf"
    print(f"LLM model: {model_path_or_name}")

    max_new_tokens = 256
    print(f"max_new_tokens {max_new_tokens}")

    vllm_model = LLM(model=model_path_or_name, download_dir='./hf_cache_dir')

    tokenizer = vllm_model.get_tokenizer()

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])

    prompter = Prompter('reverse_INST', template_path='./src/SFT_finetuning/templates', eos_text='')

    for WITH_DEFINITION in [False, True]:

        for data in to_eval_on:

            for subdataset_name in data['subdataset_names']:

                print(f"\n\nEvaluating model {WITH_DEFINITION}_DEF on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

                cutoff_len = 768
                print(f"cutoff_len: {cutoff_len}")

                dataset_GenQA_format = load_or_build_dataset_GenQA_format(data['datasets_cluster_name'],
                                                                          subdataset_name, data['data_handler'],
                                                                          WITH_DEFINITION)
                print(dataset_GenQA_format)
                print(dataset_GenQA_format[0])
                sys.stdout.flush()

                indices_per_tagName = {}
                for i, sample in enumerate(dataset_GenQA_format):
                    tagName = sample['tagName']
                    if tagName not in indices_per_tagName:
                        indices_per_tagName[tagName] = []
                    indices_per_tagName[tagName].append(i)

                # retrieving gold answers (saved in ouput during dataset conversion from uniNER eval datatasets)
                all_gold_answers = dataset_GenQA_format['output']

                instructions = dataset_GenQA_format['instruction']
                print(instructions[0])
                sys.stdout.flush()

                inputs = dataset_GenQA_format['input']

                batch_instruction_input_pairs = [
                    (instruction,
                     truncate_input({"input": context, "instruction": instruction}, tokenizer, prompter,
                                    cutoff_len))
                    for context, instruction in zip(inputs, instructions)
                ]

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

                print("\ngold_answers")
                print(all_gold_answers[0:10])
                print("pred_answers")
                print(all_pred_answers[0:10])
                #for pred in all_pred_answers:
                    #print(pred)
                if partial_evaluate:
                    eval_result = uniNER_official_eval_script.NEREvaluator().partial_evaluate(all_pred_answers,
                                                                                              all_gold_answers)
                else:
                    eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(all_pred_answers,
                                                                                      all_gold_answers)
                precision = round(eval_result["precision"] * 100, 2)
                recall = round(eval_result["recall"] * 100, 2)
                f1 = round(eval_result["f1"] * 100, 2)
                print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(subdataset_name,
                                                                                                        precision,
                                                                                                        recall, f1))
        print("\n\n-------------------------------------------\n\n")