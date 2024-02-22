"""
Evaluating pileNER-finetuned Llama-2-7b for zero-shot NER on CrossNER/MIT datasets

- Using provided uniNER official evaluation script

- Using vLLM library for faster inference

UniNER's authors provide the crossNER/MIT test datasets already converted to QA format
./datasets/eval_data_UniNER/CrossNER_ai.json

We use convert_official_uniNER_eval_dataset_for_GenQA for:
 - replacing question with definition if with_definition=True
 - format to input expected by SFT_finetuning preprocess and tokenizer function
"""

__package__ = "SFT_finetuning.evaluating"

# use vllm_pip_container.sif
# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams

from datasets import Dataset, DatasetDict, load_dataset
import json
import sys
import os

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

    print("CrossNER/MIT ZERO-SHOT NER EVALUATIONS with UniNER official eval script:\n")

    to_eval_on = [
        # converting from uniNER eval datasets using function inside data_handler_pileNER
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science']},
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant']},
        {'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER']},
        {'datasets_cluster_name': 'pileNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['pileNER']},
    ]

    WITH_DEFINITION = False
    print(f"\nWith definition: {WITH_DEFINITION}")

    #model_path_or_name = "./merged_models/llama2_4_NER_noQuant"
    #model_path_or_name = "./merged_models/llama2_4_NER_FalseDef_mid_eval_cp"
    model_path_or_name = "./merged_models/llama2_4_NER_FalseDef"
    print(f"LLM model: {model_path_or_name}")

    # TODO: load from configs parameters
    max_new_tokens = 256
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
                cutoff_len = 1528
            print(f"cutoff_len: {cutoff_len}")

            dataset_MSEQA_format = load_or_build_dataset_GenQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], WITH_DEFINITION)

            indices_per_tagName = {}
            for i, sample in enumerate(dataset_MSEQA_format):
                tagName = sample['tagName']
                if tagName not in indices_per_tagName:
                    indices_per_tagName[tagName] = []
                indices_per_tagName[tagName].append(i)

            # retrieving gold answers (saved in ouput during dataset conversion from uniNER eval datatasets)
            all_gold_answers = dataset_MSEQA_format['output']

            instructions = dataset_MSEQA_format['instruction']
            inputs = dataset_MSEQA_format['input']

            batch_instruction_input_pairs = [
                (instruction,
                 truncate_input({"input": context, "instruction": instruction}, tokenizer, prompter, cutoff_len))
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
                support = sum(len(sublist) for sublist in this_tagName_golds)

                print("{} --> support: {}".format(tagName, support))
                precision = round(eval_result["precision"] * 100, 2)
                recall = round(eval_result["recall"] * 100, 2)
                f1 = round(eval_result["f1"] * 100, 2)
                print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, precision, recall, f1))
                print("\n ------------------------------------------------------- ")

            preds_to_save = []
            for i, sample in enumerate(dataset_MSEQA_format):
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
