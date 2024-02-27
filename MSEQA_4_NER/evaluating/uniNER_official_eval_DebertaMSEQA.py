"""

Evaluating pileNER-finetuned DeBERTa-XXL-MSEQA model for zero-shot NER on CrossNER/MIT datasets

UniNER's authors provide the crossNER/MIT test datasets already converted to QA format

We use convert_official_uniNER_eval_dataset_for_inference for:
 - replacing question with definition if with_definition=True
 - format to input expected by MSEQA preprocess and tokenizer function

"""

__package__ = "MSEQA_4_NER.evaluating"

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from functools import partial
import transformers
import torch
import json
import sys
import os

# my libraries
from ..models.MSEQA_DebertaXXL import DebertaXXLForQuestionAnswering
from ..data_handlers import data_handler_pileNER
from ..collator_MSEQA import collate_fn_MSEQA
from .. import inference_EQA_MS

from SFT_finetuning.evaluating import uniNER_official_eval_script
from SFT_finetuning.commons.initialization import get_HF_access_token

def load_or_build_dataset_MSEQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition):

    if datasets_cluster_name == 'BUSTER':
        if with_definition:
            path_to_BUSTER_MSEQA = f"./datasets/BUSTER/MSEQA_format_guidelines/BUSTER"
        else:
            path_to_BUSTER_MSEQA = f"./datasets/BUSTER/MSEQA_format_no_def/BUSTER"
        return DatasetDict.load_from_disk(path_to_BUSTER_MSEQA)

    if datasets_cluster_name == 'pileNER':
        if with_definition:
            path_to_pileNER_MSEQA = f"./datasets/pileNER/MSEQA_TrueDef"
        else:
            path_to_pileNER_MSEQA = f"./datasets/pileNER/MSEQA_FalseDef"
        return DatasetDict.load_from_disk(path_to_pileNER_MSEQA)

    path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json" if datasets_cluster_name == 'crossNER' else f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
    path_to_guidelines_folder = f"./src/MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"

    #path_to_subdataset_guidelines = None
    #if with_definition:
    path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')

    print("Loading train/validation/test Datasets in MS-EQA format...")
    print(" ...converting uniNER Datasets in MS-EQA format for inference")
    sys.stdout.flush()

    dataset_MSEQA_format = data_handler.convert_official_uniNER_eval_dataset_for_inference(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines)

    return dataset_MSEQA_format


if __name__ == '__main__':

    print("DeBERTa-XXL-MSEQA ZERO-SHOT NER EVALUATIONS on CrossNER/MIT datasets with UniNER official eval script:\n")

    to_eval_on = [
        # converting from provided uniNER eval datasets using function inside data_handler_pileNER
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
        {'datasets_cluster_name': 'BUSTER', 'data_handler': None, 'subdataset_names': ['BUSTER'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
        {'datasets_cluster_name': 'pileNER', 'data_handler': None, 'subdataset_names': ['pileNER'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10}
    ]

    accelerator = Accelerator(mixed_precision='bf16')

    HF_ACCESS_TOKEN = get_HF_access_token('./.env')

    tokenizer_to_use = "microsoft/deberta-v2-xxlarge"
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use, cache_dir='./hf_cache_dir')

    # models_TrueDef = ['andrewzamai/MSEQA-DeBERTaXXL-0', 'andrewzamai/MSEQA-DeBERTaXXL-TrueDef-A', 'andrewzamai/MSEQA-DeBERTaXXL-TrueDef-B-bis', 'andrewzamai/MSEQA-DeBERTaXXL-TrueDef-C', 'andrewzamai/MSEQA-DeBERTaXXL-TrueDef-D']
    models_TrueDef = ['andrewzamai/MSEQA-DeBERTaXXL-TrueDef-D']
    #models_FalseDef = ['andrewzamai/MSEQA-DeBERTaXXL-FalseDef-A', 'andrewzamai/MSEQA-DeBERTaXXL-FalseDef-B', 'andrewzamai/MSEQA-DeBERTaXXL-FalseDef-C-bis', 'andrewzamai/MSEQA-DeBERTaXXL-FalseDef-D', 'andrewzamai/MSEQA-DeBERTaXXL-FalseDef-0']
    models_FalseDef = ['andrewzamai/MSEQA-DeBERTaXXL-FalseDef-C-bis', 'andrewzamai/MSEQA-DeBERTaXXL-FalseDef-D', 'andrewzamai/MSEQA-DeBERTaXXL-FalseDef-0']

    WITH_DEFINITION = False
    print(f"With definition: {WITH_DEFINITION}")

    if WITH_DEFINITION:
        path_to_models = models_TrueDef
    else:
        path_to_models = models_FalseDef

    for path_to_model in path_to_models:

        print(f"Model name: {path_to_model.split('/')[-1]}")
        model = DebertaXXLForQuestionAnswering.from_pretrained(path_to_model, token=HF_ACCESS_TOKEN, cache_dir='./hf_cache_dir')
        # model = DebertaXXLForQuestionAnswering.from_pretrained(path_to_model)

        model = accelerator.prepare(model)

        for data in to_eval_on:
            for subdataset_name in data['subdataset_names']:
                print(f"\n\nEvaluating MS-EQA model named '{path_to_model.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

                datasets_cluster_name = data['datasets_cluster_name']

                MAX_SEQ_LENGTH = data['MAX_SEQ_LENGTH']
                DOC_STRIDE = data['DOC_STRIDE']
                MAX_QUERY_LENGTH = 150 if WITH_DEFINITION else 50

                MAX_ANS_LENGTH_IN_TOKENS = data['MAX_ANS_LENGTH_IN_TOKENS']

                assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                MODEL_CONTEXT_WINDOW = tokenizer.model_max_length
                print(f"Pretrained model relying on: {path_to_model} has context window size of {MODEL_CONTEXT_WINDOW}")
                assert MAX_SEQ_LENGTH <= MODEL_CONTEXT_WINDOW, "MAX SEQ LENGTH must be smallerEqual than model context window"
                print(f"MAX_SEQ_LENGTH used to chunk documents: {MAX_SEQ_LENGTH}")
                assert DOC_STRIDE < (MAX_SEQ_LENGTH - MAX_QUERY_LENGTH), "DOC_STRIDE must be smaller, otherwise parts of the doc will be skipped"
                print("DOC_STRIDE used: {}".format(DOC_STRIDE))
                sys.stdout.flush()

                ''' ------------------ PREPARING MODEL & DATA FOR EVALUATION ------------------ '''

                print("\nPREPARING MODEL and DATA FOR EVALUATION ...")
                sys.stdout.flush()

                dataset_MSEQA_format = load_or_build_dataset_MSEQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], WITH_DEFINITION)

                # dataset_MSEQA_format = DatasetDict({"test": Dataset.from_dict(dataset_MSEQA_format['test'][0:20])})

                if datasets_cluster_name == 'BUSTER':
                    EVAL_BATCH_SIZE = 4
                elif datasets_cluster_name == 'pileNER':
                    EVAL_BATCH_SIZE = 32
                else:
                    EVAL_BATCH_SIZE = 64

                print("BATCH_SIZE for evaluation: {}".format(EVAL_BATCH_SIZE))
                sys.stdout.flush()

                test_dataloader = DataLoader(
                    dataset_MSEQA_format['test'],
                    shuffle=False,
                    batch_size=EVAL_BATCH_SIZE,
                    collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
                )

                test_dataloader = accelerator.prepare(test_dataloader)

                # run inference through the model
                current_step_loss_eval, model_outputs_for_metrics = inference_EQA_MS.run_inference(model, test_dataloader).values()
                print(f"current_step_loss_eval: {current_step_loss_eval}")
                # extract answers
                question_on_document_predicted_answers_list = inference_EQA_MS.extract_answers_per_passage_from_logits(
                    max_ans_length_in_tokens=MAX_ANS_LENGTH_IN_TOKENS,
                    batch_step="test",
                    print_json_every_batch_steps=0,
                    fold_name="test",
                    tokenizer=tokenizer,
                    datasetdict_MSEQA_format=dataset_MSEQA_format,
                    model_outputs_for_metrics=model_outputs_for_metrics
                )

                # compute metrics with official uniNER eval script
                golds = None
                if datasets_cluster_name in ['crossNER', 'MIT']:
                    # load gold answers from uniNER data
                    path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json" if datasets_cluster_name == 'crossNER' else f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
                    with open(path_to_eval_dataset_uniNER, 'r') as fh:
                        eval_dataset_uniNER = json.load(fh)
                    golds = [example['conversations'][-1]['value'] for example in eval_dataset_uniNER]

                    # sort question_on_document_predicted_answers_list with IDs same ordering in eval_dataset_uniNER
                    question_on_document_predicted_answers_list = sorted(question_on_document_predicted_answers_list, key=lambda x: [d['id'] for d in eval_dataset_uniNER].index(x['doc_question_pairID']))

                # collect preds and golds from each document-question sample
                ids_preds = []
                for sample_prediction in question_on_document_predicted_answers_list:
                    id = sample_prediction['doc_question_pairID']
                    gold_answers = sample_prediction['gold_answers']['text']
                    predicted_answers_doc_level = sample_prediction['predicted_answers_doc_level']

                    # extracting only text answers
                    if not isinstance(predicted_answers_doc_level, list):
                        if predicted_answers_doc_level['text'] == '':
                            predicted_answers_doc_level = []
                    else:
                        predicted_answers_doc_level = [x['text'] for x in predicted_answers_doc_level]

                    ids_preds.append({
                        'id': id,
                        'gold_answers': gold_answers,
                        'pred_answers': predicted_answers_doc_level
                    })

                path_to_save_pred_folder = os.path.join("./predictions", 'DeBERTa-XXL-MSEQA', str(WITH_DEFINITION)+'Def', path_to_model.split('/')[-1])
                if not os.path.exists(path_to_save_pred_folder):
                    os.makedirs(path_to_save_pred_folder)
                with open(os.path.join(path_to_save_pred_folder, f"{subdataset_name}_preds.json"), 'w') as f:
                    json.dump(ids_preds, f, indent=4)

                # dumps both preds and golds as uniNER_official_eval scripts expects
                preds = [json.dumps(x['pred_answers']) for x in ids_preds]
                # extract gold answers from ids_preds, otherwise use already collected goldanswers from uniNER eval files
                if datasets_cluster_name not in ['crossNER', 'MIT']:
                    golds = [json.dumps((x['gold_answers'])) for x in ids_preds]

                path_to_save_eval_folder = os.path.join("./evals", 'DeBERTa-XXL-MSEQA', str(WITH_DEFINITION)+'Def')
                if not os.path.exists(path_to_save_eval_folder):
                    os.makedirs(path_to_save_eval_folder)
                # write in append mode
                with open(os.path.join(path_to_save_eval_folder, path_to_model.split('/')[-1] + '.txt'), "a") as eval_file:
                    eval_file.write(f"\n\nEvaluating MS-EQA model named '{path_to_model.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")
                    eval_file.write("\ngold_answers\n")
                    eval_file.write(str(golds[0:10]))
                    eval_file.write("\npred_answers\n")
                    eval_file.write(str(preds[0:10]))

                    eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(preds, golds)
                    precision = round(eval_result["precision"] * 100, 2)
                    recall = round(eval_result["recall"] * 100, 2)
                    f1 = round(eval_result["f1"] * 100, 2)
                    eval_file.write("\n\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}\n".format(subdataset_name, precision, recall, f1))

                    eval_file.write("\nMetrics per NE category (100%):\n")

                    indices_per_tagName = {}
                    for i, sample in enumerate(question_on_document_predicted_answers_list):
                        tagName = sample['tagName']
                        if tagName not in indices_per_tagName:
                            indices_per_tagName[tagName] = []
                        indices_per_tagName[tagName].append(i)

                    for tagName, indices_for_this_tagName in indices_per_tagName.items():
                        this_tagName_golds = [gold_ans for idx, gold_ans in enumerate(golds) if idx in indices_for_this_tagName]
                        this_tagName_preds = [pred_ans for idx, pred_ans in enumerate(preds) if idx in indices_for_this_tagName]
                        eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(this_tagName_preds, this_tagName_golds)
                        # eval json dumps to list before counting support
                        support = sum(len(eval(sublist)) for sublist in this_tagName_golds)

                        eval_file.write("\n{} --> support: {}\n".format(tagName, support))
                        precision = round(eval_result["precision"] * 100, 2)
                        recall = round(eval_result["recall"] * 100, 2)
                        f1 = round(eval_result["f1"] * 100, 2)
                        eval_file.write("\n{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}\n".format(tagName, precision, recall, f1))
                        eval_file.write("\n-------------------------------------------------------\n")

        del model
        torch.cuda.empty_cache()

    print("\nDONE :)")
    sys.stdout.flush()
