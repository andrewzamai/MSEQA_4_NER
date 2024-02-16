"""
EVALUATE DeBERTa-MS-EQA model for zero-shot NER with uniNER official evaluation script

UniNER's authors provide the crossNER/MIT test datasets already converted to QA format

We use convert_official_uniNER_eval_dataset_for_inference for:
 - replacing question with definition if with_definition=True
 - format to input expected by MSEQA preprocess and tokenizer function

"""

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from functools import partial
import transformers
import string
import json
import sys
import os
import re

# my libraries
from models.MSEQA_DebertaXXL import DebertaXXLForQuestionAnswering
from data_handlers import data_handler_pileNER
from collator_MSEQA import collate_fn_MSEQA
import inference_EQA_MS


""" --------------- UniNER official evaluation functions --------------- """

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parser(text):
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = []
        for item in items:
            if isinstance(item, list) or isinstance(item, tuple):
                item = tuple([normalize_answer(element) for element in item])
            else:
                item = normalize_answer(item)
            if item not in formatted_items:
                formatted_items.append(item)
        return formatted_items
    except Exception:
        return []

class NEREvaluator:
    def evaluate(self, preds: list, golds: list):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        for pred, gold in zip(preds, golds):
            gold_tuples = parser(gold)
            pred_tuples = parser(pred)
            for t in pred_tuples:
                if t in gold_tuples:
                    n_correct += 1
                n_pos_pred += 1
            n_pos_gold += len(gold_tuples)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': prec,
            'recall': recall,
            'f1': f1,
        }


def load_or_build_dataset_MSEQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition):
    path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json" if datasets_cluster_name == 'crossNER' else f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
    path_to_guidelines_folder = f"./MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"

    path_to_subdataset_guidelines = None
    if with_definition:
        path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')

    print("Loading train/validation/test Datasets in MS-EQA format...")
    print(" ...converting uniNER Datasets in MS-EQA format for inference")
    sys.stdout.flush()

    dataset_MSEQA_format = data_handler.convert_official_uniNER_eval_dataset_for_inference(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines)

    return dataset_MSEQA_format


if __name__ == '__main__':

    print("CrossNER/MIT ZERO-SHOT EVALUATIONS with UniNER official eval script:\n")

    to_eval_on = [
        # converting from uniNER dataset using function inside data_handler_pileNER
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
    ]

    WITH_DEFINITION = True
    print(f"With definition: {WITH_DEFINITION}")

    tokenizer_to_use = "microsoft/deberta-v2-xxlarge"

    if WITH_DEFINITION:
        path_to_model = "andrewzamai/MSEQA-DeBERTaXXL-TrueDef-D"
    else:
        path_to_model = "andrewzamai/MSEQA-DeBERTaXXL-FalseDef-0"

    print(f"Model name: {' '.join(path_to_model.split('/')[-2:])}")

    with open('./MSEQA_4_NER/experiments/.env', 'r') as file:
        api_keys = file.readlines()

    api_keys_dict = {}
    for api_key in api_keys:
        api_name, api_value = api_key.split('=')
        api_keys_dict[api_name] = api_value
    # print(api_keys_dict)

    model = DebertaXXLForQuestionAnswering.from_pretrained(path_to_model, token=api_keys_dict['AZ_HUGGINGFACE_TOKEN'], cache_dir='./hf_cache_dir')
    #model = DebertaXXLForQuestionAnswering.from_pretrained(path_to_model)

    accelerator = Accelerator(mixed_precision='bf16')
    model = accelerator.prepare(model)

    for data in to_eval_on:
        for subdataset_name in data['subdataset_names']:
            print(f"\n\nEvaluating MS-EQA model named '{path_to_model.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            MAX_SEQ_LENGTH = data['MAX_SEQ_LENGTH']
            DOC_STRIDE = data['DOC_STRIDE']
            MAX_QUERY_LENGTH = 150 if WITH_DEFINITION else 50

            MAX_ANS_LENGTH_IN_TOKENS = data['MAX_ANS_LENGTH_IN_TOKENS']

            print("\nLoading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use, cache_dir='./hf_cache_dir')

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

            # compute metrics
            datasets_cluster_name = data['datasets_cluster_name']
            path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json" if datasets_cluster_name == 'crossNER' else f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
            with open(path_to_eval_dataset_uniNER, 'r') as fh:
                eval_dataset_uniNER = json.load(fh)
            golds = [example['conversations'][-1]['value'] for example in eval_dataset_uniNER]

            # sort question_on_document_predicted_answers_list with IDs same ordering in eval_dataset_uniNER
            sorted_question_on_document_predicted_answers_list = sorted(question_on_document_predicted_answers_list, key=lambda x: [d['id'] for d in eval_dataset_uniNER].index(x['doc_question_pairID']))

            ids_preds = []
            for sample_prediction in sorted_question_on_document_predicted_answers_list:
                id = sample_prediction['doc_question_pairID']
                gold_answers = sample_prediction['gold_answers']['text']
                predicted_answers_doc_level = sample_prediction['predicted_answers_doc_level']

                if not isinstance(predicted_answers_doc_level, list):
                    if predicted_answers_doc_level['text'] == '':
                        predicted_answers_doc_level = []
                else:
                    predicted_answers_doc_level = [x['text'] for x in predicted_answers_doc_level]

                ids_preds.append({
                    'id': id,
                    'gold_answers': str(gold_answers),
                    'pred_answers': predicted_answers_doc_level
                })

            with open(f"./predictions/{subdataset_name}_preds.json", 'w') as f:
                json.dump(ids_preds, f, indent=4)

            preds = [json.dumps(x['pred_answers']) for x in ids_preds]
            eval_result = NEREvaluator().evaluate(preds, golds)
            print(f"\n{subdataset_name}")
            print(f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}')
            print("\n ------------------------------------ ")

    print("\nDONE :)")
    sys.stdout.flush()
