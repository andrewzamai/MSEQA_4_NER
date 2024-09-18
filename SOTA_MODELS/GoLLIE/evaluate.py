from datasets import load_dataset
import numpy as np

import GoLLIE.src.tasks.utils_typing
from GoLLIE.src.tasks.utils_typing import AnnotationList

from BUSTER_guidelines_GoLLIE import *

from typing import List, Dict

from GNER.evaluate import normalize_answer

if __name__ == '__main__':

    #BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./preds/BUSTER_GoLLIE-CodeLLaMA2-7B-pileNER391x50.jsonl')['train']
    #BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./preds/BUSTER_GoLLIE-LLaMA2-7B-chat-pileNER391x50_maxLength260.jsonl')['train']
    BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./preds/BUSTER_GoLLIE-LLaMA2-7B-chat-pileNER391x50_FULL.jsonl')['train']
    #BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./preds/BUSTER_GoLLIE-LLaMA2-7B-chat-pileNER391x50_maxLength260_noexamples_sp.jsonl')['train']
    #BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./preds/BUSTER_GoLLIE-LLaMA2-7B-chat-pileNER391x50_maxLength260_t06_sp_buying.jsonl')['train']
    print(BUSTER_test_GoLLIE)

    """
    example_goldSpans = BUSTER_test_GoLLIE[0]['goldSpans']
    print(eval(example_goldSpans))
    example_prediction = BUSTER_test_GoLLIE[0]['prediction']
    print(example_prediction)

    # faking class type hallucination
    example_prediction = example_prediction[:-4]
    example_prediction += ", AcquiredddCompany(span='Permian Basin')]"
    print(example_prediction)

    filtered_preds = AnnotationList.from_output(example_prediction, task_module='BUSTER_guidelines_GoLLIE')
    print(filtered_preds)
    """

    # WITH max_output_length=256 the number of truncated responses would be 138, with MAXOUT=460 is 129
    number_truncated_responses = 0
    all_perDoc_goldSpans = []
    all_perDoc_predSpans = []
    for sample in BUSTER_test_GoLLIE:
        goldSpans = eval(sample['goldSpans'])
        list_gold_tuples = []
        for gold in goldSpans:
            # similarly to GNER each tuple (span_text, named_entity_type)
            list_gold_tuples.append((gold.span, type(gold).__name__))
        all_perDoc_goldSpans.append(list_gold_tuples)

        raw_prediction_str = sample['prediction']
        list_pred_tuples = []
        # adjusting truncated outputs and hallucinated classes
        #if raw_prediction_str and raw_prediction_str.strip()[-1] != ']':
        if raw_prediction_str and not raw_prediction_str.endswith(']'):
            number_truncated_responses += 1
            # print(raw_prediction_str)
            raw_prediction_str = raw_prediction_str.strip()
            last_comma_index = raw_prediction_str.rfind(',')
            if last_comma_index != -1:
                raw_prediction_str = raw_prediction_str[:last_comma_index] + ']'
        print(raw_prediction_str)

        filtered_preds = AnnotationList.from_output(raw_prediction_str, task_module='BUSTER_guidelines_GoLLIE')

        for pred in filtered_preds:
            print(pred)
            #print(pred is GoLLIE.src.tasks.utils_typing.HallucinatedType)
            if pred is GoLLIE.src.tasks.utils_typing.HallucinatedType or isinstance(pred, GoLLIE.src.tasks.utils_typing.HallucinatedType):
                list_pred_tuples.append(("", "Hallucination"))
            else:
                list_pred_tuples.append((pred.span, type(pred).__name__))
        all_perDoc_predSpans.append(list_pred_tuples)

    print(f"number_truncated_responses: {number_truncated_responses}")
    print(all_perDoc_goldSpans)
    print(all_perDoc_predSpans)

    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    ne_types_list = [
        'GenericConsultingCompany',
        'LegalConsultingCompany',
        'AnnualRevenues',
        'AcquiredCompany',
        'BuyingCompany',
        'SellingCompany',
        'Hallucination'
    ]
    scores_per_NE = {ne: {"n_correct": 0, "n_pos_gold": 0, "n_pos_pred": 0} for ne in ne_types_list}

    def list_of_tuples_to_normalized_set(list_tuples):
        set_tuples = set()
        for tuple in list_tuples:
            text_span = normalize_answer(tuple[0])
            set_tuples.add((text_span, tuple[1]))
        return list(set_tuples)

    for pred_tuples, gold_tuples in zip(all_perDoc_predSpans, all_perDoc_goldSpans):
        pred_tuples = list_of_tuples_to_normalized_set(pred_tuples)
        #print(pred_tuples)
        gold_tuples = list_of_tuples_to_normalized_set(gold_tuples)
        #print(gold_tuples)
        for t in pred_tuples:
            if t in gold_tuples:
                n_correct += 1
                scores_per_NE[t[-1]]['n_correct'] += 1
            n_pos_pred += 1
            scores_per_NE[t[-1]]['n_pos_pred'] += 1
        n_pos_gold += len(gold_tuples)
        for g_t in gold_tuples:
            scores_per_NE[g_t[-1]]['n_pos_gold'] += 1

    prec = n_correct / (n_pos_pred + 1e-10)
    recall = n_correct / (n_pos_gold + 1e-10)
    f1 = 2 * prec * recall / (prec + recall + 1e-10)

    precision = round(prec * 100, 2)
    recall = round(recall * 100, 2)
    f1 = round(f1 * 100, 2)
    print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format("BUSTER", precision, recall, f1))

    macro_precision = []
    macro_recall = []
    macro_f1 = []
    for ne, ne_scores in scores_per_NE.items():
        prec = ne_scores['n_correct'] / (ne_scores['n_pos_pred'] + 1e-10)
        recall = ne_scores['n_correct'] / (ne_scores['n_pos_gold'] + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)

        precision = round(prec * 100, 2)
        macro_precision.append(precision)
        recall = round(recall * 100, 2)
        macro_recall.append(recall)
        f1 = round(f1 * 100, 2)
        macro_f1.append(f1)
        print("{} --> support: {}".format(ne, ne_scores['n_pos_gold']))
        print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(ne, ne_scores['n_correct'],
                                                             ne_scores['n_pos_gold'] - ne_scores['n_correct'],
                                                             ne_scores['n_pos_pred'] - ne_scores['n_correct'], -1))
        print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(ne, precision, recall, f1))
        print("------------------------------------------------------- ")

    # remove Hallucination before computing macros

    print("\n{} ==> Macro-Precision: {:.2f}, Macro-Recall: {:.2f}, Macro-F1: {:.2f}".format("BUSTER", np.average(macro_precision[:-1]), np.average(macro_recall[:-1]), np.average(macro_f1[:-1])))

