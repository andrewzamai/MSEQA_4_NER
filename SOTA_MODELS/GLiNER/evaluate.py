from datasets import load_dataset, Dataset
from collections import defaultdict
import numpy as np

import GNER.evaluate as GNER_evaluate
from MSEQA_4_NER.data_handlers import data_handler_BUSTER


if __name__ == '__main__':

    BUSTER_test_w_preds = load_dataset(path='json', data_files='./GLiNER_large_on_BUSTER_chunked_123_4_5.jsonl')['train']
    print(BUSTER_test_w_preds[0])

    all_gold_tuples_per_doc = defaultdict(set)
    all_pred_tuples_per_doc = defaultdict(set)

    all_gold_tuples_list = []
    all_pred_tuples_list = []
    for BIO_chunk_sample_w_pred in BUSTER_test_w_preds:
        # deleting tagFamily from labels and converting to natural language format as predicted labels
        gold_labels_per_token = []
        for label in BIO_chunk_sample_w_pred['labels']:
            if label != 'O':
                label_prefix, label_tagFamTagName = label.split('-')  # splitting B-tagFamily.TagName
                # converting tagName in Natural language format as label_list
                label_NL = data_handler_BUSTER.convert_tagName_in_natural_language_format(label_tagFamTagName)
                label = label_prefix + '-' + label_NL
                gold_labels_per_token.append(label)
            else:
                gold_labels_per_token.append(label)
        gold_tuples = GNER_evaluate.parser(BIO_chunk_sample_w_pred['tokens'], gold_labels_per_token)

        predicted_spans = BIO_chunk_sample_w_pred['prediction']
        pred_tuples = [(GNER_evaluate.normalize_answer(x['text']), x['label']) for x in predicted_spans]
        # pred_tuples = GNER_evaluate.parser(BIO_chunk_sample['tokens'], BIO_chunk_pred)

        print(gold_tuples)
        print(pred_tuples)

        # all_gold_tuples_list.append(gold_tuples)
        # all_pred_tuples_list.append(pred_tuples)

        document_id = BIO_chunk_sample_w_pred['document_id']
        all_gold_tuples_per_doc[document_id].update(gold_tuples)
        all_pred_tuples_per_doc[document_id].update(pred_tuples)


    """ 3) compute scores """

    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    ne_types_list = ['generic consulting company', 'legal consulting company', 'annual revenues', 'acquired company', 'buying company', 'selling company']
    scores_per_NE = {ne: {"n_correct": 0, "n_pos_gold": 0, "n_pos_pred": 0} for ne in ne_types_list}

    # for pred_tuples, gold_tuples in zip(all_pred_tuples_per_doc, all_gold_tuples_per_doc):
    unique_set_document_ids = set(BUSTER_test_w_preds['document_id'])
    # DO NOT USE BUSTER_BIO_chunked_test['document_id'] otherwise multiple IDs per ID
    for document_id in unique_set_document_ids:
        pred_tuples = all_pred_tuples_per_doc[document_id]
        gold_tuples = all_gold_tuples_per_doc[document_id]
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
    print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format("BUSTER", precision, recall,
                                                                                            f1))

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

    print("\n{} ==> Macro-Precision: {:.2f}, Macro-Recall: {:.2f}, Macro-F1: {:.2f}".format("BUSTER",
                                                                                            np.average(macro_precision),
                                                                                            np.average(macro_recall),
                                                                                            np.average(macro_f1)))
