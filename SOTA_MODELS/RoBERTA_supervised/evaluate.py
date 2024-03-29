from collections import defaultdict

import numpy as np
from datasets import load_dataset

import GNER.evaluate as GNER_evaluate


if __name__ == '__main__':

    path_to_BUSTER_BIO_chunked = '../../../datasets/BUSTER/CHUNKED_KFOLDS/123_4_5/test.json'
    BUSTER_BIO_chunked_test = load_dataset(path='json', data_files=path_to_BUSTER_BIO_chunked)['train']

    #print(' '.join(BUSTER_BIO_chunked_test[0]['tokens']))

    path_to_predictions = "./chunked_123_4_5_predictions.txt"
    BIOpreds_per_chunk = []
    with open(path_to_predictions, "r") as file:
        for line in file:
            # convert each BIO string pred to list
            BIOpreds_per_chunk.append(line.split())
    print(len(BIOpreds_per_chunk))
    print(len(BUSTER_BIO_chunked_test))

    all_gold_tuples_per_doc = defaultdict(set)
    all_pred_tuples_per_doc = defaultdict(set)

    all_gold_tuples_list = []
    all_pred_tuples_list = []
    for BIO_chunk_sample, BIO_chunk_pred in zip(BUSTER_BIO_chunked_test, BIOpreds_per_chunk):
        print(BIO_chunk_sample)
        print(BIO_chunk_pred)
        if len(BIO_chunk_sample['labels']) != len(BIO_chunk_pred):
            raise Exception("Number pred labels differs from number gold labels")

        gold_tuples = GNER_evaluate.parser(BIO_chunk_sample['tokens'], BIO_chunk_sample['labels'])
        pred_tuples = GNER_evaluate.parser(BIO_chunk_sample['tokens'], BIO_chunk_pred)

        print(gold_tuples)
        print(pred_tuples)

        #all_gold_tuples_list.append(gold_tuples)
        #all_pred_tuples_list.append(pred_tuples)

        document_id = BIO_chunk_sample['document_id']
        all_gold_tuples_per_doc[document_id].update(gold_tuples)
        all_pred_tuples_per_doc[document_id].update(pred_tuples)
        """
        for g_t in gold_tuples:
            all_gold_tuples_per_doc[BIO_chunk_sample['document_id']].update(g_t)
        for p_t in pred_tuples:
            all_pred_tuples_per_doc[BIO_chunk_sample['document_id']].add(p_t)
        """

    print(len(all_gold_tuples_per_doc.keys()))
    print(len(all_pred_tuples_per_doc.keys()))
    print(list(all_gold_tuples_per_doc.values())[:10])
    print(list(all_pred_tuples_per_doc.values())[:10])

    """ 3) compute scores """
    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    ne_types_list = ['advisorsgenericconsultingcompany', 'advisorslegalconsultingcompany', 'genericinfoannualrevenues', 'partiesacquiredcompany', 'partiesbuyingcompany', 'partiessellingcompany']
    scores_per_NE = {ne: {"n_correct": 0, "n_pos_gold": 0, "n_pos_pred": 0} for ne in ne_types_list}

    #for pred_tuples, gold_tuples in zip(all_pred_tuples_per_doc, all_gold_tuples_per_doc):
    unique_set_document_ids = set(BUSTER_BIO_chunked_test['document_id'])
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

    print("\n{} ==> Macro-Precision: {:.2f}, Macro-Recall: {:.2f}, Macro-F1: {:.2f}".format("BUSTER", np.average(macro_precision), np.average(macro_recall), np.average(macro_f1)))