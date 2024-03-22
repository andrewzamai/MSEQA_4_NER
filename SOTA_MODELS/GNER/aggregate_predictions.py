from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import json

from GNER import evaluate as GNER_evaluate
from MSEQA_4_NER.data_handlers import data_handler_BUSTER


if __name__ == '__main__':

    """ 1) extracting GOLD TUPLES for each document (document level) """

    BUSTER_BIO_test = data_handler_BUSTER.loadDataset('../../../datasets/BUSTER/FULL_KFOLDS/123_4_5')['test']
    gold_tuples_per_doc = {x: None for x in BUSTER_BIO_test['document_id']}
    total_support = 0
    for gold_sample in BUSTER_BIO_test:
        # we need to convert BIO labels as GNER parser expects e.g B-TagFamily.TagName --> B-buying company
        gold_labels_NL_format = []
        for label in gold_sample['labels']:
            if label != 'O':
                label_prefix, label_tagFamTagName = label.split('-')  # splitting B-tagFamily.TagName
                # converting tagName in Natural language format as label_list
                label_NL = data_handler_BUSTER.convert_tagName_in_natural_language_format(label_tagFamTagName)
                label = label_prefix + '-' + label_NL
                gold_labels_NL_format.append(label)
            else:
                gold_labels_NL_format.append(label)

        gold_tuples = GNER_evaluate.parser(gold_sample['tokens'], gold_labels_NL_format)
        total_support += len(gold_tuples)

        gold_tuples_per_doc[gold_sample['document_id']] = gold_tuples

    print(total_support)

    """ 2) LOADING PREDICTIONS, applying Hierarchical Matching Algorithm, parsing to pred_tuples """

    # load tokenizer and prediction data
    tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")

    # grouping all chunks in which a document was divided
    chunks_per_document = defaultdict(list)
    with open('./BUSTER_GNER/BUSTER_GNER_test_w_preds.jsonl', 'r') as fh:
        for line in fh.readlines():
            line_data = json.loads(line)
            chunks_per_document[line_data['instance']['id']].append(line_data)

    # print(chunks_per_document.keys())
    print(f"Number of ducuments: {len(chunks_per_document)}")

    preds_per_doc = {x: None for x in BUSTER_BIO_test['document_id']}
    for document_id, thisDoc_chunks in chunks_per_document.items():
        # print(document_id)
        this_document_preds = set()
        for chunk_sample in thisDoc_chunks:
            # print(chunk_sample)
            thischunk_predictions_BIO_sequence = GNER_evaluate.extract_predictions(chunk_sample, tokenizer)
            # print(thischunk_predictions_BIO_sequence) # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-buying company', 'O', 'O'

            words = chunk_sample['instance']['words']
            thischunk_predictions = GNER_evaluate.parser(words, thischunk_predictions_BIO_sequence)
            # print(thischunk_predictions) # [('rosetta resources inc', 'buying company'), ('permian', 'acquired company') ...]
            for pred in thischunk_predictions:
                this_document_preds.add(pred)

        preds_per_doc[document_id] = list(this_document_preds)

    """ 3) compute scores """
    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    ne_types_list = ['generic consulting company', 'legal consulting company', 'annual revenues', 'acquired company', 'buying company', 'selling company']
    scores_per_NE = {ne: {"n_correct": 0, "n_pos_gold": 0, "n_pos_pred": 0} for ne in ne_types_list}

    for document_id in BUSTER_BIO_test['document_id']:
        gold_tuples = gold_tuples_per_doc[document_id]
        pred_tuples = preds_per_doc[document_id]
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

    for ne, ne_scores in scores_per_NE.items():
        prec = ne_scores['n_correct'] / (ne_scores['n_pos_pred'] + 1e-10)
        recall = ne_scores['n_correct'] / (ne_scores['n_pos_gold'] + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)

        precision = round(prec * 100, 2)
        recall = round(recall * 100, 2)
        f1 = round(f1 * 100, 2)
        print("{} --> support: {}".format(ne, ne_scores['n_pos_gold']))
        print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(ne, ne_scores['n_correct'], ne_scores['n_pos_gold'] - ne_scores['n_correct'], ne_scores['n_pos_pred'] - ne_scores['n_correct'], -1))
        print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(ne, precision, recall, f1))
        print("------------------------------------------------------- ")





    """
    gold_labels_NL_format = []
    for label in BUSTER_BIO_test[0]['labels']:
        if label != 'O':
            label_prefix, label_tagFamTagName = label.split('-')  # splitting B-tagFamily.TagName
            # converting tagName in Natural language format as label_list
            label_NL = data_handler_BUSTER.convert_tagName_in_natural_language_format(label_tagFamTagName)
            label = label_prefix + '-' + label_NL
            gold_labels_NL_format.append(label)
        else:
            gold_labels_NL_format.append(label)
    print(gold_labels_NL_format)

    gold_tuples = GNER_evaluate.parser(BUSTER_BIO_test[0]['tokens'], gold_labels_NL_format)
    print(gold_tuples)

    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    for t in this_document_preds:
        if t in gold_tuples:
            n_correct += 1
        n_pos_pred += 1
    n_pos_gold += len(gold_tuples)
    prec = n_correct / (n_pos_pred + 1e-10)
    recall = n_correct / (n_pos_gold + 1e-10)
    f1 = 2 * prec * recall / (prec + recall + 1e-10)

    print(prec, recall, f1)
    """



