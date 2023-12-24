""" ---  MULTI-SPAN EQA for NER metrics --- """


import collections
import json
from typing import Tuple, List


def compute_overlap_char_indices(interval_1: Tuple[int, int], interval_2: Tuple[int, int]) -> int:
    """
    Compute overlap between 2 int intervals: [a, b] and [c, d]
    where a is the starting position of the first answer in characters
    and b is the ending position of the first answer in characters,
    similarly for c and d of the second answer to check overlap with

    Parameters:
    - interval_1 (Tuple[int, int]): First interval [a, b]
    - interval_2 (Tuple[int, int]): Second interval [c, d]

    Returns:
    - int: Length of the overlap between the two intervals (0 if no overlap)
    """
    start_1, end_1 = interval_1
    start_2, end_2 = interval_2

    overlap_start = max(start_1, start_2)
    overlap_end = min(end_1, end_2)

    overlap = max(0, overlap_end - overlap_start + 1)

    return overlap


def compute_precision_recall_f1_for_a_NE_category(all_predictions_related_to_a_tagName: List[dict]) -> dict:
    """
    Given all predictions related to a NE category compute precision, recall, f1 for this NE category (question)
    Each prediction is a dict with fields 'gold_answers', 'predicted_answers_doc_level' ...
    """

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for sample in all_predictions_related_to_a_tagName:

        gold_answers = sample["gold_answers"]  # text and start_char indices only
        pred_answers_for_doc = sample["predicted_answers_doc_level"]

        # building gold answers with (start, end) char indices
        gold_answers_with_char_intervals = []
        for answer_start, text in zip(gold_answers['answer_start'], gold_answers['text']):
            answer_end = answer_start + len(text)
            gold_answers_with_char_intervals.append({'text': text, 'start_end_indices': [answer_start, answer_end]})
        if not gold_answers_with_char_intervals:
            gold_answers_with_char_intervals.append({'text': '', 'start_end_indices': [0, 0]})

        hit_gold_answers_indices = []  # indices of the GAs that are hit by some prediction (used to find the FN)
        # if model pred is noanswer then is just a single dict item {}, i.e. not a list
        if not isinstance(pred_answers_for_doc, list):
            if pred_answers_for_doc['text'] == '' and gold_answers_with_char_intervals[0]['text'] == '':
                tn += 1
            elif pred_answers_for_doc['text'] == '' and gold_answers_with_char_intervals[0]['text'] != '':
                fn += len(gold_answers_with_char_intervals)  # FNs: number of missed GAs by predicting '' no answer
        else:
            # one (!= '') or more predicted answers
            for pred in pred_answers_for_doc:
                hit = False
                for i, ga in enumerate(gold_answers_with_char_intervals):
                    # hard hit between character intervals
                    if pred['start_end_indices'] == ga['start_end_indices']:
                        hit = True
                        hit_gold_answers_indices.append(i)
                if hit:
                    # if '' had highest score in the passage than all the other answers were removed
                    # but a '' answer may still appear as answer in a list of non empty answers, with a low score
                    if pred['text'] != '':
                        tp += 1
                    else:
                        tn += 1  # TN if a '' prediction hits '' GA
                else:
                    fp += 1  # error, but we do not count here FNs

            # FNs is the number of missed GAs not hit by any prediction
            missed_ga = set(range(len(gold_answers_with_char_intervals))) - set(hit_gold_answers_indices)
            # when pred is some FP but GA was '' then len(missed_ga) = 1, we don't have to count it as fn
            if gold_answers_with_char_intervals[0]['text'] != '':
                fn += len(missed_ga)

    tagName = all_predictions_related_to_a_tagName[0]['tagName']
    # print("NE category: {} --> TP: {}, FN: {}, FP: {}, TN: {}".format(tagName, tp, fn, fp, tn))

    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

    f1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall, 'f1': f1}

    return metrics


def compute_micro_precision_recall_f1(pred_answers_for_a_quest_on_doc_list: List[dict]) -> dict:
    """
    Compute micro precision, recall and F1 across all predictions (NE category agnostic)
    """

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    false_positives = []
    true_positives = []
    false_negatives = []

    for sample in pred_answers_for_a_quest_on_doc_list:

        gold_answers = sample['gold_answers']  # text and start_char indices
        pred_answers_for_doc = sample['predicted_answers_doc_level']

        # building gold answers with (start, end) char indices
        gold_answers_with_char_intervals = []
        for answer_start, text in zip(gold_answers['answer_start'], gold_answers['text']):
            answer_end = answer_start + len(text)
            gold_answers_with_char_intervals.append({'text': text, 'start_end_indices': [answer_start, answer_end]})
        if not gold_answers_with_char_intervals:
            gold_answers_with_char_intervals.append({'text': '', 'start_end_indices': [0, 0]})

        hit_gold_answers_indices = []  # indices of the GAs that are hit by some prediction (used to find the FN)
        # if model pred is noanswer then is just a single dict item {}
        if not isinstance(pred_answers_for_doc, list):
            if pred_answers_for_doc['text'] == '' and gold_answers_with_char_intervals[0]['text'] == '':
                tn += 1
            elif pred_answers_for_doc['text'] == '' and gold_answers_with_char_intervals[0]['text'] != '':
                fn += len(gold_answers_with_char_intervals)  # FNs: number of missed GAs by predicting ''
                # adding to false negatives list
                for fn_occ in gold_answers_with_char_intervals:
                    sentence_fn = sample['document_context'][max(0, fn_occ['start_end_indices'][0] - 50): min(len(sample['document_context']), fn_occ['start_end_indices'][1] + 50)]
                    false_negatives.append({'doc_question_pairID': sample['doc_question_pairID'],
                                            'tagName': sample['tagName'],
                                            'missed_entity': fn_occ['text'],
                                            'start_end_indices': fn_occ['start_end_indices'],
                                            'sentence': sentence_fn})
        else:
            # one (!= '') or more predicted answers
            for pred in pred_answers_for_doc:
                hit = False
                for i, ga in enumerate(gold_answers_with_char_intervals):
                    # hard hit between character intervals
                    if pred['start_end_indices'] == ga['start_end_indices']:
                        hit = True
                        hit_gold_answers_indices.append(i)
                if hit:
                    # if '' had highest score in the passage than all the other answers were removed
                    # but a '' answer may still appear as answer in a list of non empty answers (with low score)
                    if pred['text'] != '':
                        tp += 1
                        sentence_tp = sample['document_context'][max(0, pred['start_end_indices'][0] - 50): min(len(sample['document_context']), pred['start_end_indices'][1] + 50)]
                        true_positives.append({'doc_question_pairID': sample['doc_question_pairID'],
                                               'tagName': sample['tagName'],
                                               'pred_text': pred['text'],
                                               'start_end_indices': pred['start_end_indices'],
                                               'sentence': sentence_tp})
                    else:
                        tn += 1  # TN if a '' prediction hits '' GA
                else:
                    fp += 1  # error, but we do not count here FNs

                    # TODO: extracting FPs
                    # we don't care about FPs that are so only because of not perfect start/end match, but real FPs
                    if not any([compute_overlap_char_indices(pred['start_end_indices'], ga['start_end_indices']) > 0 for ga in gold_answers_with_char_intervals]):
                        sentence_fp = sample['document_context'][max(0, pred['start_end_indices'][0] - 50): min(len(sample['document_context']), pred['start_end_indices'][1] + 50)]
                        false_positives.append({'doc_question_pairID': sample['doc_question_pairID'],
                                                'tagName': sample['tagName'],
                                                'pred_text': pred['text'],
                                                'start_end_indices': pred['start_end_indices'],
                                                'sentence': sentence_fp
                                                })

            # FNs is the number of missed GAs not hit by any prediction
            missed_ga = set(range(len(gold_answers_with_char_intervals))) - set(hit_gold_answers_indices)
            # when pred is some FP but GA was '' then len(missed_ga) = 1, we don't have to count it as fn
            if gold_answers_with_char_intervals[0]['text'] != '':
                fn += len(missed_ga)

            # adding to false negatives list
            for i, ga in enumerate(gold_answers_with_char_intervals):
                if i not in hit_gold_answers_indices and ga['text'] != '':
                    sentence_fn = sample['document_context'][max(0, ga['start_end_indices'][0] - 50): min(len(sample['document_context']), ga['start_end_indices'][1] + 50)]
                    false_negatives.append({'doc_question_pairID': sample['doc_question_pairID'],
                                            'tagName': sample['tagName'],
                                            'missed_entity': ga['text'],
                                            'start_end_indices': ga['start_end_indices'],
                                            'sentence': sentence_fn})

    print("TP: {}, FN: {}, FP: {}, TN: {}".format(tp, fn, fp, tn))

    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

    f1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1}

    # TODO: saving FPs
    with open('./predictions/false_positives.json', 'w', encoding='utf-8') as f:
        json.dump(false_positives, f, ensure_ascii=False, indent=4)
    with open('./predictions/true_positives.json', 'w', encoding='utf-8') as f:
        json.dump(true_positives, f, ensure_ascii=False, indent=4)
    with open('./predictions/false_negatives.json', 'w', encoding='utf-8') as f:
        json.dump(false_negatives, f, ensure_ascii=False, indent=4)

    return metrics


def compute_all_metrics(pred_answers_for_a_quest_on_doc_list: List[dict]) -> Tuple[dict, dict]:
    """
    Compute micro, macro and average Precision, Recall, F1
    """
    # first construct a dictionary that for each tagName keeps the indices of the quest-doc in the list with that tagName
    indices_per_tagName = collections.defaultdict(list)
    for sample in pred_answers_for_a_quest_on_doc_list:
        indices_per_tagName[sample["tagName"]].append(sample)

    # now we compute for each NE category its TP, FN, FP, TN and Prec, Recall, F1
    metrics_per_tagName = {}
    for tagName, preds_associated_to_this_tagName in indices_per_tagName.items():
        metrics_per_tagName[tagName] = compute_precision_recall_f1_for_a_NE_category(preds_associated_to_this_tagName)

    # sorted by decreasing support count
    metrics_per_tagName = dict(sorted(metrics_per_tagName.items(), key=lambda x: x[1]['tp'] + x[1]['fn'], reverse=True))

    """ ---------- MACRO-AVERAGES ---------- """
    # MACRO-precision
    macro_precision = 0.0
    for q, m in metrics_per_tagName.items():
        macro_precision += m["precision"]
    macro_precision /= len(metrics_per_tagName.keys())

    # MACRO-recall
    macro_recall = 0.0
    for q, m in metrics_per_tagName.items():
        macro_recall += m["recall"]
    macro_recall /= len(metrics_per_tagName.keys())

    # MACRO-F1
    macro_f1 = 0
    for q, m in metrics_per_tagName.items():
        macro_f1 += m["f1"]
    macro_f1 /= len(metrics_per_tagName.keys())

    """ ---------- WEIGHTED AVERAGES ---------- """
    question_support = {}
    for q, m in metrics_per_tagName.items():
        question_support[q] = m["tp"] + m["fn"]
    total_support = sum(question_support.values())

    weighted_averaged_precision = 0
    for q, m in metrics_per_tagName.items():
        weighted_averaged_precision += (question_support[q] / total_support) * m["precision"]

    weighted_averaged_recall = 0
    for q, m in metrics_per_tagName.items():
        weighted_averaged_recall += (question_support[q] / total_support) * m["recall"]

    weighted_averaged_f1 = 0
    for q, m in metrics_per_tagName.items():
        weighted_averaged_f1 += (question_support[q] / total_support) * m["f1"]

    """ ---------- micro-AVERAGES ---------- """
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for q, m in metrics_per_tagName.items():
        tp_total += m["tp"]
        fp_total += m["fp"]
        fn_total += m["fn"]
    micro_precision = (0 if (tp_total + fp_total) == 0 else tp_total / (tp_total + fp_total))
    micro_recall = 0 if (tp_total + fn_total) == 0 else tp_total / (tp_total + fn_total)
    micro_f1 = (0 if (micro_precision + micro_recall) == 0 else (2 * micro_precision * micro_recall) / (micro_precision + micro_recall))

    overall_metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_averaged_precision": weighted_averaged_precision,
        "weighted_averaged_recall": weighted_averaged_recall,
        "weighted_averaged_f1": weighted_averaged_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }

    return overall_metrics, metrics_per_tagName


def get_predictions_squad_for_official_eval_script(pred_answers_for_a_quest_on_doc_list):
    # constructing json file for squad-evaluation official script
    predictions_squad_format = {}
    for sample in pred_answers_for_a_quest_on_doc_list:
        id = sample['doc_question_pairID']
        # merging answers from passage level to document level
        pred_answers_for_doc = sample['predicted_answers_doc_level']
        # keeping always only top answer
        if isinstance(pred_answers_for_doc, list):
            pred_answers_for_doc = pred_answers_for_doc[0]

        predictions_squad_format[id] = pred_answers_for_doc['text']

    #with open(os.path.join(output_dir, 'predictions_squad_format.json'), 'w', encoding='utf-8') as f:
    #json.dump(predictions_squad_format, f)
    return predictions_squad_format


# ------------- OTHER USEFUL METRICS -------------
"""
import numpy as np
import collections
import string
import re

# lower text, remove punctuation, articles and extra whitespace
# from SQuAD2 official eval script, different normalization if is_annual_revenue
def normalize_answer(s, is_annual_revenue=False):

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text, is_annual_revenue=False):
        # string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        if is_annual_revenue:
            # otherwise $3.61 millions becomes 361 millions
            keep = ['$', '£', '€', '%', '-', '+', ',', '.', '#']
        else:
            keep = ['&']  # eg. D & G
        punc_set = list(filter(lambda item: item not in keep, set(string.punctuation)))
        return ''.join(ch for ch in text if ch not in punc_set)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s), is_annual_revenue)))

def get_tokens(s, is_annual_revenue=False):
    # is s is None or empty string '' return empty array
    if not s:
        return []
    return normalize_answer(s, is_annual_revenue).split()


# -------------- EXACT overlap score returns {0,1} --------------
# STEPS: 1) normalize answers; 2) 1 if ==, 0 otherwise

def exact_overlap_score(a_pred, a_gold, is_annual_revenue=False):
    # normalizing answers and splitting into words
    pred_tokens = get_tokens(a_pred, is_annual_revenue)
    gold_tokens = get_tokens(a_gold, is_annual_revenue)

    return int(pred_tokens == gold_tokens)  # {0, 1}


# -------------- SOFT overlap score returns [0,1] ---------------

def soft_overlap_score(a_pred, a_gold, is_annual_revenue=False):
    # normalizing answers and splitting into words
    pred_tokens = get_tokens(a_pred, is_annual_revenue)
    gold_tokens = get_tokens(a_gold, is_annual_revenue)

    # counting common words order independent
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)  # Counter({'acorn': 1, 'energy': 1})
    num_same = sum(common.values())

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # If either is no-answer, then F1 overlap score is 1 if they agree, 0 otherwise
        return int(gold_tokens == pred_tokens)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1_overlap = (2 * precision * recall) / (precision + recall)

    return f1_overlap


# compute soft overlap score of a normalized answer against a list of gold normalized answers
def compute_soft_overlap_listOfGolds(a_pred, a_gold_list, is_annual_revenue=False):
    # for unanswerable questions, only correct answer is empty string ''
    a_gold_list_new = [a for a in a_gold_list if normalize_answer(a, is_annual_revenue)]
    # if all answers are '' after normalization then answer is empty string, or if already [] (no answer)
    if not a_gold_list_new:
        a_gold_list_new = ['']
    # max against all gold labels
    overlap_scores = [soft_overlap_score(a_pred, a, is_annual_revenue) for a in a_gold_list_new]
    # since a_gold_list_new may be != from a_gold_list to retrieve answer maximizing score
    argmax_in_new = np.argmax(overlap_scores)
    gold_answer_maximizing_overlap = a_gold_list_new[argmax_in_new]
    return max(overlap_scores), gold_answer_maximizing_overlap

"""
