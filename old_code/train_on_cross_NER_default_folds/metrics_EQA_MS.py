""" ------------------------ MULTI-SPAN EQA for NER METRICS ------------------------ """

import numpy as np
import collections
import string
import re

# personal libraries
import merge_passage_answers

"""
compute overlap between 2 int intervals: [a,b] and [c,d]
(indices of the start/end characters of the answers to check for overlap)
"""
def compute_overlap_char_indices(interval_1, interval_2):
    start_1, end_1 = interval_1
    start_2, end_2 = interval_2

    overlap_start = max(start_1, start_2)
    overlap_end = min(end_1, end_2)

    overlap = max(0, overlap_end - overlap_start + 1)

    return overlap


""" 
Given all predictions related to a question computes precision, recall, f1 for the question (i.e. for a NE category) 
Aggregations from passage predictions to doc level predictions is done HERE
"""
def precision_recall_f1_multi_answer(this_question_preds):

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for sample in this_question_preds:
        docID = ":".join(str(i) for i in sample['doc_question_pairID'].split(':')[:-1])  # removing questionID ':number'
        document_context = sample['document_context']
        gold_answers = sample['gold_answers']  # text and start_char indices
        pred_answers_divided_per_passage = sample['predicted_answers']

        # merging answers from passage level to document level
        pred_answers_for_doc = merge_passage_answers.merge_predictions_from_passages(pred_answers_divided_per_passage)

        # building gold answers with (start, end) char indices
        gold_answers_with_char_intervals = []
        for answer_start, text in zip(gold_answers['answer_start'], gold_answers['text']):
            answer_end = answer_start + len(text)
            gold_answers_with_char_intervals.append({'text': text, 'start_end_indices': [answer_start, answer_end]})
        if not gold_answers_with_char_intervals:
            gold_answers_with_char_intervals.append({'text': '', 'start_end_indices': [0, 0]})

        hitten_gold_answers_indices = []  # indices of the GAs that are hit by some prediction (used to find the FN)
        # if model pred is noanswer then is just a single dict item {}
        if not isinstance(pred_answers_for_doc, list):
            if pred_answers_for_doc['text'] == '' and gold_answers_with_char_intervals[0]['text'] == '':
                tn += 1
            elif pred_answers_for_doc['text'] == '' and gold_answers_with_char_intervals[0]['text'] != '':
                fn += len(gold_answers_with_char_intervals)  # FNs: number of missed GAs by predicting ''
        else:
            # one (!= '') or more predicted answers
            for pred in pred_answers_for_doc:
                hit = False
                for i, ga in enumerate(gold_answers_with_char_intervals):
                    # hard hit between character intervals
                    if pred['start_end_indices'] == ga['start_end_indices']:
                        hit = True
                        hitten_gold_answers_indices.append(i)
                if hit:
                    # if '' had highest score in the passage than all the other answers were removed
                    # but a '' answer may still appear as answer in a list of non empty answers (with low score)
                    if pred['text'] != '':
                        tp += 1
                    else:
                        tn += 1  # TN if a '' prediction hits '' GA
                else:
                    fp += 1  # error, but we do not count here FNs

            # FNs is the number of missed GAs not hit by any prediction
            missed_ga = set(range(len(gold_answers_with_char_intervals))) - set(hitten_gold_answers_indices)
            # when pred is some FP but GA was '' then len(missed_ga) = 1, we don't have to count it as fn
            if gold_answers_with_char_intervals[0]['text'] != '':
                fn += len(missed_ga)

    print("TP: {}, FN: {}, FP: {}, TN: {}".format(tp, fn, fp, tn))

    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

    f1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall, 'f1': f1}

    return metrics


# ------------- OTHER USEFUL METRICS -------------

"""
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
