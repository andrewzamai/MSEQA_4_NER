"""
A long-document may be split in multiple passages {(question;chunkOfText)}, here:
1) we exctract multiple predictions from each passage and collect them per document-question pair
2) we aggregate separate passage prediction to document level predictions through 'merge_passage_answers' module (which is called inside precision_recall_f1_multi_answer)
3) we compute metrics performance on each question (i.e. for each NE category) and overall

Input:
- model's outputs on passages from validation/test fold (all_start_logits, all_end_logits, passage_ids ...)
Output:
- list of extracted predictions from each passage for each document-question pair (saved as .json)
- metrics computation on aggregated predictions
"""

import collections
import json
import torch

# my libraries
import data_handler_cross_NER
import metrics_EQA_MS

"""
collecting predictions for each document-question pair:
(a list of lists: multiple preds for each passage in which a document is chunked)
with little post-processing within a passage list of preds (decreasing score sort and threshold on score) 
"""
def predict_on_passages(questions,
                        batchStep,
                        foldName,
                        tokenizer,
                        dataset_doc_quest_ans,
                        floatIDs_to_strIDs,
                        model_outputs_for_metrics):

    # unpacking prediction outputs
    all_start_logits = model_outputs_for_metrics['all_start_logits']
    all_end_logits = model_outputs_for_metrics['all_end_logits']
    input_ids_list = model_outputs_for_metrics['input_ids_list']
    passage_id_list = model_outputs_for_metrics['passage_id_list']
    sequence_ids_list = model_outputs_for_metrics['sequence_ids_list']
    offset_mapping_list = model_outputs_for_metrics['offset_mapping_list']

    MAX_ANSW_LENGTH_IN_TOK = 10  # hyperparameter, set also in Model class if using newLoss

    # for each doc_quest floatID which indices in the model outputs belong to that document?
    indices_per_document = collections.defaultdict(list)
    for i, passage_id in enumerate(passage_id_list):
        indices_per_document[passage_id.item()].append(i)

    # all document-question pairs retrieved predictions
    predictions_on_docs_divided_per_passage = []

    for doc_quest_answers in dataset_doc_quest_ans[foldName]:
        floatID_doc_question = doc_quest_answers['doc_question_pairID']
        strID_doc_question = floatIDs_to_strIDs[foldName][floatID_doc_question]
        # if not running inference over all passages some doc may not have any passage predictions
        if floatID_doc_question not in indices_per_document:
            continue
        passages_indices = indices_per_document[floatID_doc_question]

        # list of lists (multiple preds from each passage) for this document-question pair
        doc_level_answers = []

        for index in passages_indices:
            start_logits = all_start_logits[index] #torch.from_numpy(all_start_logits[index])
            end_logits = all_end_logits[index] #torch.from_numpy(all_end_logits[index])

            input_ids = input_ids_list[index]
            sequence_ids = sequence_ids_list[index]  # is already tensor object from dataCollator
            offset_mapping = offset_mapping_list[index]

            # valid predictions for this passage
            valid_answers_passage = []

            # we are here working with NO BATCHED TENSORS

            # getting all (start_logits + end_logits) score combinations
            s_e_logits = torch.einsum('i,j->ij', start_logits, torch.ones_like(end_logits)) + torch.einsum('i,j->ij', torch.ones_like(start_logits), end_logits)
            # s_e_logits = torch.einsum('i,j->ij', start_logits, torch.abs(end_logits))
            # disqualify answers where end < start
            # i.e. set the lower triangular matrix to low value, excluding diagonal
            max_seq_len = s_e_logits.shape[-1]
            indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=s_e_logits.device)
            s_e_logits[indices[0][:], indices[1][:]] = -888
            # disqualify answers where answer span is greater than max_answer_length
            # (set the upper triangular matrix to low value, excluding diagonal)
            indices_long_span = torch.triu_indices(max_seq_len, max_seq_len, offset=MAX_ANSW_LENGTH_IN_TOK, device=s_e_logits.device)
            s_e_logits[indices_long_span[0][:], indices_long_span[1][:]] = -777
            # disqualify answers where start=0, but end != 0 (i.e. first row of matrix)
            s_e_logits[0, 1:] = -666
            # disqualify spans where either start and/or end is on an invalid token
            sequence_ids_outer = torch.einsum('i,j->ij', sequence_ids, sequence_ids)  # outer product
            s_e_logits = torch.where(sequence_ids_outer == 0, -1000, s_e_logits)

            # now flattening the scores and getting indices start-end with highest logits sum
            flat_scores = s_e_logits.view(max_seq_len * max_seq_len, -1)
            flat_sorted = flat_scores.sort(descending=True, dim=0)
            flat_sorted_scores = flat_sorted[0]
            flat_sorted_indices = flat_sorted[1]
            # we can retrieve back start/end indices by div and %
            start_indices_sorted = torch.div(flat_sorted_indices, max_seq_len, rounding_mode="trunc")
            end_indices_sorted = flat_sorted_indices % max_seq_len
            # concatenating them along last dimension
            sorted_pairs_indices = torch.cat((start_indices_sorted, end_indices_sorted), dim=-1)
            # threshold on score
            # TODO: this score could be learned/optimized
            THRESHOLD_ON_SCORE = 0
            number_preds_over_threshold = len(flat_sorted_scores[flat_sorted_scores > THRESHOLD_ON_SCORE])
            sorted_pairs_indices = sorted_pairs_indices[:number_preds_over_threshold]
            sorted_pairs_score = flat_sorted_scores[:number_preds_over_threshold]

            for pair_indices, score in zip(sorted_pairs_indices, sorted_pairs_score):
                start_index = pair_indices[0].item()
                end_index = pair_indices[1].item()
                score = score.item()

                # converting from token indices predictions to char start/end predictions
                start_char = offset_mapping[start_index][0].item()
                end_char = offset_mapping[end_index][1].item()

                # when saving to json we need to save float32 as strings
                # .strip() removes white-space that would appear before detokenized text (BPE considers whitespaces)
                # no answer '' if start_index == end_index == 0
                valid_answers_passage.append(
                    {
                        "score": str(score),
                        "text": tokenizer.decode(input_ids[start_index: end_index + 1]).strip() if start_index + end_index != 0 else '',
                        "start_end_indices": [start_char, end_char]  # list and not tuple
                    }
                )

            # if top answer within a passage is no-answer '' then the overall answer to that passage is ''
            # may be [] if no predictions survived to threshold
            if valid_answers_passage == []:
                no_answer_pred = {'score': str(15.0), 'text': '', 'start_end_indices': [0, 0]}
                valid_answers_passage = [no_answer_pred]
            # if after sorting the top answer is '' then we delete all the other answers for this passage
            if valid_answers_passage[0]['text'] == '':
                valid_answers_passage = [valid_answers_passage[0]]

            doc_level_answers.append(valid_answers_passage)

        document_context = doc_quest_answers['document_context']
        gold_answers = doc_quest_answers['answers']  # in char positions

        index_to_tagName = [tagName for tagName, q in questions.items()]

        data = {'doc_question_pairID': strID_doc_question,
                'tagName': index_to_tagName[int(strID_doc_question.split(":")[-1])],
                'document_context': document_context,
                'gold_answers': gold_answers,
                'predicted_answers': doc_level_answers
                }
        predictions_on_docs_divided_per_passage.append(data)

    if not isinstance(batchStep, int):
        with open('./passages_preds/passages_preds_per_doc_batch_step_{}.json'.format(batchStep), 'w', encoding='utf-8') as f:
            json.dump(predictions_on_docs_divided_per_passage, f, ensure_ascii=False, indent=4)

    return predictions_on_docs_divided_per_passage


def compute_metrics(path_to_questions_txt, batchStep, foldName, tokenizer, dataset_doc_quest_ans, floatIDs_to_strIDs, model_outputs_for_metrics):

    # loading questions
    questions = data_handler_cross_NER.load_questions_from_txt(path_to_questions_txt)
    # instead of retrieving questionID by index we retrieve it by tagName
    #index_to_question = [q for tagName, q in questions.items()]
    index_to_tagName = [tagName for tagName, q in questions.items()]
    #question_to_index = {q: i for i, q in enumerate(index_to_question)}
    tagName_to_index = {tagName: i for i, tagName in enumerate(index_to_tagName)}

    # collecting predictions still divided per passage for each document-question sample
    predictions_on_docs_divided_per_passage = predict_on_passages(questions,
                                                                  batchStep,
                                                                  foldName,
                                                                  tokenizer,
                                                                  dataset_doc_quest_ans,
                                                                  floatIDs_to_strIDs,
                                                                  model_outputs_for_metrics)
    # precision-recall-f1 for each question
    metrics_per_question = {}
    for tagName, question in questions.items():
        # all document-question samples related to this question
        question_id = tagName_to_index[tagName]
        this_question_preds = list(filter(lambda x: int(x['doc_question_pairID'].split(':')[-1]) == question_id, predictions_on_docs_divided_per_passage))
        # aggregating passage predictions and computing metrics
        metrics = metrics_EQA_MS.precision_recall_f1_multi_answer(this_question_preds)
        #metrics_per_question[question] = metrics
        metrics_per_question[tagName] = metrics

    """ ---------- MACRO-AVERAGES ---------- """
    # MACRO-precision
    macro_precision = 0.0
    for q, m in metrics_per_question.items():
        macro_precision += m['precision']
    macro_precision /= len(metrics_per_question.keys())

    # MACRO-recall
    macro_recall = 0.0
    for q, m in metrics_per_question.items():
        macro_recall += m['recall']
    macro_recall /= len(metrics_per_question.keys())

    # MACRO-F1
    macro_f1 = 0
    for q, m in metrics_per_question.items():
        macro_f1 += m['f1']
    macro_f1 /= len(metrics_per_question.keys())

    """ ---------- WEIGHTED AVERAGES ---------- """
    question_support = {}
    for q, m in metrics_per_question.items():
        question_support[q] = m['tp'] + m['fn']
    total_support = sum(question_support.values())

    weighted_averaged_precision = 0
    for q, m in metrics_per_question.items():
        weighted_averaged_precision += ((question_support[q]/total_support) * m['precision'])

    weighted_averaged_recall = 0
    for q, m in metrics_per_question.items():
        weighted_averaged_recall += ((question_support[q]/total_support) * m['recall'])

    weighted_averaged_f1 = 0
    for q, m in metrics_per_question.items():
        weighted_averaged_f1 += ((question_support[q] / total_support) * m['f1'])

    """ ---------- micro-AVERAGES ---------- """
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for q, m in metrics_per_question.items():
        tp_total += m['tp']
        fp_total += m['fp']
        fn_total += m['fn']
    micro_precision = 0 if (tp_total + fp_total) == 0 else tp_total / (tp_total + fp_total)
    micro_recall = 0 if (tp_total + fn_total) == 0 else tp_total / (tp_total + fn_total)
    micro_f1 = 0 if (micro_precision + micro_recall) == 0 else (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    overall_metrics = {'macro_precision': macro_precision,
                       'macro_recall': macro_recall,
                       'macro_f1': macro_f1,
                       'weighted_averaged_precision': weighted_averaged_precision,
                       'weighted_averaged_recall': weighted_averaged_recall,
                       'weighted_averaged_f1': weighted_averaged_f1,
                       'micro_precision': micro_precision,
                       'micro_recall': micro_recall,
                       'micro_f1': micro_f1
                       }

    return overall_metrics, metrics_per_question

