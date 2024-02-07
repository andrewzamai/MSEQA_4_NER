"""
A long-document may be split in multiple passages {(question;chunkOfText)}, here:
1) we exctract multiple predictions from each passage and collect them per document-question pair
2) we aggregate separate passage predictions to document level predictions through 'merge_passage_answers' module
3) we compute metrics performance on each question (i.e. for each NE category) and overall

Input:
- model's outputs on passages from validation/test fold (all_start_logits, all_end_logits, passage_ids ...)
Output:
- list of extracted predictions from each passage for each document-question pair (saved as .json)
- metrics computation on aggregated predictions
"""

import collections
import torch
import json

from transformers import DebertaV2TokenizerFast

# my libraries
import merge_passage_answers


def extract_answers_per_passage_from_logits(max_ans_length_in_tokens,
                                            batch_step,
                                            print_json_every_batch_steps,
                                            fold_name,  # validation or test
                                            tokenizer,  # to decode back input_ids
                                            datasetdict_MSEQA_format,
                                            model_outputs_for_metrics):
    """
    Extract a list of answers per passage from model's logits, group and merge them per document level
    this will give a list of predicted answers for each document-question pair
    little post-processing within a passage list of preds (decreasing score sort and threshold on pair score>0)
    """

    # unpacking prediction outputs
    all_start_logits = model_outputs_for_metrics['all_start_logits']
    all_end_logits = model_outputs_for_metrics['all_end_logits']
    input_ids_list = model_outputs_for_metrics['input_ids_list']
    passage_id_list = model_outputs_for_metrics['passage_id_list']
    sequence_ids_list = model_outputs_for_metrics['sequence_ids_list']
    offset_mapping_list = model_outputs_for_metrics['offset_mapping_list']

    # for each doc_quest which indices in the model outputs belong to that document?
    indices_per_document = collections.defaultdict(list)
    for i, passage_id in enumerate(passage_id_list):
        indices_per_document[passage_id].append(i)

    # all document-question pairs retrieved predictions
    pred_answers_for_a_quest_on_doc_list = []

    for doc_quest_answers in datasetdict_MSEQA_format[fold_name]:
        passages_indices = indices_per_document[doc_quest_answers['doc_question_pairID']]

        # list of lists (multiple preds from each passage) for this document-question pair
        doc_level_answers = []

        for index in passages_indices:
            start_logits = all_start_logits[index]
            end_logits = all_end_logits[index]

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
            indices_long_span = torch.triu_indices(max_seq_len, max_seq_len, offset=max_ans_length_in_tokens, device=s_e_logits.device)
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
                if isinstance(tokenizer, DebertaV2TokenizerFast) and start_char != 0:
                    start_char += 1
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
            if not valid_answers_passage:
                no_answer_pred = {'score': str(15.0), 'text': '', 'start_end_indices': [0, 0]}
                valid_answers_passage = [no_answer_pred]
            # if after sorting the top answer is '' then we delete all the other answers for this passage
            if valid_answers_passage[0]['text'] == '':
                valid_answers_passage = [valid_answers_passage[0]]

            doc_level_answers.append(valid_answers_passage)

        data = {'doc_question_pairID': doc_quest_answers['doc_question_pairID'],
                'tagName': doc_quest_answers['tagName'],
                'document_context': doc_quest_answers['document_context'],
                'gold_answers': doc_quest_answers['answers'],
                'predicted_answers_divided_per_passage': doc_level_answers,
                'predicted_answers_doc_level': merge_passage_answers.merge_predictions_from_passages(doc_level_answers)
                }
        pred_answers_for_a_quest_on_doc_list.append(data)

    if fold_name == 'test' or batch_step % print_json_every_batch_steps == 0:
        with open('./predictions/predictions_batch_step_{}.json'.format(batch_step), 'w', encoding='utf-8') as f:
            json.dump(pred_answers_for_a_quest_on_doc_list, f, ensure_ascii=False, indent=4)

    return pred_answers_for_a_quest_on_doc_list


def run_inference(model, dataloader):
    model.eval()
    current_step_loss_eval = 0  # total loss on validation fold at reached batch step

    # initializing lists to store prediction outputs
    # first we run all passages through model, then we merge passage answers to document level and eventually we evaluate metrics
    all_start_logits = []
    all_end_logits = []
    input_ids_list = []
    passage_id_list = []
    sequence_ids_list = []
    offset_mapping_list = []

    for eval_batch in dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=eval_batch['input_ids'],
                attention_mask=eval_batch['attention_mask'],
                start_positions=eval_batch['start_positions'],
                end_positions=eval_batch['end_positions'],
                sequence_ids=eval_batch['sequence_ids']
            )

            loss = outputs.loss
            current_step_loss_eval += loss

            all_start_logits.extend(row for row in outputs.start_logits.cpu())
            all_end_logits.extend(row for row in outputs.end_logits.cpu())

            input_ids_list.extend([i.cpu() for i in eval_batch.get('input_ids')])
            passage_id_list.extend([i for i in eval_batch.get('passage_id')])
            sequence_ids_list.extend([i.cpu() for i in eval_batch.get('sequence_ids')])
            offset_mapping_list.extend([i.cpu() for i in eval_batch.get('offset_mapping')])

    # overall validation set loss
    current_step_loss_eval /= len(dataloader)

    # overall validations set metrics computation
    model_outputs_for_metrics = {
        'all_start_logits': all_start_logits,
        'all_end_logits': all_end_logits,
        'input_ids_list': input_ids_list,
        'passage_id_list': passage_id_list,
        'sequence_ids_list': sequence_ids_list,
        'offset_mapping_list': offset_mapping_list
    }

    return {"loss": current_step_loss_eval, "model_outputs": model_outputs_for_metrics}
