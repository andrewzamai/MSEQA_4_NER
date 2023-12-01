"""
- merge answers from passages to answer the question at document level -

Each Document is splitt into multiple passages to fit the model CONTEXT WINDOW (particularly the MAX_SEQ_LENGTH).
When making inference through the model we collect a LIST of answers for each passage (with little post-processing);
We now need to aggregate this answers to answer to the question at document level, i.e.:
if 3 passages out of 4 say that there is no answer and the 4th is some text answer
does not mean there is no answer (like voting rule)
but means only that the 3 passages do not contain any span of text answering to that question, but the 4th yes
"""

import json

# personal libraries
import metrics_EQA_MS

""" 
deletes overlapping answers keeping the one with highest score
can be applied both to answers within same passage 
and to all the answers at doc level to clean furthermore the aggregation from passages
"""
def delete_overlaps(list_of_answers):
    # if not at least 2 answers is already cleaned from overlaps
    if len(list_of_answers) <= 1:
        return list_of_answers

    cleaned_answers = [list_of_answers[0]]
    for i in range(1, len(list_of_answers)):
        answer = list_of_answers[i]
        is_new_answer = True  # append as new answer or not
        # check with all already filtered answers if there is some overlap
        for j in range(len(cleaned_answers)):
            overlap = metrics_EQA_MS.compute_overlap_char_indices(answer['start_end_indices'], cleaned_answers[j]['start_end_indices'])
            # print("Number of overlapping character between answers: {}".format(overlap))
            if overlap > 1:
                # if there is some overlap (just of 1 char only) we need to keep only the answer with higher score
                # for sure we do not append it, but we need to check if to substitute the existing one or discard it
                is_new_answer = False
                # remove float() if using no more json file
                candidate_answer_score = float(answer['score']) if isinstance(answer['score'], str) else answer['score']
                to_check_against_score = float(cleaned_answers[j]['score']) if isinstance(cleaned_answers[j]['score'], str) else cleaned_answers[j]['score']
                # if has higher score we substitute it
                if candidate_answer_score > to_check_against_score:
                    cleaned_answers[j] = answer
        # if no overlap found we add it as new non overlapping answer
        if is_new_answer:
            cleaned_answers.append(answer)

    return cleaned_answers


""" merge prediction from passages to doc level """
def merge_predictions_from_passages(answers_divided_per_passage):
    predicted_answers = []
    # from each passage a list of answers is extracted
    for passage_answers in answers_divided_per_passage:
        # if more than 1 answer from passage
        if len(passage_answers) > 1:
            # clean answers overlap within this passage
            cleaned_from_overlaps_answers = delete_overlaps(passage_answers)
            predicted_answers.extend(cleaned_from_overlaps_answers)
        else:
            # not adding no-answers from passage (text: '')
            # neither [], that is: after filtering >0 the passage_answers can be empty array []
            if passage_answers != [] and passage_answers[0]['text'] != '':
                predicted_answers.append(passage_answers[0])  # [0] since still a list with 1 answer only from passage

    # just a passage returning some answer != '' (with some score > T)
    # is sufficient to not have an empty answer to all doc

    # if all passage answers were '' then predicted_answers is now [] and we give '' as answer to all document
    if not predicted_answers:
        return {'score': '15.0', 'text': '', 'start_end_indices': [0, 0]}

    # sorting by decreasing score merged answers
    predicted_answers = sorted(predicted_answers, key=lambda x: float(x['score']), reverse=True)
    # delete again overlaps that could have been created from merging answers from different passages
    # and answer was in the overlapping
    if len(predicted_answers) > 1:
        predicted_answers = delete_overlaps(predicted_answers)

    # minScore = 2.5
    # predicted_answers = list(filter(lambda x: float(x['score']) > minScore, predicted_answers))

    return predicted_answers


if __name__ == "__main__":

    post_processed_predictions = None
    with open('./post_processed_predictions.json', 'r', encoding='utf-8') as f:
        post_processed_predictions = json.load(f)

    for sample_index in range(100):
        # 54, 22
        # sample_index = 19
        doc_question_predictions = post_processed_predictions[sample_index]

        print("\ndoc_question_pairID: ")
        print(doc_question_predictions['doc_question_pairID'])
        print("\nGold answers to question at document level: ")
        print(doc_question_predictions['gold_answers'])

        print("\nPredicted answers for each passage:")
        print("1st: raw list of predictions per passage (already only > 0 and sorted) \n2nd: cleaned from overlaps list\n")
        for passage_answers in doc_question_predictions['predicted_answers']:
            print(passage_answers)
            if len(passage_answers) > 1:
                print(delete_overlaps(passage_answers))
            print("--------------------------------")

        # print(doc_question_predictions['document_context'])

        print("\nPredicted answers after passages aggregation: ")
        for a in merge_predictions_from_passages(doc_question_predictions['predicted_answers']):
            print(a)

        print("\nGold answers: ")
        print(doc_question_predictions['gold_answers'])
