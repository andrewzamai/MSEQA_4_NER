"""
--- Train a Multi-Span Extractive QA model on SQUAD2 dataset (single-span QA dataset) ---

1) implement a data_handler that converts dataset to MS-EQA format:
    --> DatasetDict with 3 Dataset partitions train/validation/test
each item in the Dataset must have features:
    - doc_question_pairID: a str "docID:questionID_on_that_doc"
    - document_context: the passage of text
    - tagName: the NE category being extracted
    - question: the question
    - answers: a dict {'answer_start': [], 'text': []}
The answer_start are the starting positions in characters from the beginning of the context.

2) import the data_handler and run this script

Finetuning a MSEQA model on SQUAD2 (which is instead a single-span EQA dataset)
Despite the MS_EQA bulding up from already pretrained weights on SQUAD2,
we need to finetune the weights to be able to extract, through the multispan extraction algorithm,
also single span answers.
The new extraction algorithm works differently:
In the single-span setting we were scoring the top answer against a list of gold answers,
but the top answer was "hard" picked from the best candidates (controlled by hyperparams)
In the multi-span setting we automatically let emerge only good spans of text (sum of logits > 0),
so we need to fine-tune the model again on squad-2 to learn to do single-span questions with the same multi-span extraction algorithm,
in order to not switch manually between single-span e multi-span extraction algorithms

The squad evaluation script cannot be used anymore and we develop our metrics (more strict also on char interval == )
"""

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from functools import partial
import transformers
import sys
import os

# my libraries
# from models.MultiSpanRobertaQuestionAnswering import MultiSpanRobertaQuestionAnswering
from models.MultiSpanRobertaQuestionAnswering_from_scratch import MultiSpanRobertaQuestionAnswering_from_scratch
from utils.EarlyStopping import EarlyStopping
from collator_MSEQA import collate_fn_MSEQA
import inference_EQA_MS
import metrics_EQA_MS

from utils import officialSQUADevalscript_mod


if __name__ == '__main__':

    """ -------------------- Training parameters -------------------- """

    # training MSEQA on SQUAD2 dataset (single-span dataset but with Multi-Span model)

    from data_handlers import data_handler_squad2 as data_handler_MSEQA_dataset

    path_to_dataset_MSEQA_format = './datasets/squad2/squad2_MSEQA'  # if already existing, otherwise will be saved when first created
    path_to_dataset_NER_format = None
    path_to_questions = None

    tokenizer_to_use = "roberta-base"
    pretrained_model_relying_on = "roberta-base"

    name_finetuned_model = "MSEQA_on_squad2"

    MAX_SEQ_LENGTH = 256  # question + context + special tokens
    DOC_STRIDE = 96  # overlap between 2 consecutive passages from same document
    MAX_QUERY_LENGTH = 48  # not used, but questions must not be too long given a chosen DOC_STRIDE, based on squad2 train max question length

    BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64

    learning_rate = 3e-5
    num_train_epochs = 20
    warmup_ratio = 0.2

    EARLY_STOPPING_PATIENCE = 5
    EVALUATE_EVERY_N_STEPS = 1000
    EARLY_STOPPING_ON_F1_or_LOSS = False  # True means ES on metrics, False means ES on loss
    GRADIENT_ACCUMULATION_STEPS = 1

    MAX_ANS_LENGTH_IN_TOKENS = 40  # squad2 max answer length in tokens in train fold

    EVALUATE_ZERO_SHOT = False

    """ -------------------- Loading Datasets in MS-EQA format -------------------- """

    print(f"Training MS-EQA model on {path_to_dataset_MSEQA_format.split('/')[-1]} dataset\n")

    print("Loading train/validation/test Datasets in MS-EQA format...")
    if not os.path.exists(path_to_dataset_MSEQA_format):
        print(" ...building Datasets from huggingface repository in MS-EQA format")
        dataset_MSEQA_format = data_handler_MSEQA_dataset.build_dataset_MSEQA_format()
        dataset_MSEQA_format.save_to_disk(path_to_dataset_MSEQA_format)
    else:
        print(" ...using already existing Datasets in MS-EQA format")
        dataset_MSEQA_format = DatasetDict.load_from_disk(path_to_dataset_MSEQA_format)

    print(f"\nMS-EQA model relies on pre-trained model: {pretrained_model_relying_on}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    MODEL_CONTEXT_WINDOW = tokenizer.model_max_length
    print(f"Pretrained model relying on: {pretrained_model_relying_on} has context window size of {MODEL_CONTEXT_WINDOW}")
    assert MAX_SEQ_LENGTH <= MODEL_CONTEXT_WINDOW, "MAX SEQ LENGTH must be smallerEqual than model context window"
    print(f"MAX_SEQ_LENGTH used to chunk documents: {MAX_SEQ_LENGTH}")
    assert DOC_STRIDE < (MAX_SEQ_LENGTH - MAX_QUERY_LENGTH), "DOC_STRIDE must be smaller, otherwise parts of the doc will be skipped"
    print("DOC_STRIDE used: {}".format(DOC_STRIDE))

    ''' ------------------ PREPARING MODEL & DATA FOR TRAINING ------------------ '''

    print("\nPREPARING MODEL and DATA FOR TRAINING ...")

    print("BATCH_SIZE for training: {}".format(BATCH_SIZE))
    print("BATCH_SIZE for evaluation: {}".format(EVAL_BATCH_SIZE))

    train_dataloader = DataLoader(
        dataset_MSEQA_format['train'],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
    )

    eval_dataloader = DataLoader(
        dataset_MSEQA_format['validation'],
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
    )

    test_dataloader = DataLoader(
        dataset_MSEQA_format['test'],
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
    )

    # loading MS-EQA model with weights pretrained on SQuAD2
    model = MultiSpanRobertaQuestionAnswering_from_scratch.from_pretrained(pretrained_model_relying_on)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    accelerator = Accelerator(cpu=False, mixed_precision='fp16')
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_ratio * num_training_steps,
        num_training_steps=num_training_steps,
    )

    ''' ------------------ ZERO-SHOT EVALUATION ------------------ '''
    if EVALUATE_ZERO_SHOT:
        print(f"\nZERO-SHOT EVALUATION ON TEST SET ...\n")
        # run inference through the model
        current_step_loss_eval, model_outputs_for_metrics = inference_EQA_MS.run_inference(model, test_dataloader).values()
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
        micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list)
        print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(micro_metrics['precision'] * 100,
                                                                     micro_metrics['recall'] * 100,
                                                                     micro_metrics['f1'] * 100))
        # running official evaluation script (softer text level match against list of gold answers)
        print("\nManually picking 1st top answer and using official squad2 evaluation script: \n")
        predictions_squad_for_official_eval_script = metrics_EQA_MS.get_predictions_squad_for_official_eval_script(question_on_document_predicted_answers_list)
        officialSQUADevalscript_mod.run_personalized_evaluation('./datasets/squad2/dev-v2_gold_answers.json',
                                                                predictions_squad_for_official_eval_script)

    ''' ------------------ TRAINING WITH EARLY STOPPING ON METRICS/LOSS after N_BATCH_STEPS ------------------ '''

    # per epoch
    training_loss_over_epochs = []
    # every N BATCH STEPS
    validation_loss_every_N_batch_steps = []
    metrics_per_question_on_validation_every_N_batch_steps = []
    metrics_overall_on_validation_every_N_batch_steps = []

    if EARLY_STOPPING_ON_F1_or_LOSS:
        print("\nTRAINING MODEL with EARLY STOPPING on F1 METRIC (maximizing) ...")
    else:
        print("\nTRAINING MODEL with EARLY STOPPING on LOSS (minimizing) ...")

    if EARLY_STOPPING_ON_F1_or_LOSS:
        early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE, mode='max', save_best_weights=True)
    else:
        early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE, mode='min', save_best_weights=True)

    global_steps_counter = 0  # the global step number across the training epochs
    early_stopping_occurred = False
    gradient_accumulation_count = 0

    for epoch in range(1, num_train_epochs + 1):
        print("\n-----------------------------\n")
        print(f"EPOCH {epoch} training ...")
        sys.stdout.flush()

        epoch_loss_train = 0
        for step, batch in enumerate(train_dataloader):
            if early_stopping_occurred:
                break
            model.train()
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            start_positions=batch['start_positions'],
                            end_positions=batch['end_positions'],
                            sequence_ids=batch['sequence_ids'])
            loss = outputs.loss
            epoch_loss_train += loss

            gradient_accumulation_count += 1

            # perform optimization after accumulation_steps
            if gradient_accumulation_count == GRADIENT_ACCUMULATION_STEPS:
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                gradient_accumulation_count = 0  # reset accumulation count

            # evaluate every N BATCH STEPS (if not first batch step)
            if global_steps_counter % EVALUATE_EVERY_N_STEPS == 0 and global_steps_counter != 0:
                print(f"\nBATCH STEP {global_steps_counter} reached: EVALUATING ON VALIDATION SET ...\n")

                current_step_loss_eval, model_outputs_for_metrics = inference_EQA_MS.run_inference(model,
                                                                                                   eval_dataloader).values()

                print("BATCH STEP {} validation loss {:.7f}".format(global_steps_counter, current_step_loss_eval))
                validation_loss_every_N_batch_steps.append(current_step_loss_eval.cpu())

                # extracting answers from model output logits per passage and aggregating them per document level
                question_on_document_predicted_answers_list = inference_EQA_MS.extract_answers_per_passage_from_logits(
                    max_ans_length_in_tokens=MAX_ANS_LENGTH_IN_TOKENS,
                    batch_step=global_steps_counter,
                    print_json_every_batch_steps=EVALUATE_EVERY_N_STEPS * 4,
                    fold_name="validation",
                    tokenizer=tokenizer,
                    datasetdict_MSEQA_format=dataset_MSEQA_format,
                    model_outputs_for_metrics=model_outputs_for_metrics)

                # computing metrics
                print("\nAutomatically emerged answers (sum of logits > 0) with metrics_EQA_MS (strict == on char intervals):")
                micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list)
                print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(micro_metrics['precision'] * 100,
                                                                             micro_metrics['recall'] * 100,
                                                                             micro_metrics['f1'] * 100))
                # metrics_per_question_on_validation_every_N_batch_steps.append(metrics_per_question)
                metrics_overall_on_validation_every_N_batch_steps.append(micro_metrics)
                # running official evaluation script (softer text level match against list of gold answers)
                print("\nManually picking 1st top answer and using official squad2 evaluation script: \n")
                predictions_squad_for_official_eval_script = metrics_EQA_MS.get_predictions_squad_for_official_eval_script(question_on_document_predicted_answers_list)
                officialSQUADevalscript_mod.run_personalized_evaluation('./datasets/squad2/dev-v2_gold_answers.json', predictions_squad_for_official_eval_script)

                sys.stdout.flush()

                early_stopping_on = micro_metrics['f1'] if EARLY_STOPPING_ON_F1_or_LOSS else current_step_loss_eval
                if early_stopping.step(early_stopping_on, model):
                    early_stopping_occurred = True
                    # if stopping, load best model weights
                    model.load_state_dict(early_stopping.best_weights)
                    print("\n-----------------------------------\n")
                    print("Early stopping occurred at global step count: {}".format(global_steps_counter))
                    print("Retrieving best model weights from step count: {}".format(
                        global_steps_counter - EARLY_STOPPING_PATIENCE * EVALUATE_EVERY_N_STEPS))
                    model.save_pretrained(os.path.join("./finetunedModels", name_finetuned_model), from_pt=True)

            global_steps_counter += 1

        if early_stopping_occurred:
            break

        epoch_loss_train = epoch_loss_train / len(train_dataloader)
        training_loss_over_epochs.append(epoch_loss_train.cpu())

        print('\nEPOCH {} training loss: {:.7f}'.format(epoch, epoch_loss_train))
        sys.stdout.flush()

        model.save_pretrained(os.path.join("./finetunedModels", name_finetuned_model + "_CP"), from_pt=True)

    # saving as pickle training logs
    losses_trends = {'name_finetuned_model': name_finetuned_model,
                     'training_loss_over_epochs': training_loss_over_epochs,
                     'validation_loss_every_N_batch_steps': validation_loss_every_N_batch_steps,
                     'metrics_per_question_on_validation_every_N_batch_steps': metrics_per_question_on_validation_every_N_batch_steps,
                     'metrics_overall_on_validation_every_N_batch_steps': metrics_overall_on_validation_every_N_batch_steps
                     }

    """ -------------------- TEST SET evaluation -------------------- """

    print(f"\n\nEVALUATING ON TEST SET ...\n")
    # run inference through the model
    current_step_loss_eval, model_outputs_for_metrics = inference_EQA_MS.run_inference(model, test_dataloader).values()
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
    micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list)
    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(micro_metrics['precision'] * 100,
                                                                 micro_metrics['recall'] * 100,
                                                                 micro_metrics['f1'] * 100))

    print("\nManually picking 1st top answer and using official squad2 evaluation script: \n")
    predictions_squad_for_official_eval_script = metrics_EQA_MS.get_predictions_squad_for_official_eval_script(question_on_document_predicted_answers_list)
    officialSQUADevalscript_mod.run_personalized_evaluation('./datasets/squad2/dev-v2_gold_answers.json',
                                                            predictions_squad_for_official_eval_script)

    print("\nDONE :)")
    sys.stdout.flush()
