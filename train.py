"""
Train from scratch or fine-tune a Multi-Span Extractive Question Answering (MS-EQA) model for Named Entity Recognition (NER) tasks.

1) Data Handling:
   - Implement a data_handler responsible for converting a NER dataset in BIO format to MS-EQA format.
   - MSEQA_dataset is a DatasetDict with three partitions: train, validation, and test.
   - Each sample in each Dataset must contain the following features:
        - doc_question_pairID: a string in the format "fold:docID:questionID_on_that_doc"
        - document_context: the passage of text (any length)
        - tagName: the named entity category being extracted
        - question: the question to extract the desired NE
        - answers: a dictionary {'answer_start': [], 'text': []}
          - answer_start: a list of starting positions in characters from the beginning of the context.

2) Usage:
   - Import the data_handler module and execute this script.
"""


from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from functools import partial
import transformers
import torch
import sys
import os

# my libraries
from preprocess_MSEQA import tokenize_and_preprocess
from utils.EarlyStopping import EarlyStopping
import inference_EQA_MS
import metrics_EQA_MS


if __name__ == '__main__':

    """ -------------------- Training parameters -------------------- """

    print("Training parameters:\n")

    dataset_name = 'pileNER'

    # pre-training on universalNER gpt conversations (i.e. pileNER corpus)
    from data_handlers import data_handler_pileNER as data_handler_MSEQA_dataset

    # train from scratch or continue fine-tune of an already existing MSEQA model?
    start_training_from_scratch = True
    print(f"start_training_from_scratch: {start_training_from_scratch}")
    if start_training_from_scratch:
        # to load a MSEQA with only Roberta pre-trained weights, but newly initialized qa_classifier weights
        from models.MultiSpanRobertaQuestionAnswering_from_scratch import MultiSpanRobertaQuestionAnswering_from_scratch as MSEQA_model
    else:
        # to load a MSEQA model with pre-trained weights
        from models.MultiSpanRobertaQuestionAnswering import MultiSpanRobertaQuestionAnswering as MSEQA_model

    # pileNER corpus with def;guidelines as prefix or question 'what describes X in the text?'
    pileNER_dataset_with_def = True
    if pileNER_dataset_with_def:
        path_to_pileNER_definitions_json = './MSEQA_4_NER/data_handlers/questions/pileNER/all_423_NE_definitions.json'
        path_to_dataset_MSEQA_format = './datasets/pileNER/MSEQA_prefix'  # dataset with gpt definitions if already built, otherwise will be built and saved here
    else:
        path_to_dataset_MSEQA_format = './datasets/pileNER/min_occur_100_MSEQA'  # dataset "what describes X in the text?"
    # path_to_dataset_MSEQA_format = './datasets/pileNER/MSEQA_prefix_w_negatives_2'
    print(f"pileNER_dataset_with_def: {pileNER_dataset_with_def}")
    print(f"path_to_dataset_MSEQA_format: {path_to_dataset_MSEQA_format}")

    roberta_base_or_large = 'large'
    pretrained_model_relying_on = f"roberta-{roberta_base_or_large}"
    # pretrained_model_relying_on = "./pretrainedModels/MS_EQA_on_SQUAD2_model_hasansf1_83"
    print(f"pretrained_model_relying_on: {pretrained_model_relying_on}")

    tokenizer_to_use = f"roberta-{roberta_base_or_large}"
    print(f"tokenizer_to_use: {tokenizer_to_use}")

    name_finetuned_model = f"MSEQA_pileNER_{pileNER_dataset_with_def}Def_{roberta_base_or_large}_wgradclip1_lr3"
    print(f"finetuned_model will be saved as: {name_finetuned_model}")

    # TODO: if changing chunking parameters --> re-build tokenized dataset
    MAX_SEQ_LENGTH = 380  # question + context + special tokens
    DOC_STRIDE = 50  # overlap between 2 consecutive passages from same document
    MAX_QUERY_LENGTH = 150  # not used, average prefix length in tokens (task instruction, definition, guidelines)
    print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    print(f"DOC_STRIDE: {DOC_STRIDE}")
    print(f"MAX_QUERY_LENGTH: {MAX_QUERY_LENGTH}")

    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 32
    EVAL_BATCH_SIZE = 64
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"EVAL_BATCH_SIZE: {EVAL_BATCH_SIZE}")

    learning_rate = 3e-5
    num_train_epochs = 1
    lr_scheduler_strategy = 'cosine'
    warmup_ratio = 0.2
    MAX_GRAD_NORM = 1.0
    print(f"learning_rate: {learning_rate}")
    print(f"lr_scheduler_strategy: {lr_scheduler_strategy}")
    print(f"num_train_epochs: {num_train_epochs}")
    print(f"warmup_ratio: {warmup_ratio}")
    print(f"MAX_GRAD_NORM: {MAX_GRAD_NORM}")

    EARLY_STOPPING_PATIENCE = 5
    EVALUATE_EVERY_N_STEPS = 5000
    EARLY_STOPPING_ON_F1_or_LOSS = False  # True means ES on metrics, False means ES on loss
    print(f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
    print(f"EVALUATE_EVERY_N_STEPS: {EVALUATE_EVERY_N_STEPS}")
    print(f"EARLY_STOPPING_ON_F1_or_LOSS: {EARLY_STOPPING_ON_F1_or_LOSS}")

    MAX_ANS_LENGTH_IN_TOKENS = 10  # hyperparameter to change depending on dataset
    print(f"MAX_ANS_LENGTH_IN_TOKENS: {MAX_ANS_LENGTH_IN_TOKENS}")

    EVALUATE_ZERO_SHOT = False  # do zero-shot evaluation on test fold before starting fine-tuning
    print(f"EVALUATE_ZERO_SHOT: {EVALUATE_ZERO_SHOT}")

    METRICS_EVAL_AFTER_N = 0  # compute metrics only after N batch steps --> helps speeding up first evals
    print(f"METRICS_EVAL_AFTER_N: {METRICS_EVAL_AFTER_N}")

    """ -------------------- Loading Datasets in MS-EQA format -------------------- """

    print(f"\n\nTraining MS-EQA model on {path_to_dataset_MSEQA_format.split('/')[-1]} dataset\n")
    print(f"Fine-tuned model will be named: {name_finetuned_model}\n")

    print("Loading train/validation/test Datasets in MS-EQA format...")
    if not os.path.exists(path_to_dataset_MSEQA_format):
        print(" ...building Datasets from huggingface repository in MS-EQA format")
        sys.stdout.flush()

        if pileNER_dataset_with_def:
            dataset_MSEQA_format = data_handler_MSEQA_dataset.build_dataset_MSEQA_format_with_guidelines(path_to_pileNER_definitions_json)
            # dataset_MSEQA_format = data_handler_MSEQA_dataset.add_negative_examples_to_MSEQA_dataset(dataset_MSEQA_format, path_to_pileNER_definitions_json)
        else:
            dataset_MSEQA_format = data_handler_MSEQA_dataset.build_dataset_MSEQA_format()
            # removing outliers
            dataset_MSEQA_format = data_handler_MSEQA_dataset.remove_outlier_ne_types(dataset_MSEQA_format, 100)

        dataset_MSEQA_format.save_to_disk(path_to_dataset_MSEQA_format)
    else:
        print(" ...using already existing Datasets in MS-EQA format")
        dataset_MSEQA_format = DatasetDict.load_from_disk(path_to_dataset_MSEQA_format)

    print(dataset_MSEQA_format)

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

    print("Tokenizing and preprocessing MSEQA dataset for trainining...")

    path_to_already_tokenized_dataset = path_to_dataset_MSEQA_format + '_tokenized_' + roberta_base_or_large + '_' + str(MAX_SEQ_LENGTH) + '_' + str(DOC_STRIDE)
    if not os.path.exists(path_to_already_tokenized_dataset):
        print(" ...tokenizing dataset for training")
        sys.stdout.flush()
        dataset_MSEQA_format_tokenized = dataset_MSEQA_format.map(
            lambda examples_batch: tokenize_and_preprocess(examples_batch, tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE),
            batched=True,
            remove_columns=dataset_MSEQA_format["train"].column_names
        )

        dataset_MSEQA_format_tokenized.save_to_disk(path_to_already_tokenized_dataset)
    else:
        print(" ...using already existing tokenized dataset")
        dataset_MSEQA_format_tokenized = DatasetDict.load_from_disk(path_to_already_tokenized_dataset)

    print(dataset_MSEQA_format_tokenized)

    # Datacollator with padding over already tokenized dataset
    def collate_and_pad_already_tokenized_dataset(in_batch):
        collated_batch = {key: [] for key in in_batch[0].keys()}
        for item in in_batch:
            for key, in_values in item.items():
                collated_batch[key].append(in_values)

        # padding and returning
        max_len = max(len(seq) for seq in collated_batch['offset_mapping'])
        padded_offset_mapping = [seq + [(-1, -1)] * (max_len - len(seq)) for seq in collated_batch['offset_mapping']]

        return {
            'input_ids': pad_sequence([torch.tensor(t) for t in collated_batch['input_ids']], batch_first=True, padding_value=tokenizer.pad_token_id),
            'attention_mask': pad_sequence([torch.tensor(t) for t in collated_batch['attention_mask']], batch_first=True, padding_value=0),
            'start_positions': pad_sequence([torch.tensor(t) for t in collated_batch['start_positions']], batch_first=True, padding_value=0),
            'end_positions': pad_sequence([torch.tensor(t) for t in collated_batch['end_positions']], batch_first=True, padding_value=0),
            'sequence_ids': pad_sequence([torch.tensor(t) for t in collated_batch['sequence_ids']], batch_first=True, padding_value=0),
            'passage_id': collated_batch['passage_id'],
            'offset_mapping': torch.tensor(padded_offset_mapping)
        }

    train_dataloader = DataLoader(
        dataset_MSEQA_format_tokenized['train'],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=collate_and_pad_already_tokenized_dataset
        # collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
    )

    eval_dataloader = DataLoader(
        dataset_MSEQA_format_tokenized['validation'],
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=collate_and_pad_already_tokenized_dataset
        # collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
    )

    test_dataloader = DataLoader(
        dataset_MSEQA_format_tokenized['test'],
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=collate_and_pad_already_tokenized_dataset
        # collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
    )

    # loading MS-EQA model
    model = MSEQA_model.from_pretrained(pretrained_model_relying_on)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler_strategy,
        optimizer=optimizer,
        num_warmup_steps=warmup_ratio * num_training_steps,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, test_dataloader
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
        micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list, dataset_name=dataset_name)
        print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(micro_metrics['precision'] * 100,
                                                                     micro_metrics['recall'] * 100,
                                                                     micro_metrics['f1'] * 100))

        # compute all other metrics
        overall_metrics, metrics_per_tagName = metrics_EQA_MS.compute_all_metrics(question_on_document_predicted_answers_list)

        print("\nOverall metrics (100%):")
        print("\n------------------------------------------")
        for metric_name, value in overall_metrics.items():
            print("{}: {:.2f}".format(metric_name, value * 100))
        print("------------------------------------------\n")

        print("\nMetrics per NE category (100%):\n")
        for tagName, m in metrics_per_tagName.items():
            print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, m['precision'] * 100, m['recall'] * 100, m['f1'] * 100))
            print("------------------------------------------")

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

    for epoch in range(1, num_train_epochs + 1):
        print("\n-----------------------------\n")
        print(f"EPOCH {epoch} training ...")
        sys.stdout.flush()

        epoch_loss_train = 0
        for step, batch in enumerate(train_dataloader):
            if early_stopping_occurred:
                break
            model.train()
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                start_positions=batch['start_positions'],
                                end_positions=batch['end_positions'],
                                sequence_ids=batch['sequence_ids'])
                loss = outputs.loss
                epoch_loss_train += loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if MAX_GRAD_NORM and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()

            # evaluate every N BATCH STEPS (if not first batch step)
            if global_steps_counter % EVALUATE_EVERY_N_STEPS == 0 and global_steps_counter != 0:

                print(f"\nBATCH STEP {global_steps_counter} reached: EVALUATING ON VALIDATION SET ...\n")

                # print learning rate
                current_lr = optimizer.param_groups[0]['lr']
                print("\nCurrent Learning Rate:", current_lr)
                sys.stdout.flush()

                current_step_loss_eval, model_outputs_for_metrics = inference_EQA_MS.run_inference(model, eval_dataloader).values()

                print("\nBATCH STEP {} validation loss {:.7f}".format(global_steps_counter, current_step_loss_eval))
                validation_loss_every_N_batch_steps.append(current_step_loss_eval.cpu())

                if METRICS_EVAL_AFTER_N <= 0:
                    # extracting answers from model output logits per passage and aggregating them per document level
                    question_on_document_predicted_answers_list = inference_EQA_MS.extract_answers_per_passage_from_logits(max_ans_length_in_tokens=MAX_ANS_LENGTH_IN_TOKENS,
                                                                                                                           batch_step=global_steps_counter,
                                                                                                                           print_json_every_batch_steps=EVALUATE_EVERY_N_STEPS*4,
                                                                                                                           fold_name="validation",
                                                                                                                           tokenizer=tokenizer,
                                                                                                                           datasetdict_MSEQA_format=dataset_MSEQA_format,
                                                                                                                           model_outputs_for_metrics=model_outputs_for_metrics)
                    # computing metrics
                    micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list, dataset_name=dataset_name)
                    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(micro_metrics['precision'] * 100, micro_metrics['recall'] * 100, micro_metrics['f1'] * 100))
                    # metrics_per_question_on_validation_every_N_batch_steps.append(metrics_per_question)
                    metrics_overall_on_validation_every_N_batch_steps.append(micro_metrics)

                METRICS_EVAL_AFTER_N -= 1

                sys.stdout.flush()

                early_stopping_on = micro_metrics['f1'] if EARLY_STOPPING_ON_F1_or_LOSS else current_step_loss_eval
                if early_stopping.step(early_stopping_on, model):
                    early_stopping_occurred = True
                    # if stopping, load best model weights
                    model.load_state_dict(early_stopping.best_weights)
                    print("\n-----------------------------------\n")
                    print("Early stopping occurred at global step count: {}".format(global_steps_counter))
                    print("Retrieving best model weights from step count: {}".format(global_steps_counter - EARLY_STOPPING_PATIENCE * EVALUATE_EVERY_N_STEPS))
                    model.save_pretrained(os.path.join("./finetunedModels", name_finetuned_model), from_pt=True)

                # saving a model checkpoint at the end of every evaluation, i.e. every EVALUATE_EVERY_N_STEPS
                model.save_pretrained(os.path.join("./finetunedModels", name_finetuned_model + "_CP"), from_pt=True)

            global_steps_counter += 1

        if early_stopping_occurred:
            break

        epoch_loss_train = epoch_loss_train / len(train_dataloader)
        training_loss_over_epochs.append(epoch_loss_train)

        print('\nEPOCH {} training loss: {:.7f}'.format(epoch, epoch_loss_train))
        sys.stdout.flush()

    # saving as pickle training logs
    losses_trends = {'name_finetuned_model': name_finetuned_model,
                     'training_loss_over_epochs': training_loss_over_epochs,
                     'validation_loss_every_N_batch_steps': validation_loss_every_N_batch_steps,
                     'metrics_per_question_on_validation_every_N_batch_steps': metrics_per_question_on_validation_every_N_batch_steps,
                     'metrics_overall_on_validation_every_N_batch_steps': metrics_overall_on_validation_every_N_batch_steps
                     }

    """ -------------------- TEST SET evaluation -------------------- """

    # if training reaches end of epochs, load best weights up to now and save
    model.load_state_dict(early_stopping.best_weights)
    model.save_pretrained(os.path.join("./finetunedModels", name_finetuned_model + "_ET"), from_pt=True)

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
    micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list, dataset_name=dataset_name)
    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(micro_metrics['precision'] * 100, micro_metrics['recall'] * 100, micro_metrics['f1'] * 100))

    # compute all other metrics
    overall_metrics, metrics_per_tagName = metrics_EQA_MS.compute_all_metrics(question_on_document_predicted_answers_list)

    print("\nOverall metrics (100%):")
    print("\n------------------------------------------")
    for metric_name, value in overall_metrics.items():
        print("{}: {:.2f}".format(metric_name, value * 100))
    print("------------------------------------------\n")

    print("\nMetrics per NE category (100%):\n")
    for tagName, m in metrics_per_tagName.items():
        print("{} --> support: {}".format(tagName, m['tp'] + m['fn']))
        print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(tagName, m['tp'], m['fn'], m['fp'], m['tn']))
        print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, m['precision'] * 100, m['recall'] * 100, m['f1'] * 100))
        print("------------------------------------------")

    print("\nDONE :)")
    sys.stdout.flush()
