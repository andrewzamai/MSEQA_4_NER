"""
FULL-finetuning of a MSEQA model based on Deberta-XXL 1.5b model

- NO LORA, no model quantization, only adamw_8bit optimizer
- Batch size: 16 x ga=16 --> 256; eval_batch_size = 32;
- learning rate 1e-5, cosine sched. over 3 epochs
- training in bf16

- tokenizing all MSEQA dataset before starting training but padding only when collating

1) Data Handling:
   - Implement a data_handler responsible for converting a NER dataset in BIO format to MS-EQA format.
   - MSEQA_dataset is a DatasetDict with three partitions: train, validation, and test.
   - Each sample in each Dataset must contain the following features:
        - doc_question_pairID: a string in the format "[fold:]docID:questionID_on_that_doc"
        - document_context: the passage of text (ANY length)
        - tagName: the named entity category being extracted
        - question: the question to extract the desired tagName
        - answers: a dictionary {'answer_start': [], 'text': []}
          - answer_start: a list of starting positions in characters from the beginning of the context.

2) Usage:
   - Import the data_handler module, set your training parameters and execute this script.
"""

__package__ = "MSEQA_4_NER.training"

from transformers import EarlyStoppingCallback
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_scheduler
from accelerate import Accelerator
import transformers
import torch
import sys
import os

# my libraries
# tokenizer similar to Roberta [CLS] quest [SEP] context [SEP]
from ..preprocess_MSEQA import tokenize_and_preprocess
from .. import inference_EQA_MS
from .. import metrics_EQA_MS


if __name__ == '__main__':

    """ -------------------- Training parameters -------------------- """

    print("Training parameters:\n")

    dataset_name = 'pileNER'

    pretrained_model_relying_on = "microsoft/deberta-v2-xxlarge"
    print(f"pretrained_model_relying_on: {pretrained_model_relying_on}")
    tokenizer_to_use = "microsoft/deberta-v2-xxlarge"
    print(f"tokenizer_to_use: {tokenizer_to_use}")

    # pre-training on universalNER GPT conversations (i.e. pileNER corpus)
    from ..data_handlers import data_handler_pileNER as data_handler_MSEQA_dataset

    # train from scratch or continue fine-tuning an already existing MSEQA model
    start_training_from_scratch = True
    print(f"start_training_from_scratch: {start_training_from_scratch}")
    if start_training_from_scratch:
        # to load a MSEQA with only encoder pre-trained weights, but newly initialized qa_classifier weights
        from ..models.MSEQA_DebertaXXL import DebertaXXLForQuestionAnswering as MSEQA_model
    else:
        # to load a MSEQA model with pre-trained weights
        raise NotImplementedError

    # pileNER corpus with [def;guidelines] as prefix or question 'what describes X in the text?'
    pileNER_dataset_with_def = True
    enhance_TrueDef_training = False  # applies random NE masking and switching to enhance guidelines following
    print(f"enhance_TrueDef_training: {enhance_TrueDef_training}")
    corruption_prob = 0.2
    print(f"corruption_prob: {corruption_prob}")
    masking_prob = 0.8
    print(f"masking_prob: {masking_prob}")
    this_model_mask = '[UNK]'  # depends on tokenizer
    print(f"this_model_mask: {this_model_mask}")

    if pileNER_dataset_with_def:
        path_to_pileNER_definitions_json = './src/MSEQA_4_NER/data_handlers/questions/pileNER/all_423_NE_definitions.json'
        path_to_dataset_MSEQA_format = './datasets/pileNER/MSEQA_TrueDef'  # MSEQA dataset with gpt definitions if it has already been built, otherwise it will be built and stored here
        #path_to_dataset_MSEQA_format = './datasets/pileNER/5_samples_per_NE_MSEQA_TrueDef'
    else:
        path_to_dataset_MSEQA_format = './datasets/pileNER/MSEQA_FalseDef'  # dataset "what describes X in the text?" if already built, otherwise it will be built and stored here
    print(f"pileNER_dataset_with_def: {pileNER_dataset_with_def}")
    print(f"path_to_dataset_MSEQA_format: {path_to_dataset_MSEQA_format}")

    output_dir = f"./trained_models/DeBERTa_MSEQA_pileNERpt_{pileNER_dataset_with_def}Def_{enhance_TrueDef_training}_5pNE"
    print(f"finetuned_model will be saved as: {output_dir}")

    # TODO: if changing chunking parameters --> delete and re-build tokenized dataset (stored and reused to save time)
    MAX_SEQ_LENGTH = 380 if pileNER_dataset_with_def else 280  # question + context + special tokens
    DOC_STRIDE = 50  # overlap between 2 consecutive passages from same document
    MAX_QUERY_LENGTH = 150 if pileNER_dataset_with_def else 50  # not used, average prefix length in tokens (task instruction, definition, guidelines)
    print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    print(f"DOC_STRIDE: {DOC_STRIDE}")
    print(f"MAX_QUERY_LENGTH: {MAX_QUERY_LENGTH}")

    BATCH_SIZE = 16  # >= 32 OOM
    GRADIENT_ACCUMULATION_STEPS = 16
    EVAL_BATCH_SIZE = 32  # >= 64 OOM
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"EVAL_BATCH_SIZE: {EVAL_BATCH_SIZE}")

    learning_rate = 1e-5
    num_train_epochs = 3
    lr_scheduler_strategy = 'cosine'
    warmup_ratio = 0.2
    MAX_GRAD_NORM = 1.0
    print(f"learning_rate: {learning_rate}")
    print(f"lr_scheduler_strategy: {lr_scheduler_strategy}")
    print(f"num_train_epochs: {num_train_epochs}")
    print(f"warmup_ratio: {warmup_ratio}")
    print(f"MAX_GRAD_NORM: {MAX_GRAD_NORM}")

    EARLY_STOPPING_PATIENCE = 5
    EVALUATE_EVERY_N_STEPS = 200  # len(trainDataloader)/10
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

    print(f"\n\nTraining MS-EQA model on {path_to_dataset_MSEQA_format.split('/')[-1]} dataset - {enhance_TrueDef_training}_enhanced \n")
    print(f"Fine-tuned model will be saved in: {output_dir}\n")

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
            # dataset_MSEQA_format = data_handler_MSEQA_dataset.remove_outlier_ne_types(dataset_MSEQA_format, 100) <-- deprecated
            dataset_MSEQA_format = data_handler_MSEQA_dataset.remove_bad_ne_types(dataset_MSEQA_format)

        dataset_MSEQA_format.save_to_disk(path_to_dataset_MSEQA_format)
    else:
        print(" ...using already existing Datasets in MS-EQA format")
        dataset_MSEQA_format = DatasetDict.load_from_disk(path_to_dataset_MSEQA_format)

    if enhance_TrueDef_training:
        dataset_MSEQA_format['train'] = data_handler_MSEQA_dataset.mask_named_entities(dataset_MSEQA_format['train'], corruption_prob, masking_prob, this_model_mask)
        dataset_MSEQA_format['validation'] = data_handler_MSEQA_dataset.mask_named_entities(dataset_MSEQA_format['validation'], corruption_prob, masking_prob, this_model_mask)

    print(dataset_MSEQA_format)

    """
    print("A masked question example: ")
    for sample in dataset_MSEQA_format['train']:
        if 'masked' in sample['doc_question_pairID']:
            print(sample['question'])
            break
    """

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use, cache_dir='./hf_cache_dir')
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

    print(" ...tokenizing dataset for training")
    sys.stdout.flush()
    # ALERT: using tokenizer before training may give parallelism deadlock warnings
    dataset_MSEQA_format_tokenized = dataset_MSEQA_format.map(
        lambda examples_batch: tokenize_and_preprocess(examples_batch, tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE),
        batched=True,
        remove_columns=dataset_MSEQA_format["train"].column_names
    )

    """
    path_to_already_tokenized_dataset = path_to_dataset_MSEQA_format + '_tokenized_' + pretrained_model_relying_on.split('/')[-1] + '_' + str(MAX_SEQ_LENGTH) + '_' + str(DOC_STRIDE)
    if not os.path.exists(path_to_already_tokenized_dataset):
        print(" ...tokenizing dataset for training")
        sys.stdout.flush()
        # ALERT: using tokenizer before training may give parallelism deadlock warnings
        dataset_MSEQA_format_tokenized = dataset_MSEQA_format.map(
            lambda examples_batch: tokenize_and_preprocess(examples_batch, tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE),
            batched=True,
            remove_columns=dataset_MSEQA_format["train"].column_names
        )

        dataset_MSEQA_format_tokenized.save_to_disk(path_to_already_tokenized_dataset)
    else:
        print(" ...using already existing tokenized dataset")
        dataset_MSEQA_format_tokenized = DatasetDict.load_from_disk(path_to_already_tokenized_dataset)
    """

    print(dataset_MSEQA_format_tokenized)

    # Datacollator with padding over already tokenized dataset
    def collate_and_pad_already_tokenized_dataset(in_batch):
        # in batch is a list of dictionaries
        collated_batch = {key: [] for key in in_batch[0].keys()}
        for item in in_batch:
            for key, in_values in item.items():
                collated_batch[key].append(in_values)

        # huggingface Trainer does not pass not used columns by the model in forward
        if 'passage_id' in collated_batch.keys() and 'offset_mapping' in collated_batch.keys():
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
        else:
            return {
                'input_ids': pad_sequence([torch.tensor(t) for t in collated_batch['input_ids']], batch_first=True, padding_value=tokenizer.pad_token_id),
                'attention_mask': pad_sequence([torch.tensor(t) for t in collated_batch['attention_mask']], batch_first=True, padding_value=0),
                'start_positions': pad_sequence([torch.tensor(t) for t in collated_batch['start_positions']], batch_first=True, padding_value=0),
                'end_positions': pad_sequence([torch.tensor(t) for t in collated_batch['end_positions']], batch_first=True, padding_value=0),
                'sequence_ids': pad_sequence([torch.tensor(t) for t in collated_batch['sequence_ids']], batch_first=True, padding_value=0),
            }

    ''' ------------------ PREPARING HF TRAINER ------------------ '''

    # DeBERTa finetuning hyperparams from https://paperswithcode.com/paper/deberta-decoding-enhanced-bert-with/review/

    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        evaluation_strategy="steps",
        eval_steps=EVALUATE_EVERY_N_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # using AdamW optimizer 8 bit
        optim="adamw_bnb_8bit",
        learning_rate=learning_rate,
        weight_decay=0.01,  # to all layers except all bias and LayerNorm weights
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        max_grad_norm=MAX_GRAD_NORM,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_strategy,
        warmup_ratio=warmup_ratio,
        logging_strategy='steps',
        logging_steps=EVALUATE_EVERY_N_STEPS,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        save_steps=EVALUATE_EVERY_N_STEPS,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True
    )

    model = MSEQA_model.from_pretrained(pretrained_model_relying_on, cache_dir='./hf_cache_dir')

    hf_trainer = transformers.Trainer(
        model,
        training_arguments,
        data_collator=collate_and_pad_already_tokenized_dataset,
        train_dataset=dataset_MSEQA_format_tokenized['train'],
        eval_dataset=dataset_MSEQA_format_tokenized['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        # optimizers=(optimizer, lr_scheduler)
    )

    print(hf_trainer.model)

    print("\nStarting training...\n")
    sys.stdout.flush()

    hf_trainer.train()
    hf_trainer.model.save_pretrained(save_directory=os.path.join(output_dir, 'finetuned_model'))

    model = hf_trainer.model

    """ ----------------- EVALUATION on TEST SET ----------------- """

    test_dataloader = DataLoader(
        dataset_MSEQA_format_tokenized['test'],
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=collate_and_pad_already_tokenized_dataset
    )

    accelerator = Accelerator(mixed_precision='bf16')
    test_dataloader = accelerator.prepare(test_dataloader)

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

    """ ----------------- ZERO-SHOT EVALUATIONS ----------------- """

    print("\n\nZERO-SHOT EVALUATIONS:\n")

    from ..data_handlers import data_handler_cross_NER
    from ..data_handlers import data_handler_pileNER
    from ..data_handlers import data_handler_BUSTER
    from ..data_handlers import data_handler_MIT
    from ..collator_MSEQA import collate_fn_MSEQA
    from functools import partial

    def load_or_build_dataset_MSEQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition, load_from_disk=False):

        if subdataset_name == 'pileNER':
            if with_definition:
                return DatasetDict.load_from_disk('./datasets/pileNER/MSEQA_TrueDef')
            else:
                return DatasetDict.load_from_disk('./datasets/pileNER/MSEQA_FalseDef')

        path_to_NER_datasets_BIO_format = f"./datasets/{datasets_cluster_name}/BIO_format"

        path_to_guidelines_folder = f"./src/MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"
        path_to_datasets_MSEQA_format_folder_with_def = f"./datasets/{datasets_cluster_name}/MSEQA_format_guidelines/"

        path_to_subdataset_questions_folder = f"./src/MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/what_describes_questions"
        path_to_dataset_MSEQA_format_folder_no_def = f"./datasets/{datasets_cluster_name}/MSEQA_format_no_def/"

        if with_definition:
            path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')
            path_to_dataset_MSEQA_format = os.path.join(path_to_datasets_MSEQA_format_folder_with_def, subdataset_name)
        else:
            path_to_subdataset_questions = os.path.join(path_to_subdataset_questions_folder, subdataset_name + '.txt')
            path_to_dataset_MSEQA_format = os.path.join(path_to_dataset_MSEQA_format_folder_no_def, subdataset_name)

        print("Loading train/validation/test Datasets in MS-EQA format...")
        if not os.path.exists(path_to_dataset_MSEQA_format) or not load_from_disk:
            print(" ...building Datasets in MS-EQA format")
            sys.stdout.flush()
            if with_definition:
                if subdataset_name == 'BUSTER':
                    dataset_MSEQA_format = data_handler.build_dataset_MSEQA_format_with_guidelines(os.path.join(path_to_NER_datasets_BIO_format, 'FULL_KFOLDS_BIO', '123_4_5'), path_to_subdataset_guidelines)
                else:
                    dataset_MSEQA_format = data_handler.build_dataset_MSEQA_format_with_guidelines(path_to_NER_datasets_BIO_format, subdataset_name, path_to_subdataset_guidelines)
            else:
                if subdataset_name == 'BUSTER':
                    dataset_MSEQA_format = data_handler.build_dataset_MSEQA_format(os.path.join(path_to_NER_datasets_BIO_format, 'FULL_KFOLDS_BIO', '123_4_5'), path_to_subdataset_questions)
                else:
                    dataset_BIO_format = data_handler.build_dataset_from_txt(os.path.join(path_to_NER_datasets_BIO_format, subdataset_name))
                    dataset_MSEQA_format = data_handler.build_dataset_MSEQA_format(dataset_BIO_format, path_to_subdataset_questions)

            dataset_MSEQA_format.save_to_disk(path_to_dataset_MSEQA_format)
        else:
            print(" ...using already existing Datasets in MS-EQA format\n")
            dataset_MSEQA_format = DatasetDict.load_from_disk(path_to_dataset_MSEQA_format)

        return dataset_MSEQA_format

    to_eval_on = [
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_MIT, 'subdataset_names': ['movie', 'restaurant'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_cross_NER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
        {'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10},
        {'datasets_cluster_name': 'pileNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['pileNER'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10}
    ]

    for data in to_eval_on:
        for subdataset_name in data['subdataset_names']:
            print(f"\n\nEvaluating MS-EQA model on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            MAX_SEQ_LENGTH = data['MAX_SEQ_LENGTH']
            DOC_STRIDE = data['DOC_STRIDE']
            MAX_QUERY_LENGTH = 150 if pileNER_dataset_with_def else 50

            MAX_ANS_LENGTH_IN_TOKENS = data['MAX_ANS_LENGTH_IN_TOKENS']

            print("\nLoading tokenizer...")
            # tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use)
            assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
            MODEL_CONTEXT_WINDOW = tokenizer.model_max_length
            print(f"Pretrained model relying on has context window size of {MODEL_CONTEXT_WINDOW}")
            assert MAX_SEQ_LENGTH <= MODEL_CONTEXT_WINDOW, "MAX SEQ LENGTH must be smallerEqual than model context window"
            print(f"MAX_SEQ_LENGTH used to chunk documents: {MAX_SEQ_LENGTH}")
            assert DOC_STRIDE < (MAX_SEQ_LENGTH - MAX_QUERY_LENGTH), "DOC_STRIDE must be smaller, otherwise parts of the doc will be skipped"
            print("DOC_STRIDE used: {}".format(DOC_STRIDE))
            sys.stdout.flush()

            ''' ------------------ PREPARING MODEL & DATA FOR EVALUATION ------------------ '''

            print("\nPREPARING MODEL and DATA FOR EVALUATION ...")
            sys.stdout.flush()

            dataset_MSEQA_format = load_or_build_dataset_MSEQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], pileNER_dataset_with_def, load_from_disk=True)

            EVAL_BATCH_SIZE = EVAL_BATCH_SIZE if data['datasets_cluster_name'] != 'BUSTER' else 4
            if data['datasets_cluster_name'] == 'pileNER':
                EVAL_BATCH_SIZE = 32

            print("BATCH_SIZE for evaluation: {}".format(EVAL_BATCH_SIZE))
            sys.stdout.flush()

            test_dataloader = DataLoader(
                dataset_MSEQA_format['test'],
                shuffle=False,
                batch_size=EVAL_BATCH_SIZE,
                collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
            )

            test_dataloader = accelerator.prepare(test_dataloader)

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
            micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list, dataset_name=subdataset_name)
            print("\n\nmicro (100%) - Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(micro_metrics['precision'] * 100, micro_metrics['recall'] * 100, micro_metrics['f1'] * 100))

            # compute all other metrics
            overall_metrics, metrics_per_tagName = metrics_EQA_MS.compute_all_metrics(question_on_document_predicted_answers_list)

            print("\nOverall metrics (100%):")
            print("\n------------------------------------------")
            for metric_name, value in overall_metrics.items():
                print("{}: {:.2f}".format(metric_name, value * 100))
            print("------------------------------------------\n")

            print("\nMetrics per NE category (100%):\n")
            for tagName, m in metrics_per_tagName.items():
                print("{} --> support: {}".format(tagName, m['tp']+m['fn']))
                print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(tagName, m['tp'], m['fn'], m['fp'], m['tn']))
                print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, m['precision'] * 100, m['recall'] * 100, m['f1'] * 100))
                print("------------------------------------------")

    print("\nDONE :)")
    sys.stdout.flush()
