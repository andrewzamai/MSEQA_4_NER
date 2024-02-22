"""
Fine-tuning 1 MS-EQA model (RoBERTa based) on a Cross-NER domain specific dataset (e.g. music)

- custom RoBERTa for multi-span with BCEwL loss and new passage predictions extraction alg.
- EARLY STOPPING maximizing on F1 METRIC (Macro-F1) OR minimizing loss (switchable)
- EARLY STOPPING EVERY_N_BATCH_STEPS
"""

import os
import sys

import pickle5 as pickle  # using pickle5 for UNIPD servers

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence


import transformers
from transformers import get_scheduler
from transformers import AutoTokenizer
from accelerate import Accelerator


# my libraries
from models.MultiSpanRobertaQuestionAnswering import MultiSpanRobertaQuestionAnswering
from utils.EarlyStopping import EarlyStopping
import evaluate_metrics_EQA_MS
import data_handler_cross_NER


''' ------------------ Fine-tuning arguments ------------------ '''

path_to_cross_ner_datasets = "./datasets/CrossNER_QA_format/"
cross_ner_dataset_name = "music_describes"
path_to_dataset = os.path.join(path_to_cross_ner_datasets, cross_ner_dataset_name)
#path_to_questions_txt = f"./src/data_handlers/cross_ner_questions/{cross_ner_dataset_name}.txt"
path_to_questions_txt = "./src/data_handlers/cross_ner_questions_improved/music.txt"

tokenizer_to_use = 'roberta-base'

#pretrained_model_relying_on = 'deepset/roberta-base-squad2'
#pretrained_model_relying_on = 'deepset/roberta-large-squad2'
#pretrained_model_relying_on = './pretrainedModels/MS_EQA_on_SQUAD2_model'
#pretrained_model_relying_on = './pretrainedModels/MS_EQA_on_SQUAD2_model_hasansf1_83'
#pretrained_model_relying_on = './pretrainedModels/BUSTER'
#pretrained_model_relying_on = './pretrainedModels/roberta-base-squad2_conll2003'
pretrained_model_relying_on = './pretrainedModels/MS_EQA_uniNER_pt'


print(f"Fine-tuning MS-EQA model on Cross-NER sub-dataset {cross_ner_dataset_name}\n\n")


''' ------------------ LOADING TOKENIZER ------------------ '''

print(f"Custom MS-EQA model relies on pre-trained model: {pretrained_model_relying_on}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

MODEL_CONTEXT_WINDOW = tokenizer.model_max_length
print(f"Pretrained model relying on: {pretrained_model_relying_on} has context window size of {MODEL_CONTEXT_WINDOW}")
MAX_SEQ_LENGTH = 256  # question + context + special tokens
assert MAX_SEQ_LENGTH <= MODEL_CONTEXT_WINDOW, "MAX SEQ LENGTH must be smallerEqual than model context window"
print(f"MAX_SEQ_LENGTH used to chunk documents: {MAX_SEQ_LENGTH}")
DOC_STRIDE = 64  # overlap between 2 consecutive passages from same document
MAX_QUERY_LENGTH = 48  # not used but one should check that its questions are not too long given a chosen DOC_STRIDE
assert DOC_STRIDE < (MAX_SEQ_LENGTH - MAX_QUERY_LENGTH), "DOC_STRIDE must be smaller than ..., otherwise parts of the doc will be skipped"
print("DOC_STRIDE used: {}".format(DOC_STRIDE))


''' ------------------ LOADING already TOKENIZED and CHUNKED DATASETS ------------------ '''

print("\nLOADING already tokenized and chunked train/validation/test datasets ...")

# loading already tokenized datasets
with open(os.path.join(path_to_cross_ner_datasets, cross_ner_dataset_name, 'tokenized_datasets.pickle'), 'rb') as handle:
    tokenized_datasets = pickle.load(handle)

# dataset where each sample is a triple (document, question, goldAnswers to that question for this document)
with open(os.path.join(path_to_cross_ner_datasets, cross_ner_dataset_name, 'dataset_doc_quest_ans.pickle'), 'rb') as handle:
    dataset_doc_quest_ans = pickle.load(handle)

# loading dict mapping original str IDs to new float IDS (used when batching by Datacollator)
with open(os.path.join(path_to_cross_ner_datasets, cross_ner_dataset_name, 'strIDs_to_floatIDs.pickle'), 'rb') as handle:
    strIDs_to_floatIDs = pickle.load(handle)

# inverting dict, to map floadIDs to original strIDs
floatIDs_to_strIDs = {splitName: {} for splitName in strIDs_to_floatIDs.keys()}
for splitName in strIDs_to_floatIDs.keys():
    floatIDs_to_strIDs[splitName] = {v: k for k, v in strIDs_to_floatIDs[splitName].items()}


''' ------------------ PREPARING MODEL & DATA FOR TRAINING ------------------ '''

print("\nPREPARING MODEL and DATA FOR TRAINING ...")
BATCH_SIZE = 32
print("BATCH_SIZE used: {}".format(BATCH_SIZE))
EVAL_BATCH_SIZE = 128
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

# Custom collate function to pad sequences within DataLoader
def collate_fn(batch):
    passage_id = [item['passage_id'] for item in batch]
    offset_mapping = [list(item['offset_mapping'].numpy()) for item in batch]

    max_len = max(len(seq) for seq in offset_mapping)
    padded_offset_mapping = [seq + [(-1, -1)] * (max_len - len(seq)) for seq in offset_mapping]

    return {
        'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
        'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0),
        'start_positions': pad_sequence([item['start_positions'] for item in batch], batch_first=True, padding_value=0),
        'end_positions': pad_sequence([item['end_positions'] for item in batch], batch_first=True, padding_value=0),
        'sequence_ids': pad_sequence([item['sequence_ids'] for item in batch], batch_first=True, padding_value=0),
        'passage_id': torch.tensor(passage_id),
        'offset_mapping': torch.tensor(padded_offset_mapping)
    }


# loading MS-EQA model with weights pretrained on SQUAD2
model = MultiSpanRobertaQuestionAnswering.from_pretrained(pretrained_model_relying_on)

tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets['train'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

# we don't care about being shuffled since metrics are on all document and not passages
# we need to make inference over all passages, collect answers per document and score metrics
eval_dataloader = DataLoader(
    tokenized_datasets['validation'],
    shuffle=False,
    batch_size=EVAL_BATCH_SIZE,
    collate_fn=collate_fn
)

test_dataloader = DataLoader(
    tokenized_datasets['test'],
    shuffle=False,
    batch_size=EVAL_BATCH_SIZE,
    collate_fn=collate_fn
)

optimizer = AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator(cpu=False, mixed_precision='fp16')
model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader
)

num_train_epochs = 20
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0.05 * num_training_steps,
    num_training_steps=num_training_steps,
)

# loading tagName:question dict
tagName_question_dict = data_handler_cross_NER.load_questions_from_txt(path_to_questions_txt)

# zero-shot evaluation
print("\nZero-shot evaluation on TEST set:\n")
model.eval()
# initializing lists to store prediction outputs
# first we run all passage through model, then we evaluate metrics merging passage answers to document level
all_start_logits = []
all_end_logits = []
input_ids_list = []
passage_id_list = []
sequence_ids_list = []
offset_mapping_list = []

for batch in test_dataloader:
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        start_positions=batch['start_positions'],
                        end_positions=batch['end_positions'],
                        sequence_ids=batch['sequence_ids']
        )

        all_start_logits.extend(row for row in outputs.start_logits.cpu())
        all_end_logits.extend(row for row in outputs.end_logits.cpu())

        input_ids_list.extend([i.cpu() for i in batch.get('input_ids')])
        passage_id_list.extend([i.cpu() for i in batch.get('passage_id')])
        sequence_ids_list.extend([i.cpu() for i in batch.get('sequence_ids')])
        offset_mapping_list.extend([i.cpu() for i in batch.get('offset_mapping')])

# overall validations set metrics computation
model_outputs_for_metrics = {'all_start_logits': all_start_logits,
                             'all_end_logits': all_end_logits,
                             'input_ids_list': input_ids_list,
                             'passage_id_list': passage_id_list,
                             'sequence_ids_list': sequence_ids_list,
                             'offset_mapping_list': offset_mapping_list
                             }

overall_metrics, metrics_per_question = evaluate_metrics_EQA_MS.compute_metrics(path_to_questions_txt,
                                                                                'zeroshot',
                                                                                'test',
                                                                                tokenizer,
                                                                                dataset_doc_quest_ans,
                                                                                floatIDs_to_strIDs,
                                                                                model_outputs_for_metrics)
average_f1 = overall_metrics['macro_f1']
print("\nMetrics per question:\n")
for tagName, m in metrics_per_question.items():
    print(f"{tagName} : {tagName_question_dict[tagName]}")
    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(m['precision']*100, m['recall']*100, m['f1']*100))
    print("------------------------------------------")

print("\nZero-shot MACRO AVERAGE F1 on test set: {:.2f}\n".format(average_f1*100))
print("Zero-shot micro-F1 on test set: {:.2f}\n".format(overall_metrics["micro_f1"]*100))
sys.stdout.flush()

''' ------------------ TRAINING WITH EARLY STOPPING ON METRICS/LOSS after N_BATCH_STEPS ------------------ '''

# total (=start+end/2) training loss at each batch step
total_loss_train = []
# total (=start+end/2) validation loss at each batch step
total_loss_eval = []
# per epoch
training_loss_over_epochs = []
# every N BATCH STEPS
validation_loss_every_N_batch_steps = []
metrics_per_question_on_validation_every_N_batch_steps = []
metrics_overall_on_validation_every_N_batch_steps = []

EARLY_STOPPING_ON_F1_or_LOSS = False  # True means ES on metrics, False means ES on loss
if EARLY_STOPPING_ON_F1_or_LOSS:
    print("\nTRAINING MODEL with EARLY STOPPING on F1 METRIC (maximizing) ...")
else:
    print("\nTRAINING MODEL with EARLY STOPPING on LOSS (minimizing) ...")

EARLY_STOPPING_PATIENCE = 5
if cross_ner_dataset_name == "conll2003":
    EVALUATE_EVERY_N_STEPS = 500  # default: 250, ES not at the end of each epoch, but every N batch steps
else:
    EVALUATE_EVERY_N_STEPS = 25

if EARLY_STOPPING_ON_F1_or_LOSS:
    early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE, mode='max', save_best_weights=True)
else:
    early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE, mode='min', save_best_weights=True)

global_steps_counter = 0  # the global step number across the training epochs
early_stopping_occurred = False
for epoch in range(1, num_train_epochs+1):
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

        accelerator.backward(loss)
        epoch_loss_train += loss

        # training loss at each batch step
        total_loss_train.append(loss.cpu())

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # evaluate every N BATCH STEPS (if not first batch step)
        if global_steps_counter % EVALUATE_EVERY_N_STEPS == 0 and global_steps_counter != 0:
            model.eval()
            print(f"\nBATCH STEP {global_steps_counter} reached: EVALUATING ON VALIDATION SET ...\n")
            current_step_loss_eval = 0  # total loss on validation fold at reached batch step

            # initializing lists to store prediction outputs
            # first we run all passage through model, then we evaluate metrics merging passage answers to document level
            all_start_logits = []
            all_end_logits = []
            input_ids_list = []
            passage_id_list = []
            sequence_ids_list = []
            offset_mapping_list = []

            for eval_batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(input_ids=eval_batch['input_ids'],
                                    attention_mask=eval_batch['attention_mask'],
                                    start_positions=eval_batch['start_positions'],
                                    end_positions=eval_batch['end_positions'],
                                    sequence_ids=eval_batch['sequence_ids']
                    )

                    loss = outputs.loss
                    current_step_loss_eval += loss
                    total_loss_eval.append(loss.cpu())

                    all_start_logits.extend(row for row in outputs.start_logits.cpu())
                    all_end_logits.extend(row for row in outputs.end_logits.cpu())

                    input_ids_list.extend([i.cpu() for i in eval_batch.get('input_ids')])
                    passage_id_list.extend([i.cpu() for i in eval_batch.get('passage_id')])
                    sequence_ids_list.extend([i.cpu() for i in eval_batch.get('sequence_ids')])
                    offset_mapping_list.extend([i.cpu() for i in eval_batch.get('offset_mapping')])

            # overall validation set loss
            current_step_loss_eval = current_step_loss_eval / len(eval_dataloader)
            print("BATCH STEP {} validation loss {:.7f}\n".format(global_steps_counter, current_step_loss_eval))
            validation_loss_every_N_batch_steps.append(current_step_loss_eval.cpu())

            # overall validations set metrics computation

            model_outputs_for_metrics = {'all_start_logits': all_start_logits,
                                         'all_end_logits': all_end_logits,
                                         'input_ids_list': input_ids_list,
                                         'passage_id_list': passage_id_list,
                                         'sequence_ids_list': sequence_ids_list,
                                         'offset_mapping_list': offset_mapping_list
                                         }

            overall_metrics, metrics_per_question = evaluate_metrics_EQA_MS.compute_metrics(path_to_questions_txt,
                                                                                            global_steps_counter,
                                                                                            'validation',
                                                                                            tokenizer,
                                                                                            dataset_doc_quest_ans,
                                                                                            floatIDs_to_strIDs,
                                                                                            model_outputs_for_metrics)
            average_f1 = overall_metrics['macro_f1']
            print("\nMetrics per question:\n")
            for tagName, m in metrics_per_question.items():
                print(f"{tagName} : {tagName_question_dict[tagName]}")
                print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(m['precision']*100, m['recall']*100, m['f1']*100))
                print("------------------------------------------")
            metrics_per_question_on_validation_every_N_batch_steps.append(metrics_per_question)
            metrics_overall_on_validation_every_N_batch_steps.append(overall_metrics)

            print("\nMACRO AVERAGE F1 on validation set: {:.2f}\n".format(average_f1*100))
            print("micro-F1 on validation set: {:.2f}\n".format(overall_metrics["micro_f1"]*100))
            sys.stdout.flush()

            early_stopping_on = average_f1 if EARLY_STOPPING_ON_F1_or_LOSS else current_step_loss_eval
            if early_stopping.step(early_stopping_on, model):
                early_stopping_occurred = True
                # if stopping, load best model weights
                model.load_state_dict(early_stopping.best_weights)
                print("\n-----------------------------------\n")
                print("Early stopping occurred at global step count: {}".format(global_steps_counter))
                print("Retrieving best model weights from step count: {}".format(global_steps_counter - EARLY_STOPPING_PATIENCE*EVALUATE_EVERY_N_STEPS))
                model.save_pretrained(os.path.join("./finetunedModels", pretrained_model_relying_on.split("/")[-1] + "_" + cross_ner_dataset_name), from_pt=True)

        global_steps_counter += 1

    if early_stopping_occurred:
        break

    epoch_loss_train = epoch_loss_train / len(train_dataloader)
    training_loss_over_epochs.append(epoch_loss_train.cpu())

    print('\nEPOCH {} training loss: {:.7f}'.format(epoch, epoch_loss_train))
    sys.stdout.flush()

    # model.save_pretrained(os.path.join("./finetunedModels", cross_ner_dataset_name + "_CP"), from_pt=True)

# saving as pickle training logs
losses_trends = {'cross_ner_dataset_name': cross_ner_dataset_name,
                 'total_loss_train': total_loss_train,
                 'total_loss_eval': total_loss_eval,
                 'training_loss_over_epochs': training_loss_over_epochs,
                 'validation_loss_every_N_batch_steps': validation_loss_every_N_batch_steps,
                 'metrics_per_question_on_validation_every_N_batch_steps': metrics_per_question_on_validation_every_N_batch_steps,
                 'metrics_overall_on_validation_every_N_batch_steps': metrics_overall_on_validation_every_N_batch_steps
                 }


''' ------------------ METRICS ON TEST SET ------------------ '''
model.eval()
print("\n\nEVALUATING ON TEST SET ... \n")
epoch_loss_test = 0

all_start_logits = []
all_end_logits = []
input_ids_list = []
passage_id_list = []
sequence_ids_list = []
offset_mapping_list = []

for batch in test_dataloader:
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        start_positions=batch['start_positions'],
                        end_positions=batch['end_positions'],
                        sequence_ids=batch['sequence_ids'])

        loss = outputs.loss
        epoch_loss_test += loss

        all_start_logits.extend(row for row in outputs.start_logits.cpu())
        all_end_logits.extend(row for row in outputs.end_logits.cpu())

        input_ids_list.extend([i.cpu() for i in batch.get('input_ids')])
        passage_id_list.extend([i.cpu() for i in batch.get('passage_id')])
        sequence_ids_list.extend([i.cpu() for i in batch.get('sequence_ids')])
        offset_mapping_list.extend([i.cpu() for i in batch.get('offset_mapping')])

epoch_loss_test = epoch_loss_test / len(test_dataloader)
print("Test loss: {:.7f}\n".format(epoch_loss_test))

model_outputs_for_metrics = {'all_start_logits': all_start_logits,
                             'all_end_logits': all_end_logits,
                             'input_ids_list': input_ids_list,
                             'passage_id_list': passage_id_list,
                             'sequence_ids_list': sequence_ids_list,
                             'offset_mapping_list': offset_mapping_list
                             }

overall_metrics, metrics_per_question = evaluate_metrics_EQA_MS.compute_metrics(path_to_questions_txt,
                                                                                "test",
                                                                                'test',
                                                                                tokenizer,
                                                                                dataset_doc_quest_ans,
                                                                                floatIDs_to_strIDs,
                                                                                model_outputs_for_metrics)
print("\nMetrics per question (i.e. per NE category):\n")
for tagName, m in metrics_per_question.items():
    print(f"{tagName} : {tagName_question_dict[tagName]}")
    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(m['precision']*100, m['recall']*100, m['f1']*100))
    print("------------------------------------------")

print("\nOverall metrics:")
print("\n------------------------------------------")
for metric_name, value in overall_metrics.items():
    print("{}: {:.2f}".format(metric_name, value*100))
print("------------------------------------------\n\n\n")

sys.stdout.flush()


with open(f"finetuning_on_{cross_ner_dataset_name}_losses_trends.pickle", 'wb') as handle:
    pickle.dump(losses_trends, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n\nDONE :)")
