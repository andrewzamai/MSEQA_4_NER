""" EVALUATE T5based-MS-EQA model on a Dataset's test fold """

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from functools import partial
import transformers
import sys
import os

# my libraries
from models.MSEQA_T5_ENCODER import T5EncoderModelForQuestionAnswering
from collator_MSEQA import collate_fn_MSEQA_T5
import inference_EQA_MS
import metrics_EQA_MS

from data_handlers import data_handler_cross_NER
from data_handlers import data_handler_pileNER
from data_handlers import data_handler_BUSTER
from data_handlers import data_handler_MIT

from peft import PeftModel, PeftConfig, PeftModelForQuestionAnswering

def load_or_build_dataset_MSEQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition, load_from_disk=False):

    if subdataset_name == 'pileNER':
        if with_definition:
            return DatasetDict.load_from_disk('./datasets/pileNER/MSEQA_TrueDef')
        else:
            return DatasetDict.load_from_disk('./datasets/pileNER/MSEQA_FalseDef')

    path_to_NER_datasets_BIO_format = f"./datasets/{datasets_cluster_name}/BIO_format"

    path_to_guidelines_folder = f"./MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"
    path_to_datasets_MSEQA_format_folder_with_def = f"./datasets/{datasets_cluster_name}/MSEQA_format_guidelines/"

    path_to_subdataset_questions_folder = f"./MSEQA_4_NER/data_handlers/questions/{datasets_cluster_name}/what_describes_questions"
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


if __name__ == '__main__':

    print("ZERO-SHOT EVALUATIONS:\n")

    to_eval_on = [
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_MIT, 'subdataset_names': ['movie', 'restaurant'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10, 'EVAL_BATCH_SIZE': 64},
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_cross_NER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10, 'EVAL_BATCH_SIZE': 64},
        {'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10, 'EVAL_BATCH_SIZE': 4},
        {'datasets_cluster_name': 'pileNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['pileNER'], 'MAX_SEQ_LENGTH': 380, 'DOC_STRIDE': 50, 'MAX_ANS_LENGTH_IN_TOKENS': 10, 'EVAL_BATCH_SIZE': 32}
    ]

    WITH_DEFINITION = True
    print(f"With definition: {WITH_DEFINITION}")

    tokenizer_to_use = "t5-3b"

    if WITH_DEFINITION:
        #path_to_model = "./baseline_T5/T5_MSEQA_pileNERpt_TrueDef_LORA/checkpoint-1400"
        #path_to_model = "./baseline_T5/T5_MSEQA_pileNERpt_TrueDef_LORA_3b_int8_adamint8/checkpoint-1400"
        #path_to_model = "./baseline_T5/T5_MSEQA_pileNERpt_TrueDef_LORA_3b_int8_adamint8/checkpoint-1800"
        path_to_model = "./baseline_T5/T5_MSEQA_pileNERpt_TrueDef_LORA_3b_int8_adamint8_bs64_stableemb_3epochs/checkpoint-3400"
    else:
        path_to_model = "./baseline_T5/T5_MSEQA_pileNERpt_FalseDef_LORA_3b_int8_adamint8_bs64_stableemb/finetuned_model"

    print(f"Model name: {' '.join(path_to_model.split('/')[-2:])}")

    # loading LORA T5-MS-EQA model
    config = PeftConfig.from_pretrained(path_to_model)
    model = T5EncoderModelForQuestionAnswering.from_pretrained(config.base_model_name_or_path, cache_dir='./hf_cache_dir')
    #model = PeftModel.from_pretrained(model, path_to_model)
    model = PeftModelForQuestionAnswering.from_pretrained(model, path_to_model)

    accelerator = Accelerator(mixed_precision='no')
    model = accelerator.prepare(model)

    for data in to_eval_on:
        for subdataset_name in data['subdataset_names']:
            print(f"\n\nEvaluating MS-EQA model named '{path_to_model.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            MAX_SEQ_LENGTH = data['MAX_SEQ_LENGTH']
            DOC_STRIDE = data['DOC_STRIDE']
            MAX_QUERY_LENGTH = 150 if WITH_DEFINITION else 50

            MAX_ANS_LENGTH_IN_TOKENS = data['MAX_ANS_LENGTH_IN_TOKENS']

            print("\nLoading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use, cache_dir='./hf_cache_dir')

            assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
            MODEL_CONTEXT_WINDOW = tokenizer.model_max_length
            print(f"Pretrained model relying on: {path_to_model} has context window size of {MODEL_CONTEXT_WINDOW}")
            assert MAX_SEQ_LENGTH <= MODEL_CONTEXT_WINDOW, "MAX SEQ LENGTH must be smallerEqual than model context window"
            print(f"MAX_SEQ_LENGTH used to chunk documents: {MAX_SEQ_LENGTH}")
            assert DOC_STRIDE < (MAX_SEQ_LENGTH - MAX_QUERY_LENGTH), "DOC_STRIDE must be smaller, otherwise parts of the doc will be skipped"
            print("DOC_STRIDE used: {}".format(DOC_STRIDE))
            sys.stdout.flush()

            ''' ------------------ PREPARING MODEL & DATA FOR EVALUATION ------------------ '''

            print("\nPREPARING MODEL and DATA FOR EVALUATION ...")
            sys.stdout.flush()

            dataset_MSEQA_format = load_or_build_dataset_MSEQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], WITH_DEFINITION, load_from_disk=True)

            EVAL_BATCH_SIZE = data['EVAL_BATCH_SIZE']
            print("BATCH_SIZE for evaluation: {}".format(EVAL_BATCH_SIZE))
            sys.stdout.flush()

            test_dataloader = DataLoader(
                dataset_MSEQA_format['test'],
                shuffle=False,
                batch_size=EVAL_BATCH_SIZE,
                collate_fn=partial(collate_fn_MSEQA_T5, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
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
            micro_metrics = metrics_EQA_MS.compute_micro_precision_recall_f1(question_on_document_predicted_answers_list, subdataset_name)
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
