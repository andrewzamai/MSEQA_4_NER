""" EVALUATE a MS-EQA model on a Dataset's test fold """

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from functools import partial
import transformers
import sys
import os

# my libraries
from models.MultiSpanRobertaQuestionAnswering import MultiSpanRobertaQuestionAnswering
from collator_MSEQA import collate_fn_MSEQA
import inference_EQA_MS
import metrics_EQA_MS


if __name__ == '__main__':

    """ -------------------- Evaluation parameters -------------------- """

    """
    from data_handlers import data_handler_pileNER as data_handler_MSEQA_dataset

    path_to_dataset_MSEQA_format = './datasets/pileNER/min_occur_100_MSEQA'
    tokenizer_to_use = "roberta-base"
    path_to_model = "./finetunedModels/MSEQA_pileNER_min_occ_100"

    MAX_SEQ_LENGTH = 256  # question + context + special tokens
    DOC_STRIDE = 64  # overlap between 2 consecutive passages from same document
    MAX_QUERY_LENGTH = 48  # not used, but questions must not be too long given a chosen DOC_STRIDE

    EVAL_BATCH_SIZE = 128

    MAX_ANS_LENGTH_IN_TOKENS = 10
    """

    from data_handlers import data_handler_cross_NER as data_handler_MSEQA_dataset

    subdataset_name = 'music'
    path_to_cross_NER_datasets = './datasets/crossNER/BIO_format'
    path_to_subdataset_guidelines = f"./MSEQA_4_NER/data_handlers/questions/crossNER/{subdataset_name}_NE_definitions.json"

    path_to_dataset_MSEQA_format = './datasets/crossNER/MSEQA_format_guidelines/music'
    tokenizer_to_use = "roberta-base"
    path_to_model = "./finetunedModels/MSEQA_pileNER_prefix_pretrained_bb_CP"

    MAX_SEQ_LENGTH = 512  # question + context + special tokens
    DOC_STRIDE = 50  # overlap between 2 consecutive passages from same document
    MAX_QUERY_LENGTH = 150  # not used, but questions must not be too long given a chosen DOC_STRIDE

    EVAL_BATCH_SIZE = 64

    MAX_ANS_LENGTH_IN_TOKENS = 10

    print(f"Evaluating MS-EQA model named {path_to_model.split('/')[-1]} on {path_to_dataset_MSEQA_format.split('/')[-1]} test set\n")

    print("Loading train/validation/test Datasets in MS-EQA format...")
    if not os.path.exists(path_to_dataset_MSEQA_format):
        print(" ...building Datasets from huggingface repository in MS-EQA format")
        # dataset_MSEQA_format = data_handler_MSEQA_dataset.build_dataset_MSEQA_format()
        dataset_MSEQA_format = data_handler_MSEQA_dataset.build_dataset_MSEQA_format_with_guidelines(path_to_cross_NER_datasets, subdataset_name, path_to_subdataset_guidelines)
        # removing outliers
        # dataset_MSEQA_format = data_handler_MSEQA_dataset.remove_outlier_ne_types(dataset_MSEQA_format, 100)
        dataset_MSEQA_format.save_to_disk(path_to_dataset_MSEQA_format)
    else:
        print(" ...using already existing Datasets in MS-EQA format")
        dataset_MSEQA_format = DatasetDict.load_from_disk(path_to_dataset_MSEQA_format)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    MODEL_CONTEXT_WINDOW = tokenizer.model_max_length
    print(f"Pretrained model relying on: {path_to_model} has context window size of {MODEL_CONTEXT_WINDOW}")
    assert MAX_SEQ_LENGTH <= MODEL_CONTEXT_WINDOW, "MAX SEQ LENGTH must be smallerEqual than model context window"
    print(f"MAX_SEQ_LENGTH used to chunk documents: {MAX_SEQ_LENGTH}")
    assert DOC_STRIDE < (MAX_SEQ_LENGTH - MAX_QUERY_LENGTH), "DOC_STRIDE must be smaller, otherwise parts of the doc will be skipped"
    print("DOC_STRIDE used: {}".format(DOC_STRIDE))

    ''' ------------------ PREPARING MODEL & DATA FOR EVALUATION ------------------ '''

    print("\nPREPARING MODEL and DATA FOR EVALUATION ...")

    print("BATCH_SIZE for evaluation: {}".format(EVAL_BATCH_SIZE))

    test_dataloader = DataLoader(
        dataset_MSEQA_format['test'],
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=partial(collate_fn_MSEQA, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE)
    )

    # loading MS-EQA model with weights pretrained on SQuAD2
    model = MultiSpanRobertaQuestionAnswering.from_pretrained(path_to_model)

    accelerator = Accelerator(cpu=False, mixed_precision='fp16')
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

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
