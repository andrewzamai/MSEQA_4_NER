"""
Run (T5)-GNER model over converted BUSTER test dataset, collecting predictions.
"""
__package__ = "SOTA_MODELS.GNER"

from datasets import Dataset as hf_Dataset
import sys

from torch.utils.data import DataLoader, Dataset

from MSEQA_4_NER.data_handlers import data_handler_BUSTER
import data_handler_for_GNER

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

from accelerate import Accelerator

class GNERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['instance']['instruction_inputs']

def collate_fn(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    return inputs


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-T5-xxl", cache_dir='./hf_cache_dir')
    model = AutoModelForSeq2SeqLM.from_pretrained("dyyyyyyyy/GNER-T5-xxl", torch_dtype=torch.bfloat16, cache_dir='./hf_cache_dir').cuda()
    model = model.eval()

    BUSTER_BIO = data_handler_BUSTER.loadDataset('./datasets/BUSTER/BIO_format/FULL_KFOLDS_BIO/123_4_5')
    BUSTER_GNER_test = data_handler_for_GNER.convert_test_dataset_for_GNER_inference_sliding_window_chunking(BUSTER_BIO, 100, 15)

    accelerator = Accelerator()

    BUSTER_GNER_test_dataset = GNERDataset(BUSTER_GNER_test)

    dataloader = DataLoader(BUSTER_GNER_test_dataset, batch_size=2, collate_fn=collate_fn)

    # Prepare everything with Accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    BUSTER_GNER_test = BUSTER_GNER_test.to_list()
    # Inference
    predictions = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"{100 * i / len(dataloader)}% completed")
            inputs = batch.to(accelerator.device)
            outputs = model.generate(**inputs, max_new_tokens=640)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_outputs)
            print(decoded_outputs)
            print("--------------\n\n")

    # Add predictions to original data
    for i, sample_GNER in enumerate(BUSTER_GNER_test):
        sample_GNER['prediction'] = predictions[i]

    # Save predictions to a JSONL file
    BUSTER_GNER_test = hf_Dataset.from_list(BUSTER_GNER_test)
    BUSTER_GNER_test.to_json('./predictions/BUSTER_GNER_test_w_T5_preds_bs4_debug.jsonl')

    """
    BUSTER_GNER_test = BUSTER_GNER_test.to_list()

    print(len(BUSTER_GNER_test))
    for i, sample_GNER in enumerate(BUSTER_GNER_test):
        print(f"{100 * i/len(BUSTER_GNER_test)}")
        # For LLaMA Model, instruction part are wrapped with [INST] tag
        # input_texts = f"[INST] {sample_GNER['instance']['instruction_inputs']} [/INST]"
        input_texts = sample_GNER['instance']['instruction_inputs']
        print(input_texts)
        inputs = tokenizer(input_texts, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=640)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        sample_GNER['prediction'] = response
        print("--------------\n\n")

    BUSTER_GNER_test = Dataset.from_list(BUSTER_GNER_test)
    BUSTER_GNER_test.to_json('./predictions/BUSTER_GNER_test_w_T5_preds_2.jsonl')
    """

