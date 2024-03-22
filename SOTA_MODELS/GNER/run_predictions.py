"""
Run (LLAMA)-GNER model over converted BUSTER test dataset, collecting predictions.
Using VLLM for faster inference.

Tried beam-search but outputs were truncated after 5/6 tokens.
Using temperature 0 as they do in the notebook
"""
__package__ = "SOTA_MODELS.GNER"

# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams
from datasets import Dataset
import sys

from MSEQA_4_NER.data_handlers import data_handler_BUSTER
import data_handler_for_GNER


if __name__ == '__main__':

    vllm_model = LLM(model="dyyyyyyyy/GNER-LLaMA-7B", download_dir='./hf_cache_dir')

    max_new_tokens = 640  # as they require
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])

    """
    sampling_params = SamplingParams(
        n=1,  # number of output sequences to return for the given prompt,
        best_of=4,
        # from these `best_of` sequences, the top `n` are returned, treated as the beam width when `use_beam_search` is True
        use_beam_search=True,
        early_stopping='never',  # stopping condition for beam search
        temperature=0,
        top_p=1,
        top_k=-1
    )
    """

    BUSTER_BIO = data_handler_BUSTER.loadDataset('./datasets/BUSTER/BIO_format/FULL_KFOLDS_BIO/123_4_5')
    # BUSTER_GNER_test = data_handler_for_GNER.convert_test_dataset_for_GNER_inference(BUSTER_BIO_CHUNKED)
    # using sliding window of 150: instruction itself is 173 tokens ca.
    # with 150 words per chunk we get approximately 412 input tokens
    # in output we expect (412-173)*2 = 478 new tokens which is < max_new_tokens=640
    BUSTER_GNER_test = data_handler_for_GNER.convert_test_dataset_for_GNER_inference_sliding_window_chunking(BUSTER_BIO, 100, 15)

    BUSTER_GNER_test = BUSTER_GNER_test.to_list()

    inputs = []
    for sample_GNER in BUSTER_GNER_test:
        # For LLaMA Model, instruction part are wrapped with [INST] tag
        input_texts = f"[INST] {sample_GNER['instance']['instruction_inputs']} [/INST]"
        inputs.append(input_texts)

    # print(input_texts)

    responses = vllm_model.generate(inputs, sampling_params)
    for i, response in enumerate(responses):
        response = response.outputs[0].text
        # response = response[response.find("[/INST]") + len("[/INST]"):].strip()
        # print(response)
        BUSTER_GNER_test[i]['prediction'] = response
        if i == 0:
            print(BUSTER_GNER_test[0])
            sys.stdout.flush()

    BUSTER_GNER_test = Dataset.from_list(BUSTER_GNER_test)

    BUSTER_GNER_test.to_json('./predictions/BUSTER_GNER_test_sw_100_15_w_preds.jsonl')

    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B", cache_dir='./hf_cache_dir')
    model = AutoModelForCausalLM.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B", torch_dtype=torch.bfloat16, cache_dir='./hf_cache_dir').cuda()

    BUSTER_BIO_CHUNKED = data_handler_BUSTER.loadDataset('./datasets/BUSTER/BIO_format/CHUNKED_KFOLDS/123_4_5')
    BUSTER_GNER_test = data_handler_for_GNER.convert_test_dataset_for_GNER_inference(BUSTER_BIO_CHUNKED)

    for sample_GNER in BUSTER_GNER_test:
        # For LLaMA Model, instruction part are wrapped with [INST] tag
        input_texts = f"[INST] {sample_GNER['instance']['instruction_inputs']} [/INST]"
        print(input_texts)
        inputs = tokenizer(input_texts, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=640)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[response.find("[/INST]") + len("[/INST]"):].strip()
        print(response)
        sample_GNER['prediction'] = response

        break

    BUSTER_GNER_test.to_json('./predictions/BUSTER_GNER_test_w_preds.jsonl')
    """

