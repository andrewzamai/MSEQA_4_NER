"""
Run GoLLIE model over converted BUSTER test dataset, collecting predictions.
Using VLLM for faster inference.
"""

__package__ = "SOTA_MODELS.GoLLIE"

import sys

# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams
from datasets import Dataset, load_dataset

#from MSEQA_4_NER.data_handlers import data_handler_BUSTER
#import data_handler_for_GoLLIE
#from BUSTER_guidelines_GoLLIE import *
#import inspect
#from jinja2 import Template


if __name__ == '__main__':

    """
    BUSTER_BIO = data_handler_BUSTER.loadDataset('./datasets/BUSTER/BIO_format/FULL_KFOLDS_BIO/123_4_5')

    BUSTER_guidelines = [inspect.getsource(definition) for definition in ENTITY_DEFINITIONS]
    print(BUSTER_guidelines)

    with open("../../GoLLIE/templates/prompt.txt", "rt") as f:
        gollie_prompt_template = Template(f.read())

    BUSTER_test_GoLLIE = data_handler_for_GoLLIE.convert_BUSTER_test_dataset_for_GoLLIE(BUSTER_BIO, prompt_template=gollie_prompt_template, guidelines=BUSTER_guidelines)
    print(BUSTER_test_GoLLIE)
    print(BUSTER_test_GoLLIE[0])

    # BUSTER_test_GoLLIE = BUSTER_test_GoLLIE.to_list()
    """

    #BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./datasets/others/BUSTER_test_GoLLIE.jsonl')['train']
    BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./datasets/others/BUSTER_test_GoLLIE_maxLength260.jsonl')['train']
    #BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./datasets/others/BUSTER_test_GoLLIE_maxLength260_noexamples.jsonl')['train']
    print(BUSTER_test_GoLLIE)

    #from transformers import AutoTokenizer
    # Load tokenizer for LLaMA or any other model you're using
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # Set the padding token if it's not already set
    #if tokenizer.pad_token_id is None:
    #tokenizer.pad_token = tokenizer.eos_token  # You can also set it to another token like "<pad>"

    #vllm_model = LLM(model="HiTZ/GoLLIE-7B", download_dir='./hf_cache_dir', max_model_len=4000)
    vllm_model = LLM(model="andrewzamai/GoLLIE-LLaMA2-7B-chat-pileNER391x50", tokenizer="meta-llama/Llama-2-7b-chat-hf", download_dir='./hf_cache_dir', max_model_len=4000, trust_remote_code=True)
    print("Evaluating model andrewzamai/GoLLIE-LLaMA2-7B-chat-pileNER391x50 on BUSTER")

    max_new_tokens = 456
    sampling_params = SamplingParams(temperature=0.6, max_tokens=max_new_tokens, stop=['</s>'])

    prompts = BUSTER_test_GoLLIE['prompt_only']
    print(prompts[0])
    sys.stdout.flush()

    # Convert BUSTER_test_GoLLIE to a list for individual processing
    BUSTER_test_GoLLIE = BUSTER_test_GoLLIE.to_list()

    # Loop over each prompt, generate one response at a time
    for i, prompt in enumerate(prompts):
        single_prompt = [prompt]  # Wrapping in a list since generate expects a list
        response = vllm_model.generate(single_prompt, sampling_params)

        # Extract the response text
        response_text = response[0].outputs[0].text
        BUSTER_test_GoLLIE[i]['prediction'] = response_text

    # Convert back to a Dataset object
    BUSTER_test_GoLLIE = Dataset.from_list(BUSTER_test_GoLLIE)

    """
    prompts = BUSTER_test_GoLLIE['prompt_only']
    # prompts = ['[INST]' + prompt.strip() + '[/INST]' for prompt in prompts]
    print(prompts[0])
    sys.stdout.flush()
    # prompts = prompts[1:10]
    BUSTER_test_GoLLIE = BUSTER_test_GoLLIE.to_list()
    responses = vllm_model.generate(prompts, sampling_params)
    for i, response in enumerate(responses):
        response = response.outputs[0].text
        # response = response[response.find("[/INST]") + len("[/INST]"):].strip()
        # print(response)
        BUSTER_test_GoLLIE[i]['prediction'] = response

    BUSTER_test_GoLLIE = Dataset.from_list(BUSTER_test_GoLLIE)
    
    """

    BUSTER_test_GoLLIE.to_json('./predictions/BUSTER_GoLLIE-LLaMA2-7B-chat-pileNER391x50_maxLength260_t0_sp_buying.jsonl')
