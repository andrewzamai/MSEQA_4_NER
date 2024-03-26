"""
Run GoLLIE model over converted BUSTER test dataset, collecting predictions.
Using VLLM for faster inference.
"""

__package__ = "SOTA_MODELS.GoLLIE"

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

    BUSTER_test_GoLLIE = load_dataset(path='json', data_files='./datasets/others/BUSTER_test_GoLLIE.jsonl')['train']
    print(BUSTER_test_GoLLIE)

    vllm_model = LLM(model="HiTZ/GoLLIE-7B", download_dir='./hf_cache_dir')

    max_new_tokens = 256
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])

    prompts = BUSTER_test_GoLLIE['prompt_only']
    # prompts = prompts[1:10]
    BUSTER_test_GoLLIE = BUSTER_test_GoLLIE.to_list()
    responses = vllm_model.generate(prompts, sampling_params)
    for i, response in enumerate(responses):
        response = response.outputs[0].text
        # response = response[response.find("[/INST]") + len("[/INST]"):].strip()
        # print(response)
        BUSTER_test_GoLLIE[i]['prediction'] = response

    BUSTER_test_GoLLIE = Dataset.from_list(BUSTER_test_GoLLIE)

    BUSTER_test_GoLLIE.to_json('./predictions/BUSTER_test_GoLLIE_w_preds.jsonl')
