from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, LlamaForCausalLM
from accelerate import Accelerator
import torch
import os

""" ---- BUSTER data_hanlers functions ---- """

def loadDataset(pathToDir):
    data_files = {"train": os.path.join(pathToDir, "train.json"),
                  "test": os.path.join(pathToDir, "test.json"),
                  "validation": os.path.join(pathToDir, "validation.json")}
    return load_dataset("json", data_files=data_files)

def getDocMetadataWithStartEndCharIndexes(documentTokens, documentLabels):
    docMetadata = dict(Parties={"BUYING_COMPANY": [], "SELLING_COMPANY": [], "ACQUIRED_COMPANY": []},
                       Advisors={"CONSULTANT": [], "LEGAL_CONSULTING_COMPANY": [], "GENERIC_CONSULTING_COMPANY": []},
                       Generic_Info={"ANNUAL_REVENUES": []})
    i = 0
    index = 0
    startIndex = index
    entity = ''  # entity being reconstructed
    while i < len(documentLabels):
        # if the token is labelled as part of an entity
        if documentLabels[i] != 'O':
            if entity == '':
                startIndex = index
            entity = entity + ' ' + documentTokens[i]  # this will add an initial space (to be removed)
            # if next label is Other or the beginning of another entity
            # or end of document, the current entity is complete
            if (i < len(documentLabels) - 1 and documentLabels[i + 1][0] in ["O", "B"]) or (i == len(documentLabels) - 1):
                # add to metadata
                tagFamily, tagName = documentLabels[i].split(".")
                # adding also if same name but will have != start-end indices
                docMetadata[tagFamily[2:]][tagName].append((entity[1:], startIndex, startIndex + len(entity[1:])))
                # cleaning for next entity
                entity = ''

        index = index + len(documentTokens[i]) + 1
        i += 1

    return docMetadata


def get_prompt(context, ne_to_extract):
    # from list of words to string if needed
    if isinstance(context, list):
        context = ' '.join(context)
    if '_' in ne_to_extract:
        ne_to_extract = ne_to_extract.lower().split('_')
        ne_to_extract = ' '.join(ne_to_extract)

    initprompt_context = f"A virtual assistant answers questions from a user based on the provided text.\nUSER: Text: {context}"
    prompt_instruction = f"\nASSISTANT: I’ve read this text.\nUSER: What describes {ne_to_extract} in the text?\nASSISTANT:"

    return {'initprompt_context': initprompt_context, 'prompt_instruction': prompt_instruction}


def get_named_entities_for_a_doc(one_sample_BIO):
    # one_sample_BIO = BUSTER_BIO_test[8]
    # print(one_sample_BIO)

    sample_metadata = getDocMetadataWithStartEndCharIndexes(one_sample_BIO['tokens'], one_sample_BIO['labels'])
    print("NEs gold answers: ")
    print(sample_metadata)

    # print(f"This BUSTER document number of words: {len(one_sample_BIO['tokens'])}")

    named_entities = [ne for tagFamily in sample_metadata.keys() for ne in sample_metadata[tagFamily]]
    # print(named_entities)

    prompt_per_ne = {ne: get_prompt(one_sample_BIO['tokens'], ne) for ne in named_entities}
    # print(prompt_per_ne)

    prompt_per_ne_tokenized = {}
    for ne, prompt in prompt_per_ne.items():
        input_tokenized = llama_tokenizer(
            prompt['initprompt_context'],
            prompt['prompt_instruction'],
            truncation='only_first',
            max_length=256,
            stride=100,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,  # not padding here
        )
        #print(input_tokenized)
        #print(type(input_tokenized))
        prompt_per_ne_tokenized[ne] = {key: value[0] for key, value in input_tokenized.items()}

    # for ne, prompt_tokenized in prompt_per_ne_tokenized.items():
        # print(len(prompt_tokenized['input_ids']))
        # print(llama_tokenizer.decode(prompt_tokenized['input_ids']))

    # Process the tokenized inputs and make inference
    results = {}

    for ne, prompt in prompt_per_ne_tokenized.items():
        # Convert input to PyTorch tensors and move to GPU
        input_ids = torch.tensor(prompt['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(prompt['attention_mask']).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = uniNER_llama_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                temperature=1.0,
                num_beams=5,
                repetition_penalty=2.0,
                pad_token_id=llama_tokenizer.pad_token_id
            )

        # Decode the outputs
        decoded_output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store the results
        results[ne] = decoded_output

    # Results now contain the decoded predictions for each named entity
    for ne, answer in results.items():
        print("\n\n")
        print(ne)
        print(answer)


if __name__ == '__main__':

    try:
        BUSTER_BIO_test = loadDataset('../../datasets/BUSTER/FULL_KFOLDS/123_4_5')['test']
    except FileNotFoundError:
        BUSTER_BIO_test = loadDataset('datasets/BUSTER/BIO_format/FULL_KFOLDS_BIO/123_4_5')['test']

    # print(f"Available GPU memory before loading tokenizer\n: {torch.cuda.memory_summary()}")
    # pip install SentencePiece, protobuf
    llama_tokenizer = AutoTokenizer.from_pretrained('Universal-NER/UniNER-7B-type', cache_dir='./hf_cache_dir')
    # print(f"Available GPU memory after loading model\n: {torch.cuda.memory_summary()}")

    # print(f"Available GPU memory before loading model\n: {torch.cuda.memory_summary()}")
    uniNER_llama_model = LlamaForCausalLM.from_pretrained('Universal-NER/UniNER-7B-type', cache_dir='./hf_cache_dir', torch_dtype=torch.float16)
    # print(f"Available GPU memory after loading model\n: {torch.cuda.memory_summary()}")

    device = torch.device("cuda")
    # uniNER_llama_model.to(device)
    accelerator = Accelerator(mixed_precision='fp16')
    # print(f"Available GPU memory before passing model\n: {torch.cuda.memory_summary()}")
    uniNER_llama_model = accelerator.prepare(uniNER_llama_model)
    print(f"Available GPU memory after passing model\n: {torch.cuda.memory_summary()}")

    print("Getting NEs for some test documents: ")
    for i in range(100):
        sample_BIO = BUSTER_BIO_test[i]
        docID = sample_BIO['document_id']
        print(f"document_id: {docID}\n")
        get_named_entities_for_a_doc(sample_BIO)
        print("\n -------------------------------------- \n")

