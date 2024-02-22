from huggingface_hub import HfApi, create_repo, upload_folder

if __name__ == '__main__':

    # load HuggingFace access token from .env file (git ignored)
    with open('./MSEQA_4_NER/experiments/.env', 'r') as file:
        api_keys = file.readlines()
    api_keys_dict = {}
    for api_key in api_keys:
        api_name, api_value = api_key.split('=')
        api_keys_dict[api_name] = api_value
    # print(api_keys_dict)

    """
    my_repo_url = "https://huggingface.co/andrewzamai"
    
    hf_api = HfApi(
        endpoint=my_repo_url,
        token=api_keys_dict['AZ_HUGGINGFACE_TOKEN'],
    )
    """

    path_to_model_to_upload = './baseline_Deberta_FT/DeBERTa_MSEQA_pileNERpt_FalseDef_C-bis/finetuned_model'
    new_repo_name = 'andrewzamai/MSEQA-DeBERTaXXL-FalseDef-C-bis'

    url_new_repo_name = create_repo(
        repo_id=new_repo_name,
        token=api_keys_dict['AZ_HUGGINGFACE_TOKEN'],
        exist_ok=False,
        private=True,
        repo_type='model',
    )
    
    print(url_new_repo_name)
    
    uploaded_folder_results = upload_folder(
        folder_path=path_to_model_to_upload,
        repo_id=new_repo_name,
        repo_type='model',
        token=api_keys_dict['AZ_HUGGINGFACE_TOKEN']
    )
    
    print(uploaded_folder_results)

    #from transformers import AutoTokenizer
    #tokenizer_name = "roberta-base"
    #tokenizer_name = "microsoft/deberta-v2-xxlarge"
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='./hf_cache_dir')
    #tokenizer.push_to_hub(new_repo_name, token=api_keys_dict['AZ_HUGGINGFACE_TOKEN'])
