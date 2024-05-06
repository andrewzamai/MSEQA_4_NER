__package__ = "MSEQA_4_NER.utils"

from huggingface_hub import HfApi, create_repo, upload_folder
from SFT_finetuning.commons.initialization import get_HF_access_token

if __name__ == '__main__':

    # load HuggingFace access token from .env file (git ignored)
    AZ_HF_ACCESS_TOKEN = get_HF_access_token('./.env')

    #path_to_model_to_upload = './merged_models/llama2_7B_5pos_5neg_perNE_TrueZeroShot_top50NEs_TrueDef'
    path_to_model_to_upload = './datasets'
    new_repo_name = 'andrewzamai/ZeroShotNER_datasets'

    url_new_repo_name = create_repo(
        repo_id=new_repo_name,
        token=AZ_HF_ACCESS_TOKEN,
        exist_ok=False,
        private=True,
        repo_type='model',
    )
    
    print(url_new_repo_name)
    
    uploaded_folder_results = upload_folder(
        folder_path=path_to_model_to_upload,
        repo_id=new_repo_name,
        repo_type='model',
        token=AZ_HF_ACCESS_TOKEN
    )
    
    print(uploaded_folder_results)

    #from transformers import AutoTokenizer
    #tokenizer_name = "roberta-base"
    #tokenizer_name = "microsoft/deberta-v2-xxlarge"
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='./hf_cache_dir')
    #tokenizer.push_to_hub(new_repo_name, token=api_keys_dict['AZ_HUGGINGFACE_TOKEN'])
