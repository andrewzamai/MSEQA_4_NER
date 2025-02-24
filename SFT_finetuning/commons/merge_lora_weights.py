""" merge fine-tuned LORA adapter weights with base model """

__package__ = "SFT_finetuning.commons"

import os.path
import sys
import time
import shutil
import argparse

# my libraries
from huggingface_hub import login
from .initialization import get_HF_access_token, init_model


def merge_main(
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",
    lora_weights: str = "./saved_models/lora_weights_model",
    merged_model_dir: str = "./where/to/store/the/merged/weights",
    cutoff_len: int = 2048,
    device_map: str = "auto",
):
    start_time = time.time()
    # load base model with lora weights
    tokenizer, model_to_merge = init_model(
        base_model,
        lora_weights=lora_weights,
        load_8bit=False,
        cutoff_len=cutoff_len,
        device_map=device_map,
    )

    # merge model, then save it
    start_merge_time = time.time()
    merged_model = model_to_merge.merge_and_unload()
    print(f"\n\nMerge only took {time.time() - start_merge_time} seconds\n")
    merge_time = time.time() - start_time
    print(f"\n\nWeights load and merge took {merge_time} seconds\n")

    tokenizer.save_pretrained(merged_model_dir)
    merged_model.save_pretrained(merged_model_dir)

    merge_and_save_time = time.time() - start_time
    print(f"\n\nAll took {merge_and_save_time} seconds\n")

    # copy also training_config file if exists
    if os.path.exists(os.path.join(path_to_lora, 'training_configs.yml')):
        shutil.copy(os.path.join(path_to_lora, 'training_configs.yml'), os.path.join(save_model_at, 'training_configs.yml'))


if __name__ == "__main__":

    print("Merged Began")
    sys.stdout.flush()

    HF_ACCESS_TOKEN = get_HF_access_token('./.env')
    login(token=HF_ACCESS_TOKEN)

    base_model = "meta-llama/Llama-2-7b-chat-hf"
    # as it is the code requires namespace/model_name format only, no more subfolders

    """
    parser = argparse.ArgumentParser(description='''Llama merger parser''')
    # adding arguments
    parser.add_argument('--with_guidelines', action='store_true', help='Whether to use guidelines')
    parser.add_argument('number_NEs', type=int, help='Number of NEs')
    parser.add_argument('number_pos_samples_per_NE', type=int, help='Number of positive samples per NE')
    parser.add_argument('number_neg_samples_per_NE', type=int, help='Number of negative samples per NE')
    # parsing arguments
    args = parser.parse_args()
    """
    #path_to_lora = f"./trained_models/llama2_7B_{args.number_pos_samples_per_NE}pos_{args.number_neg_samples_per_NE}neg_perNE_top{args.number_NEs}NEs_{args.with_guidelines}Def-E"
    #save_model_at = f"./merged_models/llama2_7B_{args.number_pos_samples_per_NE}pos_{args.number_neg_samples_per_NE}neg_perNE_top{args.number_NEs}NEs_{args.with_guidelines}Def-E"

    #path_to_lora = f"./llama_for_NER_trained_models/llama2_7B_{args.number_pos_samples_per_NE}pos_{args.number_neg_samples_per_NE}neg_perNE_top{args.number_NEs}NEs_{args.with_guidelines}Def-SI-B"
    #save_model_at = f"./merged_models/llama2_7B_{args.number_pos_samples_per_NE}pos_{args.number_neg_samples_per_NE}neg_perNE_top{args.number_NEs}NEs_{args.with_guidelines}Def-SI-B"

    path_to_lora = "/nfsd/VFdisk/zamaiandre/ZeroShotNER/trained_models/GNER-pileNER-391x50-onlyPOS"
    save_model_at = "/nfsd/VFdisk/zamaiandre/ZeroShotNER/merged_models/GNER-pileNER-391x50-onlyPOS"
    merge_main(base_model, path_to_lora, save_model_at)

    """ PUSH TO HF HUB """
    from huggingface_hub import create_repo, upload_folder
    from SFT_finetuning.commons.initialization import get_HF_access_token

    # new_repo_name = f"andrewzamai/{save_model_at.split('/')[-1]}"
    #new_repo_name = "expertai/SLIMER"

    """
    url_new_repo_name = create_repo(
        repo_id=new_repo_name,
        token=HF_ACCESS_TOKEN,
        exist_ok=False, # False
        private=True,
        repo_type='model',
    )

    print(url_new_repo_name)
    

    uploaded_folder_results = upload_folder(
        folder_path=save_model_at,
        repo_id=new_repo_name,
        repo_type='model',
        token=HF_ACCESS_TOKEN
    )

    print(uploaded_folder_results)

    print("Merged and pushed to HF hub :)\n\n")
    """
