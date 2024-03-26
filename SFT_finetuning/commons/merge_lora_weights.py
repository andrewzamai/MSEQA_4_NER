""" merge fine-tuned LORA adapter weights with base model """

__package__ = "SFT_finetuning.commons"

import os.path
import time
import shutil

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

    HF_ACCESS_TOKEN = get_HF_access_token('./.env')
    login(token=HF_ACCESS_TOKEN)

    base_model = "meta-llama/Llama-2-7b-chat-hf"
    # as it is the code requires namespace/model_name format only, no more subfolders
    path_to_lora = "./trained_models/llama2_4_NER_FalseDef_FULLpileNER"
    save_model_at = "./merged_models/llama2_4_NER_FalseDef_FULLpileNER"

    # fire.Fire(merge_main)
    merge_main(base_model, path_to_lora, save_model_at)
