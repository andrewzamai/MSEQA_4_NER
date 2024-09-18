from datasets import load_dataset

# Save to CoNLL-style format
def save_conll_format(dataset_path, words_list, labels_list):
    with open(dataset_path, 'w') as f:
        for words, labels in zip(words_list, labels_list):
            for word, label in zip(words, labels):
                if label == 'O':
                    f.write(f"{word}\t{label}\n")
                else:
                    prefix, tag = label.split('-')
                    tagClass, tagName = tag.split('.')
                    label = prefix + '-' + tagName.lower()
                    f.write(f"{word}\t{label}\n")
            f.write("\n")  # Sentence boundary

if __name__ == '__main__':

    buster_test_BIO = load_dataset("json", data_files=f'../../../datasets/BUSTER/CHUNKED_KFOLDS/123_4_5/test.json')['train']
    print(buster_test_BIO['labels'])

    save_conll_format('../../GoLLIE/data/BUSTER/test.txt', buster_test_BIO['tokens'], buster_test_BIO['labels'])