"""

- new BUSTER dataset management library for EQA in multi-span setting -

Dataset loading, document retrieval from ID, from BIO labeling to multispan EQA format,
ground truth metadata retrieval (non-positional labels) from BUSTER dataset (BIO positional labeling), etc ...

"""


from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import os
import pickle


def createDatasetsFromKFoldsPermutations(pathToKFoldDir):
    permutations = [{'train': ['FOLD_1', 'FOLD_2', 'FOLD_3'], 'validation': 'FOLD_4', 'test': 'FOLD_5', 'name': '123_4_5'},
                    {'train': ['FOLD_5', 'FOLD_1', 'FOLD_2'], 'validation': 'FOLD_3', 'test': 'FOLD_4', 'name': '512_3_4'},
                    {'train': ['FOLD_4', 'FOLD_5', 'FOLD_1'], 'validation': 'FOLD_2', 'test': 'FOLD_3', 'name': '451_2_3'},
                    {'train': ['FOLD_3', 'FOLD_4', 'FOLD_5'], 'validation': 'FOLD_1', 'test': 'FOLD_2', 'name': '345_1_2'},
                    {'train': ['FOLD_2', 'FOLD_3', 'FOLD_4'], 'validation': 'FOLD_5', 'test': 'FOLD_1', 'name': '234_5_1'}
                    ]

    for perm in permutations:
        data_files = {"train_0": os.path.join(pathToKFoldDir, perm['train'][0] + '.json'),
                      "train_1": os.path.join(pathToKFoldDir, perm['train'][1] + '.json'),
                      "train_2": os.path.join(pathToKFoldDir, perm['train'][2] + '.json'),
                      "validation": os.path.join(pathToKFoldDir, perm['validation'] + '.json'),
                      "test": os.path.join(pathToKFoldDir, perm['test'] + '.json')
                      }
        dataset_dict = load_dataset("json", data_files=data_files)
        merged_train_folds = concatenate_datasets([dataset_dict['train_0'], dataset_dict['train_1'], dataset_dict['train_2']], axis=0)
        del dataset_dict['train_0']
        del dataset_dict['train_1']
        del dataset_dict['train_2']
        dataset_dict['train'] = merged_train_folds

        print(dataset_dict.keys())
        print(len(dataset_dict['train']))
        print(len(dataset_dict['validation']))
        print(len(dataset_dict['test']))
        print("\n")

        with open(os.path.join('./DATASETS', perm['name'] + '.pickle'), 'wb') as f:
            pickle.dump(dataset_dict, f)


#  load BUSTER dataset from json files, requires as parameter path to folder where json files are
def loadDataset(pathToDir):
    data_files = {"train": os.path.join(pathToDir, "train.json"),
                  "test": os.path.join(pathToDir, "test.json"),
                  "validation": os.path.join(pathToDir, "validation.json")} 
    # , "silver": os.path.join(pathToDir, "silver.json")}
    return load_dataset("json", data_files=data_files)


# get document at position i in the split (train/validation/test/silver)
# returns "split:document_id" ad new docID, tokensList, labelsList
def getDocumentTokensLabels(raw_dataset, split, i):
    docID = split + ':' + raw_dataset[split][i]["document_id"]  # new docID
    return docID, raw_dataset[split][i]["tokens"], raw_dataset[split][i]["labels"]


# retrieves document metadata,
# but also start-end indexes (in character count and not token count)
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


# since folds can be later shuffled, we retrieve documents from their document_id within their fold name
def retrieveDocFromID(raw_dataset, docID):
    # split on ':' to retrieve fold name and document_id
    foldName, id = docID.split(':')
    document = filter(lambda x: x["document_id"] == id, raw_dataset[foldName])
    if document is not None:
        return list(document)[0]
    else:
        print("Document not found!")


# a list of questions for each class is stored in a 'questions.txt' file
# a txt with 1 single question for each class is instead stored in singleQuestionPerClass.txt
def loadQuestionsDictFromTxt(pathToTxt):
    questions = dict()
    with open(pathToTxt, "r") as f:
        newClass = True
        for line in f:
            if newClass:
                tagFamily, tagName = line.strip().split(':')
                if tagFamily not in questions.keys():
                    questions[tagFamily] = dict()
                if tagName not in questions[tagFamily].keys():
                    questions[tagFamily][tagName] = []
                newClass = False
                continue
            if line.strip() != '---':
                questions[tagFamily][tagName].append(line.strip())
            else:
                newClass = True
    return questions


# build DatasetDict object with docContext-question-goldAnswers
def build_dataset_MSEQA_format(pathToBUSTERDir, pathToQuestionsTxt):
    raw_BUSTER_dataset = loadDataset(pathToBUSTERDir)
    questions = loadQuestionsDictFromTxt(pathToQuestionsTxt)
    print(questions)

    newDataset_dict = {splitName: [] for splitName in raw_BUSTER_dataset.keys()}
    newDataset_Dataset = {splitName: None for splitName in raw_BUSTER_dataset.keys()}
    for splitName in raw_BUSTER_dataset.keys():
        for i in range(len(raw_BUSTER_dataset[splitName])):
            docID, tokens, labels = getDocumentTokensLabels(raw_BUSTER_dataset, splitName, i)
            docMetadata = getDocMetadataWithStartEndCharIndexes(tokens, labels)
            question_number = 0
            for tagFamily in questions.keys():
                for tagName in questions[tagFamily].keys():
                    question = questions[tagFamily][tagName][0]  # only 1 question per tagName
                    # splitName:docID:questioNumberForThatDocument
                    doc_question_pairID = docID + ':' + str(question_number)
                    question_number += 1
                    # document context 
                    context = ' '.join([str(elem) for elem in tokens])
                    # retrieving gold answers for this tagName
                    goldAnswers = docMetadata[tagFamily][tagName]
                    answers = {'answer_start': [], 'text': []}
                    for ga in goldAnswers:
                        answers['answer_start'].append(ga[1])
                        answers['text'].append(ga[0])
                    sample = {'doc_question_pairID': doc_question_pairID, 
                              'document_context': context,
                              'tagFamily': tagFamily,
                              'tagName': tagName,
                              'question': question,
                              'answers': answers
                              }
                    newDataset_dict[splitName].append(sample)
        newDataset_Dataset[splitName] = Dataset.from_list(newDataset_dict[splitName])

    return DatasetDict(newDataset_Dataset)


if __name__ == "__main__":
    createDatasetsFromKFoldsPermutations('../../BUSTER_KFOLDs')
