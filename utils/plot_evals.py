""" plot evaluation outputs """

import matplotlib.pyplot as plt
import numpy as np
import os
import re

# all_ne_list: list of Named Entities in this dataset
# novel_ne_list: NEs in this dataset but not in pileNER training
# same_ne_diff_def_list: NEs which have same name, but different definition wrt pileNER

movie_ne_statistics = {
    'all_ne_list': ['ACTOR', 'CHARACTER', 'DIRECTOR', 'GENRE', 'PLOT', 'RATING', 'RATINGS_AVERAGE', 'REVIEW', 'SONG', 'TITLE', 'TRAILER', 'YEAR'],
    'novel_ne_list': ['DIRECTOR', 'PLOT', 'RATING', 'RATINGS_AVERAGE', 'REVIEW', 'TRAILER'],
    'same_ne_diff_def_list': ['TITLE', 'YEAR']
}

restaurant_ne_statistics = {
    'all_ne_list': ['Amenity', 'Cuisine', 'Dish', 'Hours', 'Location', 'Price', 'Rating', 'Restaurant_Name'],
    'novel_ne_list': ['Amenity', 'Cuisine', 'Hours', 'Price', 'Rating'],
    'same_ne_diff_def_list': []
}

ai_ne_statistics = {
    'all_ne_list': ['algorithm', 'conference', 'country', 'field', 'location', 'metrics', 'misc', 'organisation', 'person', 'product', 'programlang', 'researcher', 'task', 'university'],
    'novel_ne_list': ['conference', 'misc', 'researcher'],
    'same_ne_diff_def_list': ['person']
}

literature_ne_statistics = {
    'all_ne_list': ['award', 'book', 'country', 'event', 'literarygenre', 'location', 'magazine', 'misc', 'organisation', 'person', 'poem', 'writer'],
    'novel_ne_list': ['misc', 'poem', 'writer'],
    'same_ne_diff_def_list': ['person']
}

music_ne_statistics = {
    'all_ne_list': ['album', 'award', 'band', 'country', 'event', 'location', 'misc', 'musicalartist', 'musicalinstrument', 'musicgenre', 'organisation', 'person', 'song'],
    'novel_ne_list': ['misc'],
    'same_ne_diff_def_list': ['person']
}

politics_ne_statistics = {
    'all_ne_list': ['country', 'election', 'event', 'location', 'misc', 'organisation', 'person', 'politicalparty', 'politician'],
    'novel_ne_list': ['election', 'misc', 'politician'],
    'same_ne_diff_def_list': ['person']
}

science_ne_statistics = {
    'all_ne_list': ['academicjournal', 'astronomicalobject', 'award', 'chemicalcompound', 'chemicalelement', 'country', 'discipline', 'enzyme', 'event', 'location', 'misc', 'organisation', 'person', 'protein', 'scientist', 'theory', 'university'],
    'novel_ne_list': ['academicjournal', 'astronomicalobject', 'misc', 'chemicalelement', 'discipline', 'scientist', 'theory'],
    'same_ne_diff_def_list': ['person', 'protein']
}

BUSTER_ne_statistics = {
    'all_ne_list': ['BUYING_COMPANY', 'SELLING_COMPANY', 'ACQUIRED_COMPANY', 'LEGAL_CONSULTING_COMPANY', 'GENERIC_CONSULTING_COMPANY', 'ANNUAL_REVENUES'],
    'novel_ne_list': ['BUYING_COMPANY', 'SELLING_COMPANY', 'ACQUIRED_COMPANY', 'LEGAL_CONSULTING_COMPANY', 'GENERIC_CONSULTING_COMPANY', 'ANNUAL_REVENUES'],
    'same_ne_diff_def_list': []
}

all_datasets_ne_statistics = {
    'movie': movie_ne_statistics,
    'restaurant': restaurant_ne_statistics,
    'ai': ai_ne_statistics,
    'literature': literature_ne_statistics,
    'music': music_ne_statistics,
    'politics': politics_ne_statistics,
    'science': science_ne_statistics,
    'BUSTER': BUSTER_ne_statistics
}

# uniNER-eval datasets use a list of NEs different from original crossNER/MIT
# to adapt to MSEQA setting and above ne_statistics here is a mapping
ne_mapping = {
    'crossNER': {
        'organization': 'organisation',
        'program language': 'programlang',
        'literary genre': 'literarygenre',
        'astronomical object': 'astronomicalobject',
        'chemical element': 'chemicalelement',
        'chemical compound': 'chemicalcompound',
        'academic journal': 'academicjournal',
        'political party': 'politicalparty',
        'musical artist': 'musicalartist',
        'musical instrument': 'musicalinstrument',
        'music genre': 'musicgenre',
    },
    'movie': {
        'character': 'CHARACTER',
        'plot': 'PLOT',
        'year': 'YEAR',
        'director': 'DIRECTOR',
        'rating': 'RATING',
        'average ratings': 'RATINGS_AVERAGE',
        'actor': 'ACTOR',
        'genre': 'GENRE',
        'song': 'SONG',
        'trailer': 'TRAILER',
        'review': 'REVIEW',
        'title': 'TITLE'
    },
    'restaurant': {
        'amenity': 'Amenity',
        'location': 'Location',
        'cuisine': 'Cuisine',
        'restaurant name': 'Restaurant_Name',
        'rating': 'Rating',
        'hours': 'Hours',
        'price': 'Price',
        'dish': 'Dish'
    }
}


def compute_percentage_overlap(dataset_ne_statistics):
    return len(set(dataset_ne_statistics['all_ne_list']) - set(dataset_ne_statistics['novel_ne_list'])) / len(dataset_ne_statistics['all_ne_list']) * 100


def read_dataset_evals(path_to_file, dataset_name):

    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # finding line at which evaluation metrics for dataset_name start
    evaluating_pattern = re.compile(r'^Evaluating.*? model.*? on \'(.+?)\' test fold in ZERO-SHOT setting$')
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            if evaluate_match.group(1) == dataset_name:
                toread_start = i
                break

    # Regular expression patterns for extracting metrics
    support_pattern = re.compile(r'^([\w.]+) --> support: (\d+)$')
    metrics_pattern = re.compile(r'^([\w.]+) --> Precision: (\d+\.\d+), Recall: (\d+\.\d+), F1: (\d+\.\d+)$')

    ner_metrics = {}
    # reading lines after toread_start until new evaluation starts
    for i in range(toread_start + 1, len(logs)):
        line = logs[i]
        # Check if the line contains metrics for a Named Entity
        support_match = support_pattern.match(line)
        if support_match:
            support = int(support_match.group(2))
        metrics_match = metrics_pattern.match(line)
        if metrics_match:
            current_ne = metrics_match.group(1)
            if '.' in current_ne:
                current_ne = current_ne.split('.')[-1]
            precision = float(metrics_match.group(2))
            recall = float(metrics_match.group(3))
            f1 = float(metrics_match.group(4))
            ner_metrics[current_ne] = {'Precision': precision, 'Recall': recall, 'F1': f1, 'support': support}

        # Check if a new evaluation is starting
        evaluating_match = evaluating_pattern.match(line)
        if evaluating_match:
            break

    return ner_metrics


def read_dataset_evals_llama2(path_to_file, dataset_name):

    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # finding line at which evaluation metrics for dataset_name start
    evaluating_pattern = re.compile(r'^Evaluating model named.*? on \'(.+?)\' test fold in ZERO-SHOT setting$')
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            if evaluate_match.group(1) == dataset_name:
                toread_start = i
                break

    # Regular expression patterns for extracting metrics
    ne_pattern = re.compile(r'^\s*NE:\s*([\w.]+)$')

    metrics_pattern = re.compile(r'\s*Precision: (\d+\.\d+) % -- Recall: (\d+\.\d+) % -- F1: (\d+\.\d+) %')

    ner_metrics = {}
    # reading lines after toread_start until new evaluation starts
    for i in range(toread_start + 1, len(logs)):
        line = logs[i]
        # Check if the line contains metrics for a Named Entity
        ne_match = ne_pattern.match(line)
        if ne_match:
            ne_name = ne_match.group(1)
            print(ne_name)
        metrics_match = metrics_pattern.match(line)
        if metrics_match:
            precision = float(metrics_match.group(1))
            recall = float(metrics_match.group(2))
            f1 = float(metrics_match.group(3))
            ner_metrics[ne_name] = {'Precision': precision, 'Recall': recall, 'F1': f1, 'support': 1000}

        # Check if a new evaluation is starting
        evaluating_match = evaluating_pattern.match(line)
        if evaluating_match:
            break

    return ner_metrics


def get_zero_shot_metrics(path_to_evals_folder, yes_no_Def, model_name):
    path_to_file = os.path.join(path_to_evals_folder, str(yes_no_Def) + 'Def_' + model_name + '.txt')
    ner_metrics = {}
    for dataset_name in dataset_name_list:
        ner_metrics[dataset_name] = read_dataset_evals(path_to_file, dataset_name)
    return ner_metrics


def get_zero_shot_metrics_from_file(path_to_file, dataset_name_list):
    ner_metrics = {}
    for dataset_name in dataset_name_list:
        ner_metrics[dataset_name] = read_dataset_evals(path_to_file, dataset_name)
        # ner_metrics[dataset_name] = read_dataset_evals_llama2(path_to_file, dataset_name)
    return ner_metrics


def plot_FalseTrueDef_comparison(ner_metrics_FalseDef, ner_metrics_TrueDef, dataset_name, ne_statistics):
    x_labels = sorted(ner_metrics_FalseDef[dataset_name].keys())
    thisds_metrics_FalseDef = [ner_metrics_FalseDef[dataset_name][x]['F1'] for x in x_labels]
    thisds_metrics_TrueDef = [ner_metrics_TrueDef[dataset_name][x]['F1'] for x in x_labels]

    # Set up positions for bars on x-axis
    bar_width = 0.4  # Adjust as needed
    x = np.arange(len(x_labels))

    # Plotting the first set of bars
    # width proportional to support
    # x:bar_width = ne_support:max(all_ne_supports)
    max_support = max([ner_metrics_TrueDef[dataset_name][x]['support'] for x in x_labels])
    widths = [bar_width * ner_metrics_TrueDef[dataset_name][x]['support']/max_support for x in x_labels]
    plt.bar(x - bar_width / 2, thisds_metrics_FalseDef, width=widths, label='NoDef')

    # Plotting the second set of bars
    plt.bar(x + bar_width / 2, thisds_metrics_TrueDef, width=widths, label='YesDef')

    # Adding labels and title
    plt.xlabel('NE categories')
    plt.ylabel('micro-F1')
    plt.title('FalseDef vs TrueDef comparison')

    plt.xticks(x, x_labels, rotation=10, fontsize=6)
    for i, label in enumerate(x_labels):
        if label in ne_statistics[dataset_name]['novel_ne_list']:
            plt.gca().get_xticklabels()[i].set_color("red")
        elif label in ne_statistics[dataset_name]['same_ne_diff_def_list']:
            plt.gca().get_xticklabels()[i].set_color("orange")
        else:
            plt.gca().get_xticklabels()[i].set_color("black")

    plt.legend()
    plt.show()


def plot_avg_std_FalseTrueDef_comparison(ner_metrics_FalseDef_with_avg_std, ner_metrics_TrueDef_with_avg_std, dataset_name, ne_statistics, model_name):
    x_labels = sorted(ner_metrics_FalseDef_with_avg_std[dataset_name].keys())

    thisds_metrics_FalseDef = [ner_metrics_FalseDef_with_avg_std[dataset_name][x]['avg_F1'] for x in x_labels]
    thisds_metrics_FalseDef_stds = [ner_metrics_FalseDef_with_avg_std[dataset_name][x]['std_F1'] for x in x_labels]

    thisds_metrics_TrueDef = [ner_metrics_TrueDef_with_avg_std[dataset_name][x]['avg_F1'] for x in x_labels]
    thisds_metrics_TrueDef_stds = [ner_metrics_TrueDef_with_avg_std[dataset_name][x]['std_F1'] for x in x_labels]

    # Set up positions for bars on x-axis
    bar_width = 0.4  # Adjust as needed
    x = np.arange(len(x_labels))

    # Plotting the first set of bars
    # width proportional to support x:bar_width = ne_support:max(all_ne_supports)
    max_support = max([ner_metrics_TrueDef_with_avg_std[dataset_name][x]['support'] for x in x_labels])
    widths = [bar_width * ner_metrics_TrueDef_with_avg_std[dataset_name][x]['support']/max_support for x in x_labels]
    plt.bar(x - bar_width / 2, thisds_metrics_FalseDef, width=widths, label='NoDef')
    plt.errorbar(x - bar_width / 2, thisds_metrics_FalseDef, yerr=thisds_metrics_FalseDef_stds, fmt='none', ecolor='black', capsize=3, elinewidth=0.05)

    # Plotting the second set of bars
    plt.bar(x + bar_width / 2, thisds_metrics_TrueDef, width=widths, label='YesDef')
    plt.errorbar(x + bar_width / 2, thisds_metrics_TrueDef, yerr=thisds_metrics_TrueDef_stds, fmt='none', ecolor='black', capsize=3, elinewidth=0.05)

    # Adding labels and title
    plt.xlabel('NE categories')
    plt.ylabel('average per-NE F1 across different runs')
    plt.title(f'{dataset_name} - {model_name} - FalseDef vs TrueDef comparison')

    plt.xticks(x, x_labels, rotation=10, fontsize=6)
    for i, label in enumerate(x_labels):
        if label in ne_statistics[dataset_name]['novel_ne_list']:
            plt.gca().get_xticklabels()[i].set_color("red")
        elif label in ne_statistics[dataset_name]['same_ne_diff_def_list']:
            plt.gca().get_xticklabels()[i].set_color("orange")
        else:
            plt.gca().get_xticklabels()[i].set_color("black")

    plt.legend()
    plt.show()


if __name__ == '__main__':

    path_to_evals_folder = '../../experiments_outputs/Llama2-7B'

    dataset_name_list = ['movie', 'restaurant', 'ai', 'literature', 'music', 'politics', 'science', 'BUSTER']

    filenames = os.listdir(path_to_evals_folder)
    filenames_grouped = {'TrueDef': [], 'FalseDef': []}
    """
    for fn in filenames:
        if fn != '.DS_Store' and not os.path.isdir(os.path.join(path_to_evals_folder, fn)):
            yes_no_def = fn.split('-')[-2]
            filenames_grouped[yes_no_def].append(fn)
    """
    #filenames_grouped['FalseDef'].append('uniNEReval_LLama2_7b-FalseDef-A.txt')

    #filenames_grouped['TrueDef'].append('uniNEReval_LLama2_7b-TrueDef-enhanced2midcp.txt')
    #filenames_grouped['FalseDef'].append('uniNEReval_LLama2_7b-TrueDef-enhancedcp3200.txt')

    filenames_grouped['FalseDef'].append('uniNEReval_LLama2_7b-TrueDef-enhanced2midcp.txt')
    filenames_grouped['TrueDef'].append('uniNEReval_LLama2_7b-TrueDef-enhanced2midcp_masked_eval.txt')

    #filenames_grouped['TrueDef'].append('uniNEReval_LLama2_7b-TrueDef-A.txt')
    print(filenames_grouped)

    collected_metrics = {'FalseDef': {x: [] for x in dataset_name_list}, 'TrueDef': {x: [] for x in dataset_name_list}}
    for bs_run in filenames_grouped['FalseDef']:
        path_to_file = os.path.join(path_to_evals_folder, bs_run)
        ner_metrics_FalseDef = get_zero_shot_metrics_from_file(path_to_file, dataset_name_list)
        for ds_name, ds_metrics in ner_metrics_FalseDef.items():
            collected_metrics['FalseDef'][ds_name].append(ds_metrics)

    for bs_run in filenames_grouped['TrueDef']:
        path_to_file = os.path.join(path_to_evals_folder, bs_run)
        ner_metrics_TrueDef = get_zero_shot_metrics_from_file(path_to_file, dataset_name_list)
        for ds_name, ds_metrics in ner_metrics_TrueDef.items():
            collected_metrics['TrueDef'][ds_name].append(ds_metrics)

    print(collected_metrics)

    # get something like {'FalseDef': {'movie': {'GENRE': {'Precision': [22.21, 20.76, 20.42, 22.88], 'Recall': [26.59, 25.78, 25.25, 30.98]
    grouped_metrics = {'FalseDef': {x: {} for x in dataset_name_list}, 'TrueDef': {x: {} for x in dataset_name_list}}
    for w_wo_def, w_wo_values in collected_metrics.items():
        for ds_name, ds_values in w_wo_values.items():
            for ne_mvalues in ds_values:
                for ne, mvalues in ne_mvalues.items():
                    if ne not in grouped_metrics[w_wo_def][ds_name]:
                        grouped_metrics[w_wo_def][ds_name][ne] = {}
                        for mname, v in mvalues.items():
                            grouped_metrics[w_wo_def][ds_name][ne][mname] = [v]
                    else:
                        for mname, v in mvalues.items():
                            grouped_metrics[w_wo_def][ds_name][ne][mname].append(v)
    print(grouped_metrics)

    average_std_metrics = {'FalseDef': {x: {} for x in dataset_name_list},
                           'TrueDef': {x: {} for x in dataset_name_list}}
    for w_wo_def, w_wo_values in grouped_metrics.items():
        for ds_name, ds_values in w_wo_values.items():
            for ne_name, m_values_list in ds_values.items():
                avg_precision = np.average(m_values_list['Precision'])
                std_precision = np.std(m_values_list['Precision'])
                avg_recall = np.average(m_values_list['Recall'])
                std_recall = np.std(m_values_list['Recall'])
                avg_f1 = np.average(m_values_list['F1'])
                std_f1 = np.std(m_values_list['F1'])
                average_std_metrics[w_wo_def][ds_name][ne_name] = {'avg_precision': avg_precision,
                                                                   'std_precision': std_precision,
                                                                   'avg_recall': avg_recall,
                                                                   'std_recall': std_recall,
                                                                   'avg_F1': avg_f1,
                                                                   'std_F1': std_f1,
                                                                   'support': m_values_list['support'][0]
                                                                   }
    print(average_std_metrics)

    for ds_name in dataset_name_list:
        plot_avg_std_FalseTrueDef_comparison(average_std_metrics['FalseDef'], average_std_metrics['TrueDef'], ds_name,
                                             all_datasets_ne_statistics, f'MSEQA-DeBERTa-XXL')

    """
    roberta_model = 'large'

    # list of datasets for which to compute evaluation statistics
    dataset_name_list = ['movie', 'restaurant', 'ai', 'literature', 'music', 'politics', 'science', 'BUSTER']

    path_to_evals_folder = '../../../experiments_outputs/baseline_4_evals'
    baseline_runs = ['baseline_4_a', 'baseline_4_b', 'baseline_4_c', 'baseline_4_d']
    # each baseline_4_x contains evals output in txt files e.g. TrueDef_large.txt and FalseDef_large.txt

    collected_metrics = {'FalseDef': {x: [] for x in dataset_name_list}, 'TrueDef': {x: [] for x in dataset_name_list}}
    for bs_run in baseline_runs:
        path = os.path.join(path_to_evals_folder, bs_run)
        ner_metrics_FalseDef = get_zero_shot_metrics(path, False, roberta_model)
        for ds_name, ds_metrics in ner_metrics_FalseDef.items():
            collected_metrics['FalseDef'][ds_name].append(ds_metrics)

        ner_metrics_TrueDef = get_zero_shot_metrics(path, True, roberta_model)
        for ds_name, ds_metrics in ner_metrics_TrueDef.items():
            collected_metrics['TrueDef'][ds_name].append(ds_metrics)

    print(collected_metrics)

    # get something like {'FalseDef': {'movie': {'GENRE': {'Precision': [22.21, 20.76, 20.42, 22.88], 'Recall': [26.59, 25.78, 25.25, 30.98]
    grouped_metrics = {'FalseDef': {x: {} for x in dataset_name_list}, 'TrueDef': {x: {} for x in dataset_name_list}}
    for w_wo_def, w_wo_values in collected_metrics.items():
        for ds_name, ds_values in w_wo_values.items():
            for ne_mvalues in ds_values:
                for ne, mvalues in ne_mvalues.items():
                    if ne not in grouped_metrics[w_wo_def][ds_name]:
                        grouped_metrics[w_wo_def][ds_name][ne] = {}
                        for mname, v in mvalues.items():
                            grouped_metrics[w_wo_def][ds_name][ne][mname] = [v]
                    else:
                        for mname, v in mvalues.items():
                            grouped_metrics[w_wo_def][ds_name][ne][mname].append(v)
    print(grouped_metrics)

    average_std_metrics = {'FalseDef': {x: {} for x in dataset_name_list}, 'TrueDef': {x: {} for x in dataset_name_list}}
    for w_wo_def, w_wo_values in grouped_metrics.items():
        for ds_name, ds_values in w_wo_values.items():
            for ne_name, m_values_list in ds_values.items():
                avg_precision = np.average(m_values_list['Precision'])
                std_precision = np.std(m_values_list['Precision'])
                avg_recall = np.average(m_values_list['Recall'])
                std_recall = np.std(m_values_list['Recall'])
                avg_f1 = np.average(m_values_list['F1'])
                std_f1 = np.std(m_values_list['F1'])
                average_std_metrics[w_wo_def][ds_name][ne_name] = {'avg_precision': avg_precision,
                                                                   'std_precision': std_precision,
                                                                   'avg_recall': avg_recall,
                                                                   'std_recall': std_recall,
                                                                   'avg_F1': avg_f1,
                                                                   'std_F1': std_f1,
                                                                   'support': m_values_list['support'][0]
                                                                   }
    print(average_std_metrics)

    for ds_name in dataset_name_list:
        plot_avg_std_FalseTrueDef_comparison(average_std_metrics['FalseDef'], average_std_metrics['TrueDef'], ds_name, all_datasets_ne_statistics, f'MSEQA-{roberta_model}')

    """

    """
    model_name = 't5-3b'
    path_to_evals_folder = '../../../experiments_outputs/T5-3b-MSEQA'

    dataset_name_list = ['movie', 'restaurant', 'ai', 'literature', 'music', 'politics', 'science', 'BUSTER']

    ner_metrics_FalseDef = get_zero_shot_metrics(path_to_evals_folder, False, model_name)
    print(ner_metrics_FalseDef)

    ner_metrics_TrueDef = get_zero_shot_metrics(path_to_evals_folder, True, 'deb-xxl')
    print(ner_metrics_TrueDef)

    for ds_name in dataset_name_list:
        plot_FalseTrueDef_comparison(ner_metrics_FalseDef, ner_metrics_TrueDef, ds_name, all_datasets_ne_statistics)

    # print(compute_percentage_overlap(ai_ne_statistics))
    """

    """
    path_to_evals_folder = '../../experiments_outputs/DebertaXXL-MSEQA'

    dataset_name_list = ['movie', 'restaurant', 'ai', 'literature', 'music', 'politics', 'science', 'BUSTER']

    filenames = os.listdir(path_to_evals_folder)
    filenames_grouped = {'TrueDef': [], 'FalseDef': []}
    for fn in filenames:
        if fn != '.DS_Store':
            yes_no_def = fn.split('-')[-2]
            filenames_grouped[yes_no_def].append(fn)
    print(filenames_grouped)

    collected_metrics = {'FalseDef': {x: [] for x in dataset_name_list}, 'TrueDef': {x: [] for x in dataset_name_list}}
    for bs_run in filenames_grouped['FalseDef']:
        path_to_file = os.path.join(path_to_evals_folder, bs_run)
        ner_metrics_FalseDef = get_zero_shot_metrics_from_file(path_to_file, dataset_name_list)
        for ds_name, ds_metrics in ner_metrics_FalseDef.items():
            collected_metrics['FalseDef'][ds_name].append(ds_metrics)

    for bs_run in filenames_grouped['TrueDef']:
        path_to_file = os.path.join(path_to_evals_folder, bs_run)
        ner_metrics_TrueDef = get_zero_shot_metrics_from_file(path_to_file, dataset_name_list)
        for ds_name, ds_metrics in ner_metrics_TrueDef.items():
            collected_metrics['TrueDef'][ds_name].append(ds_metrics)

    print(collected_metrics)

    # get something like {'FalseDef': {'movie': {'GENRE': {'Precision': [22.21, 20.76, 20.42, 22.88], 'Recall': [26.59, 25.78, 25.25, 30.98]
    grouped_metrics = {'FalseDef': {x: {} for x in dataset_name_list}, 'TrueDef': {x: {} for x in dataset_name_list}}
    for w_wo_def, w_wo_values in collected_metrics.items():
        for ds_name, ds_values in w_wo_values.items():
            for ne_mvalues in ds_values:
                for ne, mvalues in ne_mvalues.items():
                    if ne not in grouped_metrics[w_wo_def][ds_name]:
                        grouped_metrics[w_wo_def][ds_name][ne] = {}
                        for mname, v in mvalues.items():
                            grouped_metrics[w_wo_def][ds_name][ne][mname] = [v]
                    else:
                        for mname, v in mvalues.items():
                            grouped_metrics[w_wo_def][ds_name][ne][mname].append(v)
    print(grouped_metrics)

    average_std_metrics = {'FalseDef': {x: {} for x in dataset_name_list},
                           'TrueDef': {x: {} for x in dataset_name_list}}
    for w_wo_def, w_wo_values in grouped_metrics.items():
        for ds_name, ds_values in w_wo_values.items():
            for ne_name, m_values_list in ds_values.items():
                avg_precision = np.average(m_values_list['Precision'])
                std_precision = np.std(m_values_list['Precision'])
                avg_recall = np.average(m_values_list['Recall'])
                std_recall = np.std(m_values_list['Recall'])
                avg_f1 = np.average(m_values_list['F1'])
                std_f1 = np.std(m_values_list['F1'])
                average_std_metrics[w_wo_def][ds_name][ne_name] = {'avg_precision': avg_precision,
                                                                   'std_precision': std_precision,
                                                                   'avg_recall': avg_recall,
                                                                   'std_recall': std_recall,
                                                                   'avg_F1': avg_f1,
                                                                   'std_F1': std_f1,
                                                                   'support': m_values_list['support'][0]
                                                                   }
    print(average_std_metrics)

    for ds_name in dataset_name_list:
        plot_avg_std_FalseTrueDef_comparison(average_std_metrics['FalseDef'], average_std_metrics['TrueDef'], ds_name, all_datasets_ne_statistics, f'MSEQA-DeBERTa-XXL')

    """


