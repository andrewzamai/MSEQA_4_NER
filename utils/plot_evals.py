""" Helper module for plotting evaluation outputs: metrics per NE class, avg, std ... """
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import re

# all_ne_list: list of Named Entities in this dataset
# novel_ne_list: NEs in this dataset but not in pileNER pre-training, i.e. NOVEL NEs
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


def compute_percentage_overlap(dataset_ne_statistics):
    """ computes the percentage of overlapping NEs between training and ZERO-shot evaluation """
    return len(set(dataset_ne_statistics['all_ne_list']) - set(dataset_ne_statistics['novel_ne_list'])) / len(dataset_ne_statistics['all_ne_list']) * 100


def read_dataset_evals(path_to_file, dataset_name):
    """ reads an evaluation .txt file relative to a model on multiple datasets in ZERO-SHOT setting """

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
    support_pattern = re.compile(r'^([\w\s.]+) --> support: (\d+)$')
    metrics_pattern = re.compile(r'^([\w\s.]+) --> Precision: (\d+\.\d+), Recall: (\d+\.\d+), F1: (\d+\.\d+)$')

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


def get_zero_shot_metrics_from_file(path_to_file, dataset_name_list):
    """ loads eval metrics for all datasets in dataset_name_list from path_to_file.txt """
    ner_metrics = {}
    for dataset_name in dataset_name_list:
        ner_metrics[dataset_name] = read_dataset_evals(path_to_file, dataset_name)
    return ner_metrics


def collect_compute_prepare_metrics_for_plotting(path_to_evals_folder, filenames_grouped, dataset_name_list):

    collected_metrics = {'FalseDef': {x: [] for x in dataset_name_list}, 'TrueDef': {x: [] for x in dataset_name_list}}
    for key in filenames_grouped.keys():
        for bs_run in filenames_grouped[key]:
            path_to_file = os.path.join(path_to_evals_folder, bs_run)
            ner_metrics = get_zero_shot_metrics_from_file(path_to_file, dataset_name_list)
            for ds_name, ds_metrics in ner_metrics.items():
                collected_metrics[key][ds_name].append(ds_metrics)

    print(collected_metrics)

    # Combine all collected metrics into a single DataFrame
    df_list = []
    for w_wo_def, w_wo_values in collected_metrics.items():
        for ds_name, ds_values in w_wo_values.items():
            for ne_mvalues in ds_values:
                df = pd.DataFrame(ne_mvalues)
                df['Dataset'] = ds_name
                df['With_Definition'] = w_wo_def
                df_list.append(df)
    print(df_list[0])

    combined_df = pd.concat(df_list)
    print(combined_df)

    # Unpivot the DataFrame to have NE as a column
    combined_df['Metric'] = combined_df.index
    melted_df = combined_df.melt(id_vars=['Dataset', 'With_Definition', 'Metric'], var_name='NE',
                                 value_name='metric_value')
    print(melted_df)

    # Compute average and standard deviation for each NE in each dataset under each condition
    grouped_metrics = melted_df.groupby(['Dataset', 'With_Definition', 'NE', 'Metric']).agg({
        'metric_value': ['mean', 'std'],
    }).unstack()
    grouped_metrics.columns = [f'{col[2]}_{col[1]}' for col in grouped_metrics.columns]
    print(grouped_metrics.columns)
    print(grouped_metrics)

    grouped_metrics.dropna(how='all', inplace=True)
    print(grouped_metrics)

    # Reset index to make 'Dataset', 'With_Definition', and NE as columns
    grouped_metrics.reset_index(inplace=True)
    print(grouped_metrics)

    # aggregate back to dict for plotting
    avg_std_metrics = {'FalseDef': {x: {} for x in dataset_name_list}, 'TrueDef': {x: {} for x in dataset_name_list}}
    for index, row in grouped_metrics.iterrows():
        avg_std_metrics[row['With_Definition']][row['Dataset']][row['NE']] = {
            'avg_precision': row['Precision_mean'],
            'std_precision': row['Precision_std'],
            'avg_recall': row['Recall_mean'],
            'std_recall': row['Recall_std'],
            'avg_F1': row['F1_mean'],
            'std_F1': row['F1_std'],
            'support': int(row['support_mean'])
        }

    print(avg_std_metrics)

    return avg_std_metrics


def plot_avg_std_FalseTrueDef_comparison(ner_metrics_FalseDef_with_avg_std, ner_metrics_TrueDef_with_avg_std, dataset_name, ne_statistics, model_name):
    """ plot an histogram comparing metrics for each NE class False vs True Def - average and std across multiple eval runs """

    print("FalseDef labels:")
    print(sorted(ner_metrics_FalseDef_with_avg_std[dataset_name].keys()))

    x_labels = sorted(ner_metrics_TrueDef_with_avg_std[dataset_name].keys())
    print("TrueDef labels / x axis:")
    print(x_labels)


    thisds_metrics_FalseDef = [ner_metrics_FalseDef_with_avg_std[dataset_name][x]['avg_F1'] for x in sorted(ner_metrics_FalseDef_with_avg_std[dataset_name].keys())]
    thisds_metrics_FalseDef_stds = [ner_metrics_FalseDef_with_avg_std[dataset_name][x]['std_F1'] for x in sorted(ner_metrics_FalseDef_with_avg_std[dataset_name].keys())]

    thisds_metrics_TrueDef = [ner_metrics_TrueDef_with_avg_std[dataset_name][x]['avg_F1'] for x in x_labels]
    thisds_metrics_TrueDef_stds = [ner_metrics_TrueDef_with_avg_std[dataset_name][x]['std_F1'] for x in x_labels]

    plt.figure(figsize=(14, 6))
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
    plt.ylabel('per-NE AVERAGE F1 across multiple runs')
    plt.title(f'{model_name} - FalseDef vs TrueDef comparison - {dataset_name.upper()} dataset')

    # adding also support to NE label
    x_labels = [ne + '\n' + str(ner_metrics_TrueDef_with_avg_std[dataset_name][ne]['support']) for ne in x_labels]
    plt.xticks(x, x_labels, rotation=10, fontsize=6)
    for i, label in enumerate(x_labels):
        label = label.split("\n")[0]
        if label in ne_statistics[dataset_name]['novel_ne_list']:
            plt.gca().get_xticklabels()[i].set_color("red")
        elif label in ne_statistics[dataset_name]['same_ne_diff_def_list']:
            plt.gca().get_xticklabels()[i].set_color("orange")
        else:
            plt.gca().get_xticklabels()[i].set_color("black")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    """
    model_name = 'Llama2-7B-Enhanced-NonMasked_vs_Masked'

    path_to_evals_folder = '../../experiments_outputs/Llama2-7B'

    dataset_name_list = ['movie', 'restaurant', 'ai', 'literature', 'music', 'politics', 'science', 'BUSTER']

    filenames = os.listdir(path_to_evals_folder)
    filenames_grouped = {'TrueDef': [], 'FalseDef': []}

    """

    """
    for fn in filenames:
        if fn != '.DS_Store' and not os.path.isdir(os.path.join(path_to_evals_folder, fn)):
            if 'TrueDef' in fn:
                filenames_grouped['TrueDef'].append(fn)
            else:
                filenames_grouped['FalseDef'].append(fn)
    print(filenames_grouped)
    """

    """
    #filenames_grouped['FalseDef'].append('uniNEReval_LLama2_7b-FalseDef-A.txt')

    # filenames_grouped['TrueDef'].append('uniNEReval_LLama2_7b-TrueDef-A.txt')

    filenames_grouped['FalseDef'].append('uniNEReval_LLama2_7b-TrueDef-enhanced2midcp.txt')
    filenames_grouped['TrueDef'].append('uniNEReval_LLama2_7b-TrueDef-enhanced2midcp_masked_eval.txt')

    #filenames_grouped['FalseDef'].append('uniNEReval_LLama2_7b-TrueDef-A.txt')

    avg_std_metrics = collect_compute_prepare_metrics_for_plotting(path_to_evals_folder, filenames_grouped, dataset_name_list)

    # Change TrueDef NE names with FalseDef NE names if needed
    for ds_name, ds_values in avg_std_metrics['TrueDef'].items():
        new_ne_names_metrics = {}
        falseDef_NEs_list = list(avg_std_metrics['FalseDef'][ds_name].keys())
        for i, trueDef_ne in enumerate(avg_std_metrics['TrueDef'][ds_name]):
            new_ne = falseDef_NEs_list[i]
            new_ne_names_metrics[new_ne] = avg_std_metrics['TrueDef'][ds_name][trueDef_ne]
        avg_std_metrics['TrueDef'][ds_name] = new_ne_names_metrics

    print(avg_std_metrics)

    with open(os.path.join('../../experiments_outputs/PLOTS/models_avg_std_metrics', f"{model_name}_avg_std_metrics.json"), 'w') as f:
        json.dump(avg_std_metrics, f, indent=4)

    for ds_name in dataset_name_list:
        plot_avg_std_FalseTrueDef_comparison(avg_std_metrics['FalseDef'], avg_std_metrics['TrueDef'], ds_name, all_datasets_ne_statistics, model_name)
        
    """

    #get_overall_scores_for = 'novel_ne_list'
    #get_overall_scores_for = 'same_ne_diff_def_list'
    get_overall_scores_for = 'all_ne_list'
    #get_overall_scores_for = 'NEs_seen_in_finetuning_list'

    with open(os.path.join("../../experiments_outputs/PLOTS/models_avg_std_metrics/models_scores", get_overall_scores_for + '.txt'), 'w') as wfile:

        wfile.write(f"\nModels overall performance on '{get_overall_scores_for}'\n")
        dataset_name_list = ['movie', 'restaurant', 'ai', 'literature', 'music', 'politics', 'science', 'BUSTER']
        # NB: base, large, deberta are computed using our MSEQA metrics, Llama with official uniNER
        models_list = ['RoBERTa-base-MSEQA', 'RoBERTa-large-MSEQA', 'DeBERTa-XXL-MSEQA', 'Llama2-7B', 'Llama2-7B-enhanced', 'Llama2-7B-TrueDef_vs_Enhanced+MaskedEval', 'Llama2-7B-Enhanced-NonMasked_vs_Masked']  #, 'Llama2-7B-enhanced_masked_vs_NONmasked_eval']

        TrueDef_better_FalseDef_overall = {ds: 0 for ds in dataset_name_list}
        totalTagNames = 0
        for dataset_name in dataset_name_list:
            wfile.write(f"\nOn dataset {dataset_name}:")
            TrueDef_better_FalseDef_this_ds = {mname: 0 for mname in models_list}
            dataset_statistics = all_datasets_ne_statistics[dataset_name]
            if get_overall_scores_for == 'NEs_seen_in_finetuning_list':
                tagName_list = list(set(dataset_statistics['all_ne_list']) - set(dataset_statistics['novel_ne_list']) - set(dataset_statistics['same_ne_diff_def_list']))
            else:
                tagName_list = dataset_statistics[get_overall_scores_for]
            if 'misc' in tagName_list:
                tagName_list.pop(tagName_list.index('misc'))
            totalTagNames += len(tagName_list)
            wfile.write(f"\nNEs: {tagName_list}\n\n")
            for tagName in tagName_list:
                # print(tagName)
                this_tagName_metrics = {mname: {} for mname in models_list}
                for model_name in models_list:
                    with open(os.path.join('../../experiments_outputs/PLOTS/models_avg_std_metrics', f"{model_name}_avg_std_metrics.json"), 'r') as file:
                        this_model_metrics = json.load(file)
                        falseDef_avg_F1 = this_model_metrics['FalseDef'][dataset_name][tagName]['avg_F1']
                        falseDef_std_F1 = this_model_metrics['FalseDef'][dataset_name][tagName]['std_F1']
                        trueDef_avg_F1 = this_model_metrics['TrueDef'][dataset_name][tagName]['avg_F1']
                        trueDef_std_F1 = this_model_metrics['TrueDef'][dataset_name][tagName]['std_F1']
                        support = this_model_metrics['FalseDef'][dataset_name][tagName]['support']
                        this_tagName_metrics[model_name] = {
                            'false_avg_F1': falseDef_avg_F1,
                            'false_std_F1': falseDef_std_F1,
                            'true_avg_F1': trueDef_avg_F1,
                            'true_std_F1': trueDef_std_F1
                        }

                x_labels = models_list

                this_tagName_FalseDef = [this_tagName_metrics[model_name]['false_avg_F1'] for model_name in x_labels]
                this_tagName_FalseDef_std = [this_tagName_metrics[model_name]['false_std_F1'] for model_name in x_labels]

                this_tagName_TrueDef = [this_tagName_metrics[model_name]['true_avg_F1'] for model_name in x_labels]
                this_tagName_TrueDef_std = [this_tagName_metrics[model_name]['true_std_F1'] for model_name in x_labels]

                plt.figure(figsize=(14, 6))
                bar_width = 0.4
                x = np.arange(len(x_labels))

                plt.bar(x - bar_width / 2, this_tagName_FalseDef, width=0.1, label='NoDef')
                plt.errorbar(x - bar_width / 2, this_tagName_FalseDef, yerr=this_tagName_FalseDef_std, fmt='none', ecolor='black', capsize=3, elinewidth=0.05)

                # Plotting the second set of bars
                plt.bar(x + bar_width / 2, this_tagName_TrueDef, width=0.1, label='YesDef')
                plt.errorbar(x + bar_width / 2, this_tagName_TrueDef, yerr=this_tagName_TrueDef_std, fmt='none', ecolor='black', capsize=3, elinewidth=0.05)

                # Adding labels and title
                plt.xlabel('models')
                plt.ylabel('per-NE AVERAGE F1 across multiple runs')
                plt.title(f'{tagName} from {dataset_name.upper()} dataset - FalseDef vs TrueDef comparison across models \n support: {support}')

                plt.xticks(x, x_labels, rotation=10, fontsize=6)

                #plt.legend()
                #plt.show()
                matplotlib.pyplot.close()

                for model_name, this_model_this_tagName_metrics in this_tagName_metrics.items():
                    if this_model_this_tagName_metrics['true_avg_F1'] - this_model_this_tagName_metrics['false_avg_F1'] > 0:
                        TrueDef_better_FalseDef_this_ds[model_name] += 1
                    else:
                        TrueDef_better_FalseDef_this_ds[model_name] -= 1

            TrueDef_better_FalseDef_overall[dataset_name] = TrueDef_better_FalseDef_this_ds
            wfile.write(str(TrueDef_better_FalseDef_this_ds))
            #wfile.write("\n\n-----------------------------------------------------------------------------------------------------------------------\n")
            wfile.write(f"\n\n{'-'.join('' for x in str(TrueDef_better_FalseDef_this_ds))}")

        wfile.write("\n\nOverall:  ")
        TrueDef_better_FalseDef_overall = {mname: sum([values[mname] for values in TrueDef_better_FalseDef_overall.values()]) for mname in models_list}
        wfile.write(str(TrueDef_better_FalseDef_overall))

        wfile.write(f"\n\nTotal NEs of type {get_overall_scores_for}: {totalTagNames}")







