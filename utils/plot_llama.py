import os.path

import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import numpy as np
import re


def read_eval_outputs(path_to_file):
    """
    read txt containing evaluation outputs of different model configurations on each test Dataset
    e.g. llama2_7B_1pos_1neg_perNE_top391NEs_FalseDef-D on ai, literature, ...
    returns a Dataframe where each row represents micro-scores relative to a model config on a Dataset
    """
    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # to extract model name and dataset the scores that will follow belong to
    evaluating_pattern = re.compile(r'^Evaluating model named \'(.+?)\' on \'(.+?)\' test fold in ZERO-SHOT setting$')
    # micro-score on this Dataset
    micro_scores_pattern = re.compile(r'^([\w\s.]+) ==> micro-Precision: (\d+\.\d+), micro-Recall: (\d+\.\d+), micro-F1: (\d+\.\d+)$')

    model_name = ""
    dataset_name = ""
    number_NEs = -1
    with_definition = None
    number_samples_per_NE = -1
    df_list = []
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            model_name = evaluate_match.group(1)
            dataset_name = evaluate_match.group(2)
            print(f"{model_name} on {dataset_name} scores...")

            # extracting number of distinct NEs it has been trained on
            number_NEs_pattern = re.compile(r'top(\d+)NEs')
            number_NEs_match = number_NEs_pattern.search(model_name)
            number_NEs = int(number_NEs_match.group(1)) if number_NEs_match else -1

            # with or w/o guidelines
            with_definition = True if 'True' in model_name else False

            # number of pos training samples per NE
            number_samples_pattern = re.compile(r'llama2_7B_(\d+)pos_')
            number_samples_match = number_samples_pattern.search(model_name)
            number_samples_per_NE = int(number_samples_match.group(1)) if number_samples_match else -1

        micro_scores_match = micro_scores_pattern.match(line)
        if micro_scores_match:
            dataset_name_2 = micro_scores_match.group(1)
            # double check on dataset name
            if dataset_name_2 != dataset_name:
                raise ValueError("This dataset name differs from previously read dataset name!")
            micro_precision = float(micro_scores_match.group(2))
            micro_recall = float(micro_scores_match.group(3))
            micro_f1 = float(micro_scores_match.group(4))

            micro_scores = {
                'model': model_name,
                'dataset': dataset_name,
                'w_def': with_definition,
                'num_NEs': number_NEs,
                'samples_per_NE': number_samples_per_NE,
                'micro-Precision': micro_precision,
                'micro-Recall': micro_recall,
                'micro-F1': micro_f1
            }
            df = pd.DataFrame(micro_scores, index=['m_on_ds'])
            df.drop(columns='model', inplace=True)
            df_list.append(df)

    overall_df = pd.concat(df_list)
    return overall_df


def collect_tp_fp_from_eval_outputs(path_to_file, per_dataset_metrics=False):
    """
    collects TP/FN/FP count on each Named Entity in test, for each different model configuration
    we can then compute micro/macro scores per Dataset (i.e. micro/macro scores already computed in the txt)
    or across all datasets considering them as a single merged dataset (MIT+CrossNER+BUSTER) where all NEs have equal contribution
    """
    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # to extract model name and dataset the scores that will follow belong to
    evaluating_pattern = re.compile(r'^Evaluating model named \'(.+?)\' on \'(.+?)\' test fold in ZERO-SHOT setting$')
    # metrics on a single NE pattern
    support_pattern = re.compile(r'^([\w\s.]+) --> support: (\d+)$')
    tp_fp_fn_pattern = re.compile(r'^([\w\s.]+) --> TP: (\d+), FN: (\d+), FP: (\d+), TN: -1$')
    metrics_pattern = re.compile(r'^([\w\s.]+) --> Precision: (\d+\.\d+), Recall: (\d+\.\d+), F1: (\d+\.\d+)$')

    model_name = ""
    dataset_name = ""
    number_NEs = -1
    with_definition = None
    number_samples_per_NE = -1
    support = -1
    current_ne = ""
    tp = fn = fp = this_ne_precision = this_ne_recall = this_ne_f1 = -1
    df_list = []
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            model_name = evaluate_match.group(1)
            dataset_name = evaluate_match.group(2)
            # print(f"{model_name} on {dataset_name} scores ...")

            # extracting number of distinct NEs it has been trained on
            number_NEs_pattern = re.compile(r'top(\d+)NEs')
            number_NEs_match = number_NEs_pattern.search(model_name)
            number_NEs = int(number_NEs_match.group(1)) if number_NEs_match else -1

            # with or w/o guidelines
            with_definition = True if 'True' in model_name else False

            # number of pos training samples per NE
            number_samples_pattern = re.compile(r'llama2_7B_(\d+)pos_')
            number_samples_match = number_samples_pattern.search(model_name)
            number_samples_per_NE = int(number_samples_match.group(1)) if number_samples_match else -1

        support_match = support_pattern.match(line)
        if support_match:
            support = int(support_match.group(2))
        tp_fp_fn_pattern_match = tp_fp_fn_pattern.match(line.strip())
        if tp_fp_fn_pattern_match:
            tp = int(tp_fp_fn_pattern_match.group(2))
            fn = int(tp_fp_fn_pattern_match.group(3))
            fp = int(tp_fp_fn_pattern_match.group(4))
            if (tp + fn) != support:
                raise ValueError("TP+FN != support")
        metrics_match = metrics_pattern.match(line)
        if metrics_match:
            current_ne = metrics_match.group(1)
            if '.' in current_ne:
                current_ne = current_ne.split('.')[-1]
            this_ne_precision = float(metrics_match.group(2))
            this_ne_recall = float(metrics_match.group(3))
            this_ne_f1 = float(metrics_match.group(4))

            scores = {
                'model': model_name,
                'dataset': dataset_name,
                'w_def': with_definition,
                'num_NEs': number_NEs,
                'samples_per_NE': number_samples_per_NE,
                'test_NE': current_ne,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'this_NE_F1': this_ne_f1
            }
            df = pd.DataFrame(scores, index=['m_on_ne'])  # model on NE scores
            df = df.drop(columns='model')
            df_list.append(df)

    overall_df = pd.concat(df_list)
    #print("\nCollected the following TP/FP/FN per model configuration on each test NE:\n")
    #print(overall_df)

    # TODO: remove BUSTER if needed
    # overall_df = overall_df[overall_df['dataset'] != 'BUSTER']

    if per_dataset_metrics:
        overall_df = overall_df.groupby(['dataset', 'w_def', 'num_NEs', 'samples_per_NE']).agg(
            {'TP': 'sum', 'FP': 'sum', 'FN': 'sum', 'this_NE_F1': 'mean'}).reset_index()
    else:
        overall_df = overall_df.groupby(['w_def', 'num_NEs', 'samples_per_NE']).agg(
            {'TP': 'sum', 'FP': 'sum', 'FN': 'sum', 'this_NE_F1': 'mean'}).reset_index()

    overall_df.rename(columns={'this_NE_F1': 'macro-f1'}, inplace=True)

    # compute precision and recall
    overall_df['precision'] = 100 * overall_df['TP'] / (overall_df['TP'] + overall_df['FP'])
    overall_df['recall'] = 100 * overall_df['TP'] / (overall_df['TP'] + overall_df['FN'])
    overall_df['micro-f1'] = 2 * overall_df['precision'] * overall_df['recall'] / (overall_df['precision'] + overall_df['recall'])
    overall_df['micro-f1'].fillna(0, inplace=True)
    overall_df = overall_df.drop(columns=['precision', 'recall'])

    return overall_df


def collect_BUSTER_eval_outputs(path_to_file, num_NEs, samples_per_NE):

    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # to extract model name and dataset the scores that will follow belong to
    evaluating_pattern = re.compile(r'^Evaluating model named \'(.+?)\' on \'(.+?)\' test fold in ZERO-SHOT setting$')
    # metrics on a single NE pattern
    support_pattern = re.compile(r'^([\w\s.]+) --> support: (\d+)$')
    tp_fp_fn_pattern = re.compile(r'^([\w\s.]+) --> TP: (\d+), FN: (\d+), FP: (\d+), TN: -1$')
    metrics_pattern = re.compile(r'^([\w\s.]+) --> Precision: (\d+\.\d+), Recall: (\d+\.\d+), F1: (\d+\.\d+)$')

    model_name = ""
    dataset_name = ""
    number_NEs = -1
    with_definition = None
    number_samples_per_NE = -1
    support = -1
    current_ne = ""
    tp = fn = fp = this_ne_precision = this_ne_recall = this_ne_f1 = -1
    df_list = []
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            model_name = evaluate_match.group(1)
            dataset_name = evaluate_match.group(2)
            # print(f"{model_name} on {dataset_name} scores ...")

            # extracting number of distinct NEs it has been trained on
            number_NEs_pattern = re.compile(r'top(\d+)NEs')
            number_NEs_match = number_NEs_pattern.search(model_name)
            number_NEs = int(number_NEs_match.group(1)) if number_NEs_match else -1

            # with or w/o guidelines
            with_definition = True if 'True' in model_name else False

            # number of pos training samples per NE
            number_samples_pattern = re.compile(r'llama2_7B_(\d+)pos_')
            number_samples_match = number_samples_pattern.search(model_name)
            number_samples_per_NE = int(number_samples_match.group(1)) if number_samples_match else -1

        support_match = support_pattern.match(line)
        if support_match:
            support = int(support_match.group(2))
        tp_fp_fn_pattern_match = tp_fp_fn_pattern.match(line.strip())
        if tp_fp_fn_pattern_match:
            tp = int(tp_fp_fn_pattern_match.group(2))
            fn = int(tp_fp_fn_pattern_match.group(3))
            fp = int(tp_fp_fn_pattern_match.group(4))
            if (tp + fn) != support:
                raise ValueError("TP+FN != support")
        metrics_match = metrics_pattern.match(line)
        if metrics_match:
            current_ne = metrics_match.group(1)
            if '.' in current_ne:
                current_ne = current_ne.split('.')[-1]
            this_ne_precision = float(metrics_match.group(2))
            this_ne_recall = float(metrics_match.group(3))
            this_ne_f1 = float(metrics_match.group(4))

            scores = {
                'model': model_name,
                'dataset': dataset_name,
                'w_def': with_definition,
                'num_NEs': number_NEs,
                'samples_per_NE': number_samples_per_NE,
                'test_NE': current_ne,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'this_NE_precision': this_ne_precision,
                'this_NE_recall': this_ne_recall,
                'this_NE_F1': this_ne_f1
            }
            df = pd.DataFrame(scores, index=['m_on_ne'])  # model on NE scores
            df = df.drop(columns='model')
            df_list.append(df)

    overall_df = pd.concat(df_list)

    overall_df = overall_df[overall_df['dataset'] == 'BUSTER']
    overall_df = overall_df[overall_df['num_NEs'] == num_NEs]
    overall_df = overall_df[overall_df['samples_per_NE'] == samples_per_NE]

    return overall_df


if __name__ == '__main__':

    """ Collecting evaluations on BUSTER for best model configuration NEs=391 x 5 samples per NE """
    path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/SameInstruction/as_NEs_increase'
    run_names = ['SI', 'SI-B', 'SI-C']
    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_BUSTER_eval_outputs(
            os.path.join(path_to_eval_folder, f'inscreasing_NEs_FalseDef_{run}.txt'), num_NEs=391, samples_per_NE=5
        )
        # print(f"inscreasing_NEs_FalseDef_{run}")
        # print(eval_results_FalseDef)
        eval_results_TrueDef = collect_BUSTER_eval_outputs(
            os.path.join(path_to_eval_folder, f'inscreasing_NEs_TrueDef_{run}.txt'), num_NEs=391, samples_per_NE=5
        )
        # print(f"inscreasing_NEs_TrueDef_{run}")
        # print(eval_results_TrueDef)

        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    # all eval results with run name
    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)

    # MICRO-SCORES per run
    micro_scores_per_run = all_runs_eval_results.groupby(['run', 'w_def']).agg(
        {'TP': 'sum', 'FP': 'sum', 'FN': 'sum'}).reset_index()

    # compute precision and recall
    micro_scores_per_run['precision'] = 100 * micro_scores_per_run['TP'] / (micro_scores_per_run['TP'] + micro_scores_per_run['FP'])
    micro_scores_per_run['recall'] = 100 * micro_scores_per_run['TP'] / (micro_scores_per_run['TP'] + micro_scores_per_run['FN'])
    micro_scores_per_run['f1'] = 2 * micro_scores_per_run['precision'] * micro_scores_per_run['recall'] / (
                micro_scores_per_run['precision'] + micro_scores_per_run['recall'])
    micro_scores_per_run['f1'].fillna(0, inplace=True)
    print("on BUSTER micro-scores per run:")
    print(micro_scores_per_run)

    # now avg and std across runs for avg micro-scores
    across_runs_micro_scores = micro_scores_per_run.groupby(['w_def']).agg(
        {'precision': ['mean', np.std], 'recall': ['mean', np.std], 'f1': ['mean', np.std]}).reset_index()
    print("on BUSTER micro-scores across runs:")
    print(across_runs_micro_scores)

    print("\n\nMACRO-scores on BUSTER --> \n")

    # first averaging within a run across NEs --> MACRO SCORES
    per_run_avg_std = all_runs_eval_results.groupby(['run', 'w_def']).agg(
        {'this_NE_precision': ['mean', np.std], 'this_NE_recall': ['mean', np.std], 'this_NE_F1': ['mean', np.std]}).reset_index()
    per_run_avg_std.columns = ['run', 'w_def', 'this_NE_precision', 'this_NE_precision-std', 'this_NE_recall', 'this_NE_recall-std', 'this_NE_F1', 'this_NE_F1-std']
    per_run_avg_std.drop(columns=['this_NE_precision-std', 'this_NE_recall-std', 'this_NE_F1-std'], inplace=True)
    per_run_avg_std.rename(columns={'this_NE_precision': 'precision-mean', 'this_NE_recall': 'recall-mean', 'this_NE_F1': 'f1-mean'}, inplace=True)
    print("on BUSTER macro-scores per run (average across NEs):")
    print(per_run_avg_std)

    # now averaging and std across runs
    across_run_avg_std = per_run_avg_std.groupby(['w_def']).agg(
        {'precision-mean': ['mean', np.std], 'recall-mean': ['mean', np.std], 'f1-mean': ['mean', np.std]}).reset_index()
    print("on BUSTER macro-scores across runs:")
    print(across_run_avg_std)

    # per NE scores across runs
    results_avg_across_runs = all_runs_eval_results.groupby(['test_NE', 'w_def']).agg(
        {'this_NE_precision': ['mean', np.std], 'this_NE_recall': ['mean', np.std], 'this_NE_F1': ['mean', np.std]}).reset_index()
    # print(results_avg_across_runs)
    results_avg_across_runs.columns = ['test_NE', 'w_def', 'this_NE_precision', 'this_NE_precision-std', 'this_NE_recall', 'this_NE_recall-std', 'this_NE_F1', 'this_NE_F1-std']
    print("on BUSTER per NE scores across runs:")
    print(results_avg_across_runs)


    """ scores and std per Dataset for NEs=391 x 5 samples per NE """
    print("\n\nScores and std per Dataset for NEs=391 x 5 samples per NE --> ")
    path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/SameInstruction/as_NEs_increase'
    per_dataset_metrics = True
    run_names = ['SI', 'SI-B', 'SI-C']
    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'inscreasing_NEs_FalseDef_{run}.txt'), per_dataset_metrics)
        #print(f"inscreasing_NEs_FalseDef_{run}")
        #print(eval_results_FalseDef)
        eval_results_TrueDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'inscreasing_NEs_TrueDef_{run}.txt'), per_dataset_metrics)
        #print(f"inscreasing_NEs_TrueDef_{run}")
        #print(eval_results_TrueDef)

        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)

    # averaging runs per dataset
    results_avg_across_runs = all_runs_eval_results.groupby(['dataset', 'w_def', 'num_NEs', 'samples_per_NE']).agg(
        {'macro-f1': ['mean', np.std], 'micro-f1': ['mean', np.std]}).reset_index()
    # print(results_avg_across_runs)
    results_avg_across_runs.columns = ['dataset', 'w_def', 'num_NEs', 'samples_per_NE', 'macro-f1', 'macro-f1-std', 'micro-f1', 'micro-f1-std']
    # keeping only the ones for 391 x 5
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['num_NEs'] == 391]
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['samples_per_NE'] == 5]
    print("Per-Dataset 391 NEs x 5 samples scores:")
    results_avg_across_runs = results_avg_across_runs.drop(columns=['num_NEs', 'samples_per_NE'])
    print(results_avg_across_runs)

    # average across datasets, BUSTER excluded!
    all_runs_eval_results = all_runs_eval_results[all_runs_eval_results['dataset'] != 'BUSTER']
    all_runs_eval_results = all_runs_eval_results[all_runs_eval_results['num_NEs'] == 391]
    all_runs_eval_results = all_runs_eval_results[all_runs_eval_results['samples_per_NE'] == 5]
    all_runs_eval_results = all_runs_eval_results.drop(columns=['num_NEs', 'samples_per_NE'])
    print(all_runs_eval_results)
    per_run_avg_std = all_runs_eval_results.groupby(['run', 'w_def']).agg(
        {'macro-f1': ['mean', np.std], 'micro-f1': ['mean', np.std]}).reset_index()
    # std is across datasets for a run
    per_run_avg_std.columns = ['run', 'w_def', 'macro-f1', 'macro-f1-std', 'micro-f1', 'micro-f1-std']

    print("per_run_avg_std:")
    print(per_run_avg_std)

    # std is across datasets for a run, dropping it
    per_run_avg_std.drop(columns=['macro-f1-std', 'micro-f1-std'])
    overall_avg_std_across_runs = per_run_avg_std.groupby(['w_def']).agg(
        {'macro-f1': ['mean', np.std], 'micro-f1': ['mean', np.std]}).reset_index()
    print("AVG-STD across runs:")
    print(overall_avg_std_across_runs)

    save_images_to = '../../experiments_outputs/fewShot_experiments/plots'

    """ 
    1) Plotting FalseDef vs TrueDef as Number distinct NEs in training increase [10,20,30,50,100,200,391] with (5pos + 5neg) samples per NE
    
    Micro scores computed considering MIT+CrossNER+BUSTER as a single merged dataset (to not average on averages)
    Macro scores computed averaging F1 on each NE considering MIT+CrossNER+BUSTER as a single merged dataset
    """

    metric_to_plot_to_label = {
        'micro-f1': "micro-F1",
        'macro-f1': "MACRO-F1"
    }

    #path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/as_NEs_increase'
    path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/SameInstruction/as_NEs_increase'
    per_dataset_metrics = False
    #run_names = ['C', 'E']
    run_names = ['SI', 'SI-B', 'SI-C']
    number_distinct_NEs_list = [10, 20, 30, 50, 100, 200, 300, 391]
    # collecting FalseDef and TrueDef for all runs
    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'inscreasing_NEs_FalseDef_{run}.txt'), per_dataset_metrics)
        print(f"inscreasing_NEs_FalseDef_{run}")
        print(eval_results_FalseDef)
        eval_results_TrueDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'inscreasing_NEs_TrueDef_{run}.txt'), per_dataset_metrics)
        print(f"inscreasing_NEs_TrueDef_{run}")
        print(eval_results_TrueDef)

        # to trace where data come from
        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)

    # averaging results across runs
    #results_avg_across_runs = all_runs_eval_results.groupby(['w_def', 'num_NEs', 'samples_per_NE']).agg({'macro-f1': 'mean', 'micro-f1': 'mean'}).reset_index()
    results_avg_across_runs = all_runs_eval_results.groupby(['w_def', 'num_NEs', 'samples_per_NE']).agg(
        {'macro-f1': ['mean', np.std], 'micro-f1': ['mean', np.std]}).reset_index()

    # Renaming columns for clarity
    results_avg_across_runs.columns = ['w_def', 'num_NEs', 'samples_per_NE', 'macro-f1', 'macro-f1-std', 'micro-f1', 'micro-f1-std']

    # TODO: removing missing data for plotting
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['num_NEs'].isin(number_distinct_NEs_list)]

    print(results_avg_across_runs)

    # PLOTTING 'as Number distinct NEs increase'
    n_distinct_NEs = sorted(list(set(results_avg_across_runs['num_NEs'])))
    print(n_distinct_NEs)

    metrics_to_plot = ['micro-f1', 'macro-f1']

    # Create subplots
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 6))

    for i, metric_to_plot in enumerate(metrics_to_plot):

        TrueDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == True]
        FalseDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == False]

        # Plot subplot
        #axs[i].plot(n_distinct_NEs, FalseDef_avg_score[metric_to_plot], marker='o', label='w/o guidelines')
        #axs[i].plot(n_distinct_NEs, TrueDef_avg_score[metric_to_plot], marker='D', label='w guidelines')

        #axs[i].errorbar(n_distinct_NEs, FalseDef_avg_score[metric_to_plot], yerr=FalseDef_avg_score[metric_to_plot+'-std'], marker='o',label='w/o guidelines')
        axs[i].errorbar(n_distinct_NEs, FalseDef_avg_score[metric_to_plot], yerr=FalseDef_avg_score[metric_to_plot+'-std'], label='w/o guidelines', fmt='--o', capsize=5)
        axs[i].errorbar(n_distinct_NEs, TrueDef_avg_score[metric_to_plot], yerr=TrueDef_avg_score[metric_to_plot+'-std'], label='w guidelines', fmt='-o', capsize=5)

        #axs[i].set_xticks(n_distinct_NEs)
        axs[i].set_xscale('log')  # Using logarithmic scale for the x-axis
        n_distinct_NEs_log = np.log10(n_distinct_NEs)
        axs[i].set_xticks(10 ** n_distinct_NEs_log, n_distinct_NEs)
        axs[i].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())  # Disabling minor ticks

        axs[i].grid(axis='y', linestyle='--', color='lightgray')
        #axs[i].set_ylabel(f"avg {metric_to_plot}")
        axs[i].set_ylabel(metric_to_plot_to_label[metric_to_plot])

    # axs[0].set_title(f"Zero-Shot Evaluation runs {run_names} - MIT+CrossNER+BUSTER", fontsize=12)
    axs[0].set_title(f"Zero-Shot Evaluations on MIT+CrossNER+BUSTER", fontsize=12)
    axs[-1].legend(loc='lower right')
    axs[-1].set_xlabel('Unique NE types in training (log scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_images_to, 'IncreaseDistinctNEs.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    """ 
    2) PER-DATASET - plotting FalseDef vs TrueDef as Number distinct NEs in training increase [10,20,30,50,100,200,391] with (5pos + 5neg) samples per NE

    Micro scores within each dataset
    Macro scores computed averaging F1 on each NE within a dataset
    """
    #path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/as_NEs_increase'
    path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/SameInstruction/as_NEs_increase'
    per_dataset_metrics = True
    #run_names = ['C', 'E']
    run_names = ['SI', 'SI-B', 'SI-C']
    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'inscreasing_NEs_FalseDef_{run}.txt'), per_dataset_metrics)
        print(f"inscreasing_NEs_FalseDef_{run}")
        print(eval_results_FalseDef)
        eval_results_TrueDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'inscreasing_NEs_TrueDef_{run}.txt'), per_dataset_metrics)
        print(f"inscreasing_NEs_TrueDef_{run}")
        print(eval_results_TrueDef)

        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)

    # grouping by also on dataset name!
    results_avg_across_runs = all_runs_eval_results.groupby(['dataset', 'w_def', 'num_NEs', 'samples_per_NE']).agg(
        {'macro-f1': 'mean', 'micro-f1': 'mean'}).reset_index()
    print(results_avg_across_runs)

    # PLOTTING
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['num_NEs'].isin(number_distinct_NEs_list)]
    #results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['num_NEs'] != 300]
    metric_to_plot = 'macro-f1'
    n_distinct_NEs = sorted(list(set(results_avg_across_runs['num_NEs'])))
    print(n_distinct_NEs)

    datasets = sorted(set(results_avg_across_runs['dataset']))
    colors = plt.cm.tab10.colors[:len(datasets)]  # generate unique colors for each dataset
    plt.figure(figsize=(12, 5))
    plt.grid(axis='y', linestyle='--', color='lightgray')

    for dataset, color in zip(datasets, colors):
        # select data corresponding to the current dataset
        dataset_data = results_avg_across_runs[results_avg_across_runs['dataset'] == dataset]

        TrueDef_avg_score = dataset_data[dataset_data['w_def'] == True][metric_to_plot]
        FalseDef_avg_score = dataset_data[dataset_data['w_def'] == False][metric_to_plot]

        plt.plot(n_distinct_NEs, FalseDef_avg_score, marker='o', linestyle='--', color=color,
                 label=f'{dataset} w/o guidelines')

        plt.plot(n_distinct_NEs, TrueDef_avg_score, marker='D', linestyle='-', color=color,
                 label=f'{dataset} w/ guidelines')

    plt.xlabel('Unique NE types in training (log scale)')
    #plt.xticks(n_distinct_NEs)
    plt.ylabel(metric_to_plot_to_label[metric_to_plot])
    plt.title(f"Zero-Shot {metric_to_plot_to_label[metric_to_plot]} per-dataset MIT/CrossNER/BUSTER", fontsize=12)
    #plt.legend(loc='lower right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.xscale('log')
    n_distinct_NEs_log = np.log10(n_distinct_NEs)
    plt.xticks(10 ** n_distinct_NEs_log, n_distinct_NEs)
    plt.minorticks_off()  # Disabling minor ticks
    plt.savefig(os.path.join(save_images_to, 'IncreaseDistinctNEs_perDataset.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    """ 
    3) Plotting FalseDef vs TrueDef as Number samples per NE in training increase [1,2,3,4,5,6,8,10,20] on 391 distinct NEs

    Micro scores computed considering MIT+CrossNER+BUSTER as a single merged dataset (to not average runs on datasets averages)
    Macro scores computed averaging F1 on each NE considering MIT+CrossNER+BUSTER as a single merged dataset
    """
    per_dataset_metrics = False
    #path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/as_samples_increase_ALL'
    #run_names = ['C', 'D', 'E']
    #path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/as_samples_increase_on_50NEs'
    #run_names = ['F']
    #number_samples_per_NE_list = [1, 2, 3, 4, 5, 6, 8, 10, 20]
    path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/SameInstruction/as_samples_increase_on_50NEs'
    run_names = ['SI', 'SI-B', 'SI-C']
    number_samples_per_NE_list = [1, 2, 3, 4, 5, 6, 8, 10] #, 20]

    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'FalseDef_{run}.txt'), per_dataset_metrics)
        print(f"FalseDef_{run}")
        print(eval_results_FalseDef)
        eval_results_TrueDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'TrueDef_{run}.txt'), per_dataset_metrics)
        print(f"TrueDef_{run}")
        print(eval_results_TrueDef)

        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)
    results_avg_across_runs = all_runs_eval_results.groupby(['w_def', 'num_NEs', 'samples_per_NE']).agg(
        {'macro-f1': ['mean', np.std], 'micro-f1': ['mean', np.std]}).reset_index()
    print(results_avg_across_runs)

    results_avg_across_runs.columns = ['w_def', 'num_NEs', 'samples_per_NE', 'macro-f1', 'macro-f1-std', 'micro-f1', 'micro-f1-std']
    # PLOTTING
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['samples_per_NE'].isin(number_samples_per_NE_list)]
    n_samples_per_NE = sorted(list(set(results_avg_across_runs['samples_per_NE'])))
    print(n_samples_per_NE)

    metrics_to_plot = ['micro-f1', 'macro-f1']
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 6))

    for i, metric_to_plot in enumerate(metrics_to_plot):

        TrueDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == True]
        FalseDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == False]

        axs[i].errorbar(n_samples_per_NE, FalseDef_avg_score[metric_to_plot],
                        yerr=FalseDef_avg_score[metric_to_plot + '-std'],
                        label='w/o guidelines', fmt='--o', capsize=5)
        axs[i].errorbar(n_samples_per_NE, TrueDef_avg_score[metric_to_plot],
                        yerr=TrueDef_avg_score[metric_to_plot + '-std'],
                        label='w guidelines', fmt='-o', capsize=5)

        #axs[i].plot(n_samples_per_NE, FalseDef_avg_score[metric_to_plot], marker='o', label='w/o guidelines')
        #axs[i].plot(n_samples_per_NE, TrueDef_avg_score[metric_to_plot], marker='D', label='w guidelines')
        axs[i].set_xticks(n_samples_per_NE)
        axs[i].grid(axis='y', linestyle='--', color='lightgray')
        axs[i].set_ylabel(metric_to_plot_to_label[metric_to_plot])

    axs[0].set_title(f"Zero-Shot Evaluations on MIT+CrossNER+BUSTER", fontsize=12)
    axs[-1].legend(loc='lower right')
    axs[-1].set_xlabel('Positive samples per NE')
    plt.tight_layout()
    plt.savefig(os.path.join(save_images_to, 'IncreaseSamplesPerNE.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


    """ 
    4) PER-DATASET plot FalseDef vs TrueDef as Number samples per NE in training increase [1,2,3,4,5,6,8,10,20] on 391 distinct NEs
    Micro scores within each dataset
    Macro scores computed averaging F1 on each NE within a dataset
    """
    per_dataset_metrics = True
    #path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/as_samples_increase_ALL'
    #run_names = ['C', 'D', 'E']
    #path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/as_samples_increase_on_50NEs'
    path_to_eval_folder = '../../experiments_outputs/fewShot_experiments/SameInstruction/as_samples_increase_on_50NEs'
    #run_names = ['F']
    run_names = ['SI', 'SI-B', 'SI-C']

    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'FalseDef_{run}.txt'), per_dataset_metrics)
        print(f"FalseDef_{run}")
        print(eval_results_FalseDef)
        eval_results_TrueDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'TrueDef_{run}.txt'), per_dataset_metrics)
        print(f"TrueDef_{run}")
        print(eval_results_TrueDef)

        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)

    # grouping on dataset also!
    results_avg_across_runs = all_runs_eval_results.groupby(['dataset', 'w_def', 'num_NEs', 'samples_per_NE']).agg(
        {'macro-f1': 'mean', 'micro-f1': 'mean'}).reset_index()
    print(results_avg_across_runs)

    # PLOTTING
    metric_to_plot = 'macro-f1'
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['samples_per_NE'].isin(number_samples_per_NE_list)]
    n_samples_per_NE = sorted(list(set(results_avg_across_runs['samples_per_NE'])))
    print(n_samples_per_NE)

    datasets = sorted(set(results_avg_across_runs['dataset']))
    colors = plt.cm.tab10.colors[:len(datasets)]  # generate unique colors for each dataset
    plt.figure(figsize=(15, 5))
    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.xticks(n_samples_per_NE)

    for dataset, color in zip(datasets, colors):
        # select data corresponding to the current dataset
        dataset_data = results_avg_across_runs[results_avg_across_runs['dataset'] == dataset]

        TrueDef_avg_score = dataset_data[dataset_data['w_def'] == True][metric_to_plot]
        FalseDef_avg_score = dataset_data[dataset_data['w_def'] == False][metric_to_plot]

        plt.plot(n_samples_per_NE, FalseDef_avg_score, marker='o', linestyle='--', color=color, label=f'{dataset} w/o guidelines')

        plt.plot(n_samples_per_NE, TrueDef_avg_score, marker='D', linestyle='-', color=color, label=f'{dataset} w/ guidelines')

    plt.xlabel('Positive samples per NE')
    plt.ylabel(metric_to_plot_to_label[metric_to_plot])
    plt.title(f"Zero-Shot {metric_to_plot_to_label[metric_to_plot]} per-dataset MIT/CrossNER/BUSTER", fontsize=14)
    #plt.legend(loc='lower right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_images_to, 'IncreaseSamplesPerNE_perDataset.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


    """
    metric_to_plot = 'macro-f1'
    TrueDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == True][metric_to_plot]
    FalseDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == False][metric_to_plot]

    plt.figure(figsize=(10, 5))
    plt.xticks(n_samples_per_NE)
    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.plot(n_samples_per_NE, FalseDef_avg_score, marker='o', label='w/o guidelines')
    plt.plot(n_samples_per_NE, TrueDef_avg_score, marker='D', label='w guidelines')
    # plt.yticks(range(40, 55, 1))

    plt.xlabel('Number samples per NE')
    plt.ylabel('F1')
    plt.title(f"Zero-shot {run_names}_AVG {metric_to_plot} on MIT+CrossNER+BUSTER", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()
    """


    """
    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_D/increasing_samples_FalseDef_D.txt'
    eval_results_FalseDef_C = collect_tp_fp_from_eval_outputs(path_to_file, per_dataset_metrics=True)
    print(eval_results_FalseDef_C)
    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_D/increasing_samples_TrueDef_D.txt'
    eval_results_TrueDef_C = collect_tp_fp_from_eval_outputs(path_to_file, per_dataset_metrics=True)
    print(eval_results_TrueDef_C)

    #False_True_Def_runA = pd.concat([eval_results_FalseDef_A, eval_results_TrueDef_A])
    # False_True_Def_runA = False_True_Def_runA[False_True_Def_runA['samples_per_NE'].isin([1, 2, 3, 5, 8, 10, 20])]
    False_True_Def_runC = pd.concat([eval_results_FalseDef_C, eval_results_TrueDef_C])
    False_True_Def_runC = False_True_Def_runC[False_True_Def_runC['samples_per_NE'].isin([1, 2, 3, 5, 8, 10, 20])]
    # False_True_Def_runA['run'] = 'A'
    False_True_Def_runC['run'] = 'C'

    datasets = set(False_True_Def_runC['dataset'])
    colors = plt.cm.tab10.colors[:len(datasets)]  # generate unique colors for each dataset
    plt.figure(figsize=(10, 5))
    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.xticks(list(set(False_True_Def_runC['samples_per_NE'])))

    for dataset, color in zip(datasets, colors):
        # Select data corresponding to the current dataset
        dataset_data = False_True_Def_runC[False_True_Def_runC['dataset'] == dataset]

        # Extract necessary data for the current dataset
        n_samples_per_NE = dataset_data[dataset_data['w_def'] == False]['samples_per_NE']
        TrueDef_avg_score = dataset_data[dataset_data['w_def'] == True]['macro-f1']
        FalseDef_avg_score = dataset_data[dataset_data['w_def'] == False]['macro-f1']

        # Plot for the current dataset with solid lines
        plt.plot(n_samples_per_NE, FalseDef_avg_score, marker='o', linestyle='--', color=color,
                 label=f'{dataset} w/o guidelines')

        # Plot for the current dataset with dashed lines
        plt.plot(n_samples_per_NE, TrueDef_avg_score, marker='D', linestyle='-', color=color,
                 label=f'{dataset} w/ guidelines')

    plt.xlabel('Number samples per NE')
    plt.ylabel('avg macro-F1')
    plt.title("Zero-shot AVG macro-F1 on MIT/CrossNER/BUSTER", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()


    avg_scores_df = pd.concat([False_True_Def_runA, False_True_Def_runC])
    avg_scores_df.reset_index(drop=True, inplace=True)
    print(avg_scores_df)

    avg_scores_df = avg_scores_df.groupby(['num_NEs', 'w_def', 'samples_per_NE']).agg({
        'micro-f1': 'mean',
        'macro-f1': 'mean'
    }).reset_index()
    # avg_scores_df.rename(columns={'micro-f1': 'micro-f1-runAVG', 'macro-f1': 'macro-f1-runAVG'}, inplace=True)
    print(avg_scores_df)
    """



    """
    avg_scores_df = pd.concat([eval_results_FalseDef_C, eval_results_TrueDef_C])
    avg_scores_df = avg_scores_df[avg_scores_df['samples_per_NE'] != 30]
    #avg_scores_df = avg_scores_df[avg_scores_df['samples_per_NE'] != 5]
    avg_scores_df = avg_scores_df[avg_scores_df['samples_per_NE'] != 50]
    avg_scores_df.reset_index(drop=True, inplace=True)
    print(avg_scores_df)
    """

    """
    n_samples_per_NE = avg_scores_df[avg_scores_df['w_def'] == False]['samples_per_NE']

    TrueDef_avg_score = avg_scores_df[avg_scores_df['w_def'] == True]['macro-f1']
    FalseDef_avg_score = avg_scores_df[avg_scores_df['w_def'] == False]['macro-f1']

    plt.figure(figsize=(10, 5))
    plt.xticks(n_samples_per_NE)

    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.plot(n_samples_per_NE, FalseDef_avg_score, marker='o', label='w/o guidelines')
    plt.plot(n_samples_per_NE, TrueDef_avg_score, marker='D', label='w guidelines')
    # plt.yticks(range(40, 55, 1))

    plt.xlabel('Number samples per NE')
    plt.ylabel('avg micro-F1')
    plt.title("Zero-shot AVG micro-F1 on MIT/CrossNER", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()
    
    """

    """
    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_NEs_increase/inscreasing_NEs_FalseDef.txt'
    eval_results_FalseDef = read_eval_outputs(path_to_file)

    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_NEs_increase/inscreasing_NEs_TrueDef.txt'
    eval_results_TrueDef = read_eval_outputs(path_to_file)

    overall_df = pd.concat([eval_results_FalseDef, eval_results_TrueDef])
    overall_df = overall_df.drop(columns='model')
    # TODO: excluding BUSTER from AVG
    overall_df = overall_df[overall_df['dataset'] != 'BUSTER']
    print(overall_df)

    # Group by 'number_NEs' and compute average precision, recall, and F1
    avg_scores_df = overall_df.groupby(['num_NEs', 'w_def']).agg({
        'micro-Precision': 'mean',
        'micro-Recall': 'mean',
        'micro-F1': 'mean'
    }).reset_index()

    # Rename columns for clarity
    avg_scores_df.columns = ['num_NEs', 'w_def', 'micro-Prec-avg', 'micro-Recall-avg', 'micro-F1-avg']
    print(avg_scores_df)

    #avg_scores_df = avg_scores_df[avg_scores_df['num_NEs'] != 10]

    n_NEs = list(avg_scores_df[avg_scores_df['w_def'] == False]['num_NEs'])

    FalseDef_avg_score = list(avg_scores_df[avg_scores_df['w_def'] == False]['micro-F1-avg'])
    #FalseDef_avg_score.append(54.07)
    TrueDef_avg_score = list(avg_scores_df[avg_scores_df['w_def'] == True]['micro-F1-avg'])
    #TrueDef_avg_score.insert(0, 0.0)
    #TrueDef_avg_score.append(54.99)

    plt.figure(figsize=(10, 5))
    plt.xticks(n_NEs, rotation=0, fontsize=8)

    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.plot(n_NEs, FalseDef_avg_score, marker='o')
    plt.plot(n_NEs, TrueDef_avg_score, marker='D')
    plt.xlabel('Number distinct NEs')
    plt.ylabel('avg micro-F1')
    plt.title("Zero-Shot AVG micro-F1 on MIT/CrossNER", fontsize=14)
    plt.legend(['FalseDef', 'TrueDef'])
    plt.show()

    #difference = np.array(TrueDef_avg_score) - np.array(FalseDef_avg_score)


    """

    """
    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_A/increasing_samples_FalseDef_A.txt'
    eval_results_FalseDef_A = read_eval_outputs(path_to_file)
    print(eval_results_FalseDef_A)

    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_A/increasing_samples_TrueDef_A.txt'
    eval_results_TrueDef_A = read_eval_outputs(path_to_file)
    print(eval_results_TrueDef_A)

    overall_A_df = pd.concat([eval_results_FalseDef_A, eval_results_TrueDef_A])
    overall_A_df = overall_A_df.drop(columns='model')

    # increasing_samples_df = increasing_samples_df[increasing_samples_df['dataset'] != 'BUSTER']
    overall_A_df = overall_A_df.groupby(['samples_per_NE', 'w_def']).agg({
        'micro-Precision': 'mean',
        'micro-Recall': 'mean',
        'micro-F1': 'mean'
    }).reset_index()

    # Rename columns for clarity
    overall_A_df.columns = ['samples_per_NE', 'w_def', 'micro-Prec-avg', 'micro-Recall-avg', 'micro-F1-avg']
    # print(overall_A_df)

    overall_A_df = overall_A_df[overall_A_df['samples_per_NE'] != 30]
    # overall_D_df = overall_D_df[overall_D_df['samples_per_NE'] != 20]
    # avg_scores_df = avg_scores_df[avg_scores_df['number_samples_per_NE'] != 20]
    print(overall_A_df)


    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_C/increasing_samples_FalseDef_C.txt'
    eval_results_FalseDef_C = read_eval_outputs(path_to_file)
    print(eval_results_FalseDef_C)

    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_C/increasing_samples_TrueDef_C.txt'
    eval_results_TrueDef_C = read_eval_outputs(path_to_file)
    print(eval_results_TrueDef_C)

    overall_C_df = pd.concat([eval_results_FalseDef_C, eval_results_TrueDef_C])
    overall_C_df = overall_C_df.drop(columns='model')

    # increasing_samples_df = increasing_samples_df[increasing_samples_df['dataset'] != 'BUSTER']
    overall_C_df = overall_C_df[overall_C_df['samples_per_NE'] != 30]
    overall_C_df = overall_C_df[overall_C_df['samples_per_NE'] != 50]
    overall_C_df = overall_C_df.groupby(['samples_per_NE', 'w_def']).agg({
        'micro-Precision': 'mean',
        'micro-Recall': 'mean',
        'micro-F1': 'mean'
    }).reset_index()

    # Rename columns for clarity
    overall_C_df.columns = ['samples_per_NE', 'w_def', 'micro-Prec-avg', 'micro-Recall-avg', 'micro-F1-avg']
    print(overall_C_df)

    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_D/increasing_samples_FalseDef_D.txt'
    eval_results_FalseDef_D = read_eval_outputs(path_to_file)
    print(eval_results_FalseDef_D)

    path_to_file = '../../experiments_outputs/LLaMA_increasing_NEsSamples/as_samples_increase_D/increasing_samples_TrueDef_D.txt'
    eval_results_TrueDef_D = read_eval_outputs(path_to_file)
    print(eval_results_TrueDef_D)

    overall_D_df = pd.concat([eval_results_FalseDef_D, eval_results_TrueDef_D])
    overall_D_df = overall_D_df.drop(columns='model')

    # increasing_samples_df = increasing_samples_df[increasing_samples_df['dataset'] != 'BUSTER']
    overall_D_df = overall_D_df.groupby(['samples_per_NE', 'w_def']).agg({
        'micro-Precision': 'mean',
        'micro-Recall': 'mean',
        'micro-F1': 'mean'
    }).reset_index()

    # Rename columns for clarity
    overall_D_df.columns = ['samples_per_NE', 'w_def', 'micro-Prec-avg', 'micro-Recall-avg', 'micro-F1-avg']
    print(overall_D_df)

    overall_D_df = overall_D_df[overall_D_df['samples_per_NE'] != 5]
    # overall_D_df = overall_D_df[overall_D_df['samples_per_NE'] != 20]
    # avg_scores_df = avg_scores_df[avg_scores_df['number_samples_per_NE'] != 20]
    print(overall_D_df)

    # MERGING C and D to compute mean
    overall_A_df['run'] = 'A'
    overall_C_df['run'] = 'C'
    overall_D_df['run'] = 'D'
    avg_scores_df = pd.concat([overall_A_df, overall_C_df, overall_D_df])
    avg_scores_df.reset_index(drop=True, inplace=True)
    print(avg_scores_df)

    # increasing_samples_df = increasing_samples_df[increasing_samples_df['dataset'] != 'BUSTER']
    avg_scores_df = avg_scores_df.groupby(['samples_per_NE', 'w_def']).agg({
        'micro-Prec-avg': 'mean',
        'micro-Recall-avg': 'mean',
        'micro-F1-avg': 'mean'
    }).reset_index()

    # Rename columns for clarity
    avg_scores_df.columns = ['samples_per_NE', 'w_def', 'micro-Prec-avg', 'micro-Recall-avg', 'micro-F1-avg']
    print(avg_scores_df)

    #avg_scores_df = avg_scores_df[avg_scores_df['samples_per_NE'].isin([1, 3, 5, 10, 20])]

    n_samples_per_NE = avg_scores_df[avg_scores_df['w_def'] == False]['samples_per_NE']

    TrueDef_avg_score = avg_scores_df[avg_scores_df['w_def'] == True]['micro-F1-avg']
    FalseDef_avg_score = avg_scores_df[avg_scores_df['w_def'] == False]['micro-F1-avg']

    plt.figure(figsize=(10, 5))
    plt.xticks(n_samples_per_NE)
    #plt.xticks(range(len(n_samples_per_NE)))

    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.plot(n_samples_per_NE, FalseDef_avg_score, marker='o', label='w/o guidelines')
    plt.plot(n_samples_per_NE, TrueDef_avg_score, marker='D', label='w guidelines')
    # plt.yticks(range(40, 55, 1))

    plt.xlabel('Number samples per NE')
    plt.ylabel('avg micro-F1')
    plt.title("Zero-shot AVG micro-F1 on MIT/CrossNER", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()
    
    """

    """
    
    n_samples_per_NE = [1, 2, 3, 5, 8, 10, 20, 30]

    TrueDef_avg_score = [44.24, 47.81, 51.85, 55.40, 51.00, 51.14, 54.32, None]
    FalseDef_avg_score = [49.86, 52.21, 48.30, 48.99, 53.05, 52.29, 49.04, 52.74]

    plt.figure(figsize=(10, 5))
    #plt.xticks(n_samples_per_NE)
    plt.xticks(range(len(n_samples_per_NE)))

    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.plot(FalseDef_avg_score, marker='o', label='w/o guidelines')
    plt.plot(TrueDef_avg_score, marker='D', label='w guidelines')
    #plt.yticks(range(40, 55, 1))

    plt.xlabel('Number samples per NE')
    plt.ylabel('avg micro-F1')
    plt.title("Zero-shot AVG micro-F1 on MIT/CrossNER", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()
    """
