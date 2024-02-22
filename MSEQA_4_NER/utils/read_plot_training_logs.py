import os
import re
import matplotlib.pyplot as plt

def extract_f1_values(logs):
    toread_start = logs.find('EPOCH 1 training ...')
    toread_end = logs.find('EVALUATING ON TEST SET ...')

    pattern = r'F1: (\d+\.\d+)'
    #pattern = r'BATCH STEP \d+ validation loss (\d+\.\d+)'
    f1_matches = re.finditer(pattern, logs)
    f1_values = []

    for match in f1_matches:
        index = match.start(1)
        if toread_start < index < toread_end:
            f1_values.append(float(match.group(1)))

    return f1_values

def extract_lr_values(logs):
    toread_start = logs.find('EPOCH 1 training ...')
    toread_end = logs.find('EVALUATING ON TEST SET ...')

    pattern = r'Current Learning Rate: (\d+\.\d+e[+-]\d+)'
    lr_matches = re.finditer(pattern, logs)
    lr_values = []

    for match in lr_matches:
        index = match.start(1)
        if toread_start < index < toread_end:
            lr_values.append(float(match.group(1)))

    return lr_values


def plot_f1_trend(f1_values, eval_every_batch_steps, title, lr_values=None):
    batches = [i * eval_every_batch_steps for i in range(1, len(f1_values)+1)]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax1.plot(batches, f1_values, marker='o', label='F1 Score')
    ax1.set_ylabel('F1 Score')
    ax1.legend()

    ax2.plot(batches, lr_values, marker='o', color='orange', label='Learning Rate')
    ax2.set_xlabel('Batch Step')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()

    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':

    path_to_training_logs = '../../../experiments_outputs/training_logs'
    #training_txt_filename = 'baseline_3/train_noDef_base.txt'
    training_txt_filename = 'gradient_clipping/train_yesDef_large_1epoch_yesclipping.txt'

    # read txt file with training logs
    with open(os.path.join(path_to_training_logs, training_txt_filename), 'r') as file:
        logs = file.read()

    EVALUATE_EVERY_N_STEPS = int(list(re.finditer('EVALUATE_EVERY_N_STEPS: (\d+)', logs))[0].group(1))
    BATCH_SIZE = int(list(re.finditer('BATCH_SIZE: (\d+)', logs))[0].group(1))
    GRADIENT_ACCUMULATION_STEPS = int(list(re.finditer('GRADIENT_ACCUMULATION_STEPS: (\d+)', logs))[0].group(1))
    lr_scheduler_strategy = list(re.finditer('lr_scheduler_strategy: (.+)', logs))[0].group(1)
    num_train_epochs = int(list(re.finditer('num_train_epochs: (\d+)', logs))[0].group(1))

    print(EVALUATE_EVERY_N_STEPS)
    f1_values = extract_f1_values(logs)
    print(f1_values)

    lr_values = extract_lr_values(logs)
    print(lr_values)

    title = 'training_' + lr_scheduler_strategy + '_BS' + str(BATCH_SIZE) + '_GA' + str(GRADIENT_ACCUMULATION_STEPS) + '_epochs' + str(num_train_epochs)
    plot_f1_trend(f1_values, EVALUATE_EVERY_N_STEPS, title, lr_values)

    # print(logs)