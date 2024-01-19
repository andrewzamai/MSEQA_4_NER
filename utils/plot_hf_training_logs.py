import os
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':

    path_to_training_logs_json = '../../../models/trainer_state_MSEQA_pileNER_TrueDef_large.json'

    with open(path_to_training_logs_json, 'r') as file:
        logs = json.load(file)['log_history']

    print(logs)

    batch_steps = []
    eval_loss_trend = []
    lr_trend = []
    for x in logs:
        if 'eval_loss' in x.keys():
            batch_steps.append(x['step'])
            eval_loss_trend.append(x['eval_loss'])
        else:
            lr_trend.append(x['learning_rate'])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax1.plot(batch_steps, eval_loss_trend, marker='o', label='Eval loss')
    ax1.set_ylabel('Eval loss')
    ax1.legend()

    ax2.plot(batch_steps, lr_trend, marker='o', color='orange', label='Learning Rate')
    ax2.set_xlabel('Batch Step')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()

    plt.suptitle("Training MSEQA model on pileNER")
    plt.show()
