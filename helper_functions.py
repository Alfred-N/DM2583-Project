import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_result(result, n_epochs):
    train_loss_list, train_avg_loss_list, val_loss_list, train_acc_list, val_acc_list = result
    plt.plot(train_avg_loss_list)
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.title("Smooth loss")
    plt.savefig(f"results/distilbert/smooth_loss_{n_epochs}_epochs.png")
    plt.show()

    plt.plot(train_acc_list, label="Training accuracy")
    plt.plot(val_acc_list, label="Validation accuracy")
    plt.title("Accuracy vs. num epochs")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Num epochs")
    plt.savefig(f"results/distilbert/accuracy_{n_epochs}_epochs.png")
    plt.show()
    
    print("Train acc = ",train_acc_list)
    print("Val acc = ",val_acc_list)

def plot_testing_result(result):
    #TODO: add confusion matrix + plots
    pass
def print_sentiment_distribution(data,plot=False):

    freqs= pd.DataFrame()
    freqs=data.groupby("score",as_index=False).size().rename(columns={"size": "count"})
    freqs["freq"] = freqs["count"]/(len(data.index.values))

    print("Distribution of sentiments in dataset:")
    print(freqs)
    if plot:
        plt.xlabel("sentiment score")
        plt.ylabel("frequency")
        plt.bar(freqs["score"].astype(dtype=str),freqs["freq"])
        plt.show()