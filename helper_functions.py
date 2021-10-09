import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_result(result, n_epochs, model="distilbert", dataset=""):
    train_loss_list, train_avg_loss_list, val_loss_list, train_acc_list, val_acc_list = result
    fig1,ax1 = plt.subplots()
    ax1.plot(train_avg_loss_list)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    fig1.suptitle("Smooth loss")
    fig1.savefig("results/"+model+f"/smooth_loss_{n_epochs}_epochs_"+dataset+".png")
    fig1.show()
    # plt.close(fig1)

    fig2,ax2 = plt.subplots()
    ax2.plot(train_acc_list, label="Training")
    ax2.plot(val_acc_list, label="Validation")
    ax2.set_xlabel("Num epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    fig2.suptitle("Accuracy vs. num epochs")
    fig2.savefig("results/"+model+f"/accuracy_{n_epochs}_epochs_"+dataset+".png")
    fig2.show()
    # plt.close(fig2)
    
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