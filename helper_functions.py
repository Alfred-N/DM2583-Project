import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib

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

def plot_confusion_matrix(df_true, df_predicted, title="Confusion matrix", path="results/confusion.png", cmap=plt.cm.PuBu):
    df_confusion = confusion_matrix(df_true, df_predicted)
    
    plt.matshow(df_confusion, cmap=cmap)
    #plt.title(title) TODO fix title being cut off
    plt.colorbar()

    tick_marks = np.arange(3)
    sentiments = [-1,0,1]
    plt.xticks(tick_marks, sentiments)
    plt.yticks(tick_marks, sentiments)
    
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.savefig(path)
    plt.show()

    return plt
  
def plot_data_distribution(df_sentiment, title="Distribution of sentiments"):
    index = [1,2,3]
    counts = df_sentiment.value_counts()
    plt.bar(index,counts)
    
    plt.title(title)

    plt.xticks(index,["Negative","Neutral","Positive"])
    plt.ylabel("Counts of sentiment")
    plt.xlabel("Sentiment")

    plt.savefig("results/"+title+".png")
    plt.show()

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

def plot_predicted_sentiments(csv_path, plot_dir, model_name="DistilBERT"):
    df = pd.read_csv(csv_path,lineterminator='\n')
    if "score\r" in df.columns:
        df = df.rename(columns={"score\r":"score"})
    df = df.loc[:,["created_at","id","text","score"]]
    global_freqs = df.groupby("score",as_index=False).size().rename(columns={"size":"count"})
    print(global_freqs)

    fig, axs = plt.subplots()
    axs.set_xticks(global_freqs["score"].values)
    axs.set_xticklabels(["Negative", "Neutral", "Positive"])
    axs.set_xlabel("Sentiment")
    axs.set_ylabel("Count")
    axs.bar(global_freqs["score"].values, global_freqs["count"].values)
    axs.set_title('Predicted sentiments with ' + model_name)
    plt.savefig(plot_dir+"/Total_predicted_sentiments.png")
    plt.show()
    



    print("Grouping by month ...")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["month_date"] = df["created_at"].dt.strftime("%Y-%m")
    avg_sentiment = df.groupby(["month_date"], as_index=False).mean().reset_index(drop=True).drop(columns=["id"])
    avg_sentiment["year"] = pd.to_datetime(avg_sentiment["month_date"]).dt.strftime("%Y")
    print(avg_sentiment.head(-1))

    x = avg_sentiment.index.values
    y = avg_sentiment["score"].values
    x_plot = np.arange(0,np.max(x),0.1)
    y_plot = np.interp(x_plot,x,y)


    points = np.array([x_plot, y_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.min(y_plot),np.max(y_plot))
    
    lc = LineCollection(segments, cmap="plasma", norm=norm)
    lc.set_array(y_plot)
    lc.set_linewidth(3)

    #TODO change x_ticklabels to dates

    fig2,axs2=plt.subplots()
    line = axs2.add_collection(lc)
    fig2.colorbar(line, ax=axs2)
    axs2.set_ylim([-1,1])
    axs2.set_xlim([np.min(x_plot),np.max(x_plot)])
    axs2.set_yticks([-1,0,1])
    axs2.set_yticklabels(["Negative", "Neutral", "Positive"])
    axs2.set_xlabel("Date")
    plt.savefig(plot_dir+"/Sentiments_over_time_"+model_name+".png")
    plt.show()
    