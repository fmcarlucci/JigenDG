import matplotlib.pyplot as plt

def view_training(logger, title):
    fig, ax1 = plt.subplots()
    for k,v in logger.losses.items():
        ax1.plot(v, label=k)
        l = len(v)
    updates = l / len(logger.val_acc["class"])
    plt.legend()
    ax2 = ax1.twinx()
    for k,v in logger.val_acc.items():
        ax2.plot(range(0,l,int(updates)), v, label="Test %s" % k)
    plt.legend()
    plt.title(title + " last acc %.2f:" % logger.val_acc["class"][-1])
    plt.show()