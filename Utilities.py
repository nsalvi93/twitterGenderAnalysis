import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def pandas_printOptions():
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

# plotting ROC curve

def plot_ROC_curve(fpr, tpr, variable, algorithm):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    # plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr, tpr, label=variable)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for '+ algorithm )
    plt.legend(loc='best')
    plt.show()
