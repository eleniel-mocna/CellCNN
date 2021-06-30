from os.path import dirname, realpath, sep, pardir
import sys
import glob
sys.path.insert(0, dirname(realpath(__name__)) + sep + pardir + sep + "CellCNN")

from Dataset import DataDataset, DatasetSplit
from InputData import InputData
from Models import CellCNN
import matplotlib.pyplot as plt
# CSV data is in '../csv/*.csv'
# labels are in '../summary.csv'
CSV_FOLDER = "csv\\"
SUMMARY_FILE = "summary.csv"
CODE_NAME = "Kod"

CATEGORICAL = ["Pacient_._Kontrola",
                "Od_koho",
                "sex",
                "Subjective_evaluation_of_the_physitian"]
NUMERICAL = ["Age",
            "Age_of_onset",
            "Length_of_the_disease",
            "Age_at_diagnosis",
            "Years_on_Ig_treatment",
            "Dg_Delay",
            "IgG_at_Dg",
            "IgA_at_Dg",
            "IgM_at_Dg",
            "cl_namestab",
            "obins_hvs5",
            "ubins_hvs5",
            "ubins1",
            "ubins2",
            "cls"]

import tensorflow as tf
import pandas as pd
import numpy as np

def get_summary_file()->pd.DataFrame:
    summary = pd.read_csv(SUMMARY_FILE)
    summary = summary.replace("?", -1)
    summary = summary.fillna(-1)
    summary[CATEGORICAL] = summary[CATEGORICAL].astype("category")
    summary[NUMERICAL] = summary[NUMERICAL].apply(pd.to_numeric, errors="coerce")
    return summary

summary = get_summary_file()
data = []
labels = []
for i in range(len(summary)):
    code = summary.loc[i, CODE_NAME]
    for file_name in glob.glob(CSV_FOLDER+code+"*.csv"):
        panda = pd.read_csv(file_name).drop("Unnamed: 0", 1)
        data.append(DataDataset(panda.to_numpy())) # Ignore cell numbering
        panda_label = summary.loc[i, NUMERICAL]
        labels.append(panda_label.to_numpy())

labels = np.array(labels)
inp = InputData(data, labels=labels)
X,Y = inp.get_multi_cell_inputs(5000, DatasetSplit.TRAIN)
Xt,Yt = inp.get_multi_cell_inputs(1000, DatasetSplit.TEST)

model = CellCNN((None, 1000, 20), 15)
callb = model.fit(X,Y, epochs=10, validation_data=(Xt,Yt))
plt.plot(callb.history["loss"])
plt.show()
s = model.get_single_cell_model()
s.show_importance(X[0], scale=True) # This shows values only for first 2 dimensions!