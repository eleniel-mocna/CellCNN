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
# NUMERICAL = ["Age",
#             "Age_of_onset",
#             "Length_of_the_disease",
#             "Age_at_diagnosis",
#             "Years_on_Ig_treatment",
#             "Dg_Delay",
#             "IgG_at_Dg",
#             "IgA_at_Dg",
#             "IgM_at_Dg",
#             "obins_hvs5",
#             "ubins_hvs5",
#             "ubins1",
#             "ubins2"]
NUMERICAL = [
            "Length_of_the_disease",
            "IgG_at_Dg",
            "IgA_at_Dg",
            "IgM_at_Dg",
            "obins_hvs5",
            "ubins_hvs5"]

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
print("Getting summary")
summary = get_summary_file()
data = []
labels = []
print("reading data")
for i in range(len(summary)):
    code = summary.loc[i, CODE_NAME]
    for file_name in glob.glob(CSV_FOLDER+code+"*.csv"):
        panda = pd.read_csv(file_name).drop("Unnamed: 0", 1)
        data.append(DataDataset(panda.to_numpy())) # Ignore cell numbering
        panda_label = summary.loc[i, NUMERICAL]
        labels.append(panda_label.to_numpy())
print("Generating inputs")
labels = np.array(labels)
inp = InputData(data, labels=labels)
X,Y = inp.get_multi_cell_inputs(2500, DatasetSplit.TRAIN)
Xt,Yt = inp.get_multi_cell_inputs(500, DatasetSplit.TEST)
print("Doing stuff with Model")
model = CellCNN((None, 1000, 20), [len(NUMERICAL),], conv=[16,],)
callb = model.fit(X,Y, epochs=5, validation_data=(Xt,Yt))
plt.plot(callb.history["val_loss"])
plt.show()
s = model.get_single_cell_model()
for i in range(20):
    for j in range(i+1, 20):
        s.show_importance(X[0], scale=True, filters = (1,), dimensions = (i,j)) # This shows values only for first 2 dimensions!