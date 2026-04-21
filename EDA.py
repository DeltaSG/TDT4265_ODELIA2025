from data import collect_data,collect_test_data
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import nibabel as nib
import seaborn as sns
import pandas as pd

dataset = collect_data()
dataset_test = collect_test_data()
number_of_datapoints = len(dataset) + len(dataset_test)
print(number_of_datapoints)

num_cam = 0
num_mha = 0
num_rsh = len(dataset_test)
num_rumc = 0
num_uka = 0

for datapoint in dataset:
    if datapoint["center"] == "CAM":
        num_cam += 1
    elif datapoint["center"] == "MHA":
        num_mha += 1
    elif datapoint["center"] == "RUMC":
        num_rumc += 1
    elif datapoint["center"] == "UKA":
        num_uka += 1

print(f"CAM: {num_cam}")
print(f"MHA: {num_mha}")
print(f"RSH: {num_rsh}")
print(f"RUMC: {num_rumc}")
print(f"UKA: {num_uka}")

centers = ["CAM","MHA","RSH","RUMC","UKA"]

x_list = [1,2,3,4,5]
plot_list = [num_cam,num_mha,num_rsh,num_rumc,num_uka]
plt.figure()
plt.title("Number of images counts")
plt.xlabel("Hospital")
plt.ylabel("Number of images")
plt.bar(x_list,plot_list)
plt.xticks(x_list,centers)
plt.legend([f"Total N = {number_of_datapoints}"])
plt.savefig("plots/data_dist.png")
plt.close()

path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/"

num_channels_list = []

for center in centers:
    folder = Path(path + center + "/data_unilateral")
    for datapoint in folder.iterdir():
        number_of_files = 0
        for file in datapoint.iterdir():
            number_of_files += 1
        num_channels_list.append(number_of_files)

plt.figure()
plt.hist(num_channels_list)
plt.ylabel("Number of images")
plt.title("Number of channels counts")
plt.xlabel("Number of channels")
plt.savefig("plots/channels_hist.png")
plt.close()

centers = ["CAM","MHA","RUMC","UKA"]

class0 = [0,0,0,0]
class1 = [0,0,0,0]
class2 = [0,0,0,0]

for datapoint in dataset:
    if datapoint["center"] == "CAM":
        if datapoint["label"] == 0:
            class0[0] += 1
        elif datapoint["label"] == 1:
            class1[0] += 1
        elif datapoint["label"] == 2:
            class2[0] += 1
    elif datapoint["center"] == "MHA":
        if datapoint["label"] == 0:
            class0[1] += 1
        elif datapoint["label"] == 1:
            class1[1] += 1
        elif datapoint["label"] == 2:
            class2[1] += 1
    elif datapoint["center"] == "RUMC":
        if datapoint["label"] == 0:
            class0[2] += 1
        elif datapoint["label"] == 1:
            class1[2] += 1
        elif datapoint["label"] == 2:
            class2[2] += 1
    elif datapoint["center"] == "UKA":
        if datapoint["label"] == 0:
            class0[3] += 1
        elif datapoint["label"] == 1:
            class1[3] += 1
        elif datapoint["label"] == 2:
            class2[3] += 1

plt.figure()
width = 0.25
x = np.arange(4)
plt.bar(x-width,class0,width,label="Normal")
plt.bar(x,class1,width,label="Benign")
plt.bar(x+width,class2,width,label="Malignant")
plt.xticks(x,centers)
plt.legend()
plt.title("Lesion counts")
plt.xlabel("Hospital")
plt.ylabel("Counts")
plt.savefig("plots/targets")
plt.close()

center_means = []

centers = ["CAM","MHA","RSH","RUMC","UKA"]

intensity_dicts = []

for center in centers:
    folder = Path(path + center + "/data_unilateral")
    for subfolder in folder.iterdir():
        for file in subfolder.iterdir():
            image = nib.load(file).get_fdata()
            name = Path(file.name).stem
            name = Path(name).stem
            intensity_dicts.append({"Hospital":center,"channel":name,"Mean intensity":np.mean(image),"std_intensity":np.std(image)})

df = pd.DataFrame(intensity_dicts)

plt.figure()

sns.boxplot(data=df,x="Hospital",y="Mean intensity",hue="channel")

plt.title("Intensity per hospital and channel")
plt.legend(title="Channel")
plt.savefig("plots/intensity")
plt.close()

df_counts = df.groupby(["Hospital", "channel"]).size().reset_index(name="Count")

plt.figure(figsize=(10, 5))

sns.barplot(
    data=df_counts,
    x="Hospital",
    y="Count",
    hue="channel"
)

plt.title("Number of images per hospital and channel")
plt.ylabel("Count")
plt.xlabel("Hospital")

plt.legend(title="Channel", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("plots/channel_counts_per_hospital.png")
plt.close()


