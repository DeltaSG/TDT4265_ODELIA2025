import torch
from model import ResNet18,DenseNet121
import nibabel as nib
from data import collect_data,collect_test_data,test_transforms
import numpy as np
import pandas as pd

device = torch.device("cpu")

# model = ResNet18(num_classes=3).to(device)
model = DenseNet121(num_classes=3,growth_rate=16).to(device)

model.load_state_dict(torch.load("checkpoints/model12.pth", map_location=device))

data_list = collect_test_data()
result = []

model.eval()

for data in data_list:
    images = []
    for path in data["image"]:
        img = nib.load(path).get_fdata()
        images.append(img)

    image = np.stack(images,axis=0)

    image = torch.tensor(image,dtype=torch.float32)
    image = image.unsqueeze(0)
    image = test_transforms(image)

    with torch.no_grad():
        image = image.to(device)

        output = model(image)
        prob = torch.softmax(output,dim=1)

        prob = prob.cpu().numpy()

        result.append({"ID":data["uid"],"normal":prob[0][0],"benign":prob[0][1],"malignant":prob[0][2]})

df = pd.DataFrame(result)
df.to_csv("predictions.csv",index=False)



