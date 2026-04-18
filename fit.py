import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from model import ResNet18
from data import MRIdataset, collect_data, make_validation_set
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score

data_list = collect_data(centers = ["CAM"])
training_set, validation_set = make_validation_set(data_list)
val_set = collect_data(centers = ["MHA","RUMC","UKA"])
val_set = np.array(val_set)
validation_set = np.array(validation_set)
validation_set = np.concatenate([val_set,validation_set])

training_data = MRIdataset(training_set)
validation_data = MRIdataset(validation_set)

training_loader = DataLoader(training_data,batch_size=1,shuffle=True,num_workers=4,pin_memory=True)
validation_loader = DataLoader(validation_data,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
epochs = 50

warmup_steps = 300
base_lr = 1e-3
start_lr = 1e-5

def lr_lambda(step):
    if step < warmup_steps:
        return (start_lr + (base_lr - start_lr) * step / warmup_steps) / base_lr
    return 1

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

global_step = 0
best_auc = 0
best_val_loss = float("inf")
training_losses = []
val_losses = []
val_aucs = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images,labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        scheduler.step()
        global_step += 1

        total_loss += loss.item()

    model.eval()
    val_loss = 0

    all_outputs = []
    all_labels = []

    with torch.no_grad():

        for image,label in validation_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output,label)

            val_loss += loss.item()

            probs = torch.softmax(output,dim=1)
            all_outputs.append(probs.cpu())
            all_labels.append(label.cpu())

    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_auc = roc_auc_score(all_labels,all_outputs,multi_class="ovr",average="micro")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(),"checkpoints/model8.pth")
        iteration = epoch
    
    val_losses.append(val_loss / len(validation_loader))
    training_losses.append(total_loss / len(training_loader))

    # if val_loss / len(validation_loader) < best_val_loss:
    #     best_val_loss = val_loss / len(validation_loader)
    #     torch.save(model.state_dict(),"checkpoints/model6.pth")

torch.save({"train_losses":training_losses,"val_losses":val_losses,"iteration":iteration},"checkpoints/losses.pth")




