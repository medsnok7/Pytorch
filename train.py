import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader
from model import ChatBotModel

with open("intents.json","r") as f:
    intents=json.load(f)

tags=[]
xy=[]
all_words=[]

for intent in intents["intents"]:
    tag=intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
ignore_words=["?","!",":",",","."]

all_words=[stem(w) for w in all_words if w not in ignore_words]

all_words=sorted(set(all_words))
tags=sorted(set(tags))

# print(all_words)
# print(tags)

x_train=[]
y_train=[]
for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)

    label=tags.index(tag)
    y_train.append(label)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples

device="cuda" if torch.cuda.is_available() else "cpu"
    
dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=8,shuffle=True,num_workers=0)
model=ChatBotModel(len(all_words),len(y_train)).to(device)

#loss function
loss_fn=nn.CrossEntropyLoss()
#optimizer
optimizer=torch.optim.Adam(lr=0.01,params=model.parameters())

for epoch in range(1000):
    for (words,labels) in train_loader:
        words= words.to(device)
        labels=labels.to(device)

        y_pred=model(words)
        loss=loss_fn(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch % 100)==0:
        print(f"epoch {epoch}, loss= {loss.item():.6f}")
print(f"final loss {loss:.6f}")
data={
    "model_state":model.state_dict(),
    "input_size":len(x_train[0]),
    "output_size":len(y_train),
    "hidden_layer":64,
    "all_words":all_words,
    "all_tags":tags
}
file="data.pth"
torch.save(f=file,obj=data)

print(f'training complete ,file saved to {file}')

