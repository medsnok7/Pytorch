import random
import json
import torch
from model import ChatBotModel
from nltk_utils import tokenize,bag_of_words

device="cuda" if torch.cuda.is_available() else "cpu"

with open("intents.json","r") as f:
    intents=json.load(f)
File="data.pth"
data=torch.load(File)
# print(data)


input_size=data["input_size"]
output_size=data["output_size"]
hidden_layer=data["hidden_layer"]
all_words=data["all_words"]
tags=data["all_tags"]
model_state=data["model_state"] 
model=ChatBotModel(input_size,output_size,hidden_layer).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name="same"
print("let's chat! type 'quit' to exit")
while True:
    sentence=input("You: ")
    if sentence =="quit":
        break
    sentence=tokenize(sentence)
    x=bag_of_words(sentence,all_words)
    x=x.reshape(1,x.shape[0])
    x=torch.from_numpy(x).to(device)
    output=model(x).to(device)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    if prob.item()>0.75:   
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                print(f'bot: {random.choice( intent["responses"]   )  }')
    else :
        print(f'bot: I dont understand... ')



