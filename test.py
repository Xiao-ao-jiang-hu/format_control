import torch
from multihead_model import MultiheadConfig, MultiheadModel
from comp_tokenizer import Comp_tokenizer
from transformers import GPT2Config

tokenizer = Comp_tokenizer.from_pretrained("bpe_gpt2_tokenizer")

config = MultiheadConfig()

model = MultiheadModel(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm


print('load dataset')
import os
classes_path = os.path.expanduser('data/lyric_train.txt')
with open(classes_path, 'r', encoding='UTF-8') as f:
    data = f.readlines()
    
data_ids = [tokenizer.encode(c.strip(), length=512) for c in tqdm(data[:50])]
control_ids = [tokenizer.get_control(ids) for ids in data_ids]


print('done')


from torch import nn
from torch.autograd import Variable
import time

pre = time.time()

epoch = 20

model.to(device)
model.train()
print('start_train')
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)  # 定义优化器


cnt=0
for i in range(epoch):
    for j in tqdm(range(int(len(data_ids)/4))):
        print(data_ids[4*j][:512])
        print(control_ids[4*j][:512])
        
        id = Variable(torch.tensor([data_ids[4*j][:512]+control_ids[4*j][:512], data_ids[4*j+1][:512]+control_ids[4*j+1][:512], data_ids[4*j+2][:512]+control_ids[4*j+2][:512], data_ids[4*j+3][:512]+control_ids[4*j+3][:512]])).to(device)
        
        optimizer.zero_grad()

        loss = model(input_ids = id, labels = id).loss

        loss.backward()
        print(loss)
        optimizer.step()


