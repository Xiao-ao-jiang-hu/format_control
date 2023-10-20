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
    
data_ids = [tokenizer.encode(c.strip(), length=512) for c in tqdm(data)]
control_ids = [tokenizer.get_control(c.strip(), length = 512) for c in tqdm(data)]
target_ids = []
for lyric in tqdm(data):
    ids = tokenizer.get_target(lyric.strip())
    target_ids.append([[c] for c in ids])

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
    total_loss = 0
    for j in tqdm(range(int(len(data_ids)/4))):

        id = Variable(torch.tensor([data_ids[4*j][:512],data_ids[4*j+1][:512],data_ids[4*j+2][:512],data_ids[4*j+3][:512]])).to(device)
        con = Variable(torch.tensor([control_ids[4*j][:512],control_ids[4*j+1][:512],control_ids[4*j+2][:512],control_ids[4*j+3][:512]])).to(device)
        tar = Variable(torch.tensor([target_ids[4*j][:512],target_ids[4*j+1][:512],target_ids[4*j+2][:512],target_ids[4*j+3][:512]], dtype=torch.float)).to(device)

        optimizer.zero_grad()

        loss = model(input_ids = id, input_controls = con, labels=id, control_labels = tar).loss
        # logits = model(data, labels=target).logits
        total_loss += loss
        if cnt % 1000 == 0:
            print(loss)

        loss.backward()
        optimizer.step()
        cnt+=1
            
        if cnt % 30000 == 0:
            model.save_pretrained("result/multi/checkpoint"+str(cnt))
            print('------ saving to '+"result/multi/checkpoint"+str(cnt)+' ------')

print('训练时间：', time.time() - pre)
model.save_pretrained("result/multi/final")
