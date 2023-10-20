import torch
from base_model import BaseModel
from torch import optim
from transformers import AutoTokenizer, GPT2Config
from comp_tokenizer import add_tokenizer

tokenizer = add_tokenizer.from_pretrained("bpe_gpt2_tokenizer")

config = GPT2Config(vocab_size=50500)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaseModel(config)
model.to(device)

from torchviz import make_dot 




from tqdm import tqdm


print('load dataset')
import os
classes_path = os.path.expanduser('data/lyric_train.txt')
with open(classes_path, 'r', encoding='UTF-8') as f:
    data_ids = f.readlines()
    
data_ids = [tokenizer.encode(c.strip(), length=512) for c in tqdm(data_ids)]


dataset_tensor = torch.tensor(data_ids)

from torch.utils.data import DataLoader, TensorDataset

# 构建数据集和数据迭代器

train_set = TensorDataset(dataset_tensor,
                          dataset_tensor)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set,
                          batch_size=4,
                          shuffle=False)


print('done')


from torch import nn
from torch.autograd import Variable
import time

pre = time.time()

epoch = 20


model.train()
print('start_train')
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)  # 定义优化器


cnt=0
for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = Variable(data).to(device), Variable(
            target).to(device)

        optimizer.zero_grad()

        loss = model(data, labels=target).loss
        # logits = model(data, labels=target).logits
        total_loss += loss
        if cnt % 1000 == 0:
            print(loss)

        loss.backward()
        optimizer.step()
        cnt+=1
        if batch_idx == len(train_loader) - 1:
            # 在每个 Epoch 的最后输出一下结果
            print('average loss:', total_loss / len(train_loader))
            
        if cnt % 30000 == 0:
            model.save_pretrained("result/base/checkpoint"+str(cnt))
            print('------ saving to '+"result/base/checkpoint"+str(cnt)+' ------')

print('训练时间：', time.time() - pre)
model.save_pretrained("result/base/final")