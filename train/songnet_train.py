import torch
from torch import optim
from transformers import AutoTokenizer, GPT2Config
from copy import deepcopy
import sys

sys.path.append("/home/wangsitu/format_new")
from models.SongnetPlus import SongLMHeadModel
from utils.songplus_utils import generate_string, process_func
from tqdm import tqdm
import json


tokenizer = AutoTokenizer.from_pretrained("./../bpe_gpt2_tokenizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
config = GPT2Config(
    max_length=1024, vocab_size=50100, max_position_embeddings=1024, pad_token_id=0
)

model = SongLMHeadModel(config, 3, process_func).to(device)

tokenizer.add_special_tokens({"pad_token": "[UNK]"})
ids = tokenizer.encode(
    "今天开始我要自己上厕所，爸爸妈妈你们不要小看我，宝宝巴士教我上厕所秘诀，我等不急了我要上厕所。",
    max_length=1024,
    padding="max_length",
    add_special_tokens=True,
)[:1024]
print(ids)


from tqdm import tqdm


print("load dataset")
import os

tokenizer.add_special_tokens({"pad_token": "[UNK]"})
classes_path = os.path.expanduser("/home/wangsitu/pinyin/outputs/lyric_test_res")
data_ids = []
for i in range(1, 2):
    with open(classes_path + str(i) + ".jsonl", "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            data_ids.append(
                tokenizer.encode(
                    generate_string(json.loads(line)["lyric"]),
                    max_length=1024,
                    padding="max_length",
                    add_special_tokens=True,
                )[:1024]
            )
            # print(data_ids[-1])


dataset_tensor = torch.tensor(data_ids).to(device)

from torch.utils.data import DataLoader, TensorDataset

# 构建数据集和数据迭代器

train_set = TensorDataset(dataset_tensor, dataset_tensor)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)


print("done")


from torch import nn
from torch.autograd import Variable
import time

pre = time.time()

epoch = 20


model.train()
print("start_train")
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)  # 定义优化器

cnt = 1
for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = Variable(data).to(device), Variable(target).to(device)

        optimizer.zero_grad()

        loss = model(
            data,
            labels=target,
        ).loss
        # logits = model(data, labels=target).logits
        total_loss += loss
        if cnt % 1000 == 0:
            print(loss.data)

        loss.backward()
        optimizer.step()
        cnt += 1
        if batch_idx == len(train_loader) - 1:
            # 在每个 Epoch 的最后输出一下结果
            print("average loss:", total_loss / len(train_loader))

        if cnt % 1000 == 0:
            model.save_pretrained(
                "/data22/private/wangsitu/model_params/result/songnet_pad/checkpoint"
                + str(cnt)
            )
            print(
                "------ saving to "
                + "result/songnet_test/checkpoint"
                + str(cnt)
                + " ------"
            )

print("训练时间：", time.time() - pre)
model.save_pretrained("/data22/private/wangsitu/model_params/songnet_pad/final")
