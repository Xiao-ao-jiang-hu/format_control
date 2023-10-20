import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
ids = tokenizer.encode("今天开始我要自己上厕所")
print(ids)
