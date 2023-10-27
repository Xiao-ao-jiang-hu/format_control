import torch
from transformers import AutoTokenizer
from copy import deepcopy

tokenizer = AutoTokenizer.from_pretrained("bpe_gpt2_tokenizer")
text = "12$12$4$12%1$1$2$2%11$11$11$10%今天开始我要自己上厕所$爸爸妈妈你们不要小看我$宝宝巴士教我上厕所秘诀$我等不急了我要上厕所"
ids = tokenizer.encode(text)


def process_func(input_ids):
    text = tokenizer.decode(input_ids)
    string_list = text.split("%")
    final_list = [string_list[i].split("$") for i in range(len(string_list) - 1)]

    lyric_list = string_list[-1].split("$")
    final_list.append([tokenizer.encode(sentence)[1:-1] for sentence in lyric_list])

    sentence_num = len(final_list[-1])

    tone_emb = []
    for i in range(sentence_num):
        tmp = [int(final_list[0][i])] * len(final_list[-1][i])
        tone_emb.append(tmp)

    final_list[0] = tone_emb

    para_emb = []
    for i in range(sentence_num):
        tmp = [int(final_list[1][i])] * len(final_list[-1][i])
        para_emb.append(tmp)

    final_list[1] = para_emb

    len_emb = []
    for i in range(sentence_num):
        rest = int(final_list[2][i])
        tmp = []
        for token in final_list[-1][i]:
            token_len = len(tokenizer.decode(token))
            rest -= token_len
            tmp.append(max(0, rest))
        len_emb.append(tmp)

    final_list[2] = len_emb

    emb_list = []
    for j in range(3):
        tmp = []
        for i in final_list[j]:
            tmp += i + [50000]
        emb_list.append(torch.tensor(tmp + [2], dtype=torch.long))

    tmp = [1]
    for i in final_list[-1][:-1]:
        tmp += i + [50000]
    sentence_id = torch.tensor(tmp + final_list[j][-1] + [2], dtype=torch.long)

    return sentence_id, emb_list


print(process_func(ids))
