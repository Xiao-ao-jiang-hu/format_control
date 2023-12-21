import torch
from transformers import AutoTokenizer
from copy import deepcopy
import re
import os


tokenizer = AutoTokenizer.from_pretrained("/home/wangsitu/format_new/gpt2_tokenizer")


def generate_string_single(inputs):
    res_list = [[], []]
    for i in inputs:
        res_list[0].append(i[0])
        res_list[1].append(len(i[0]))
    res = ""
    for i in res_list[1]:
        res += str(i) + "$"
    res = res[:-1] + "%"
    for i in res_list[0]:
        res += str(i) + "$"

    return res[:-1]


def process_single(input_ids, max_length=1024, device="cpu"):
    batch_size = input_ids.shape[0]
    ids_list = input_ids.cpu().tolist()
    embs_list = []
    sentence_res = []
    for i in range(batch_size):
        ids = ids_list[i]
        if ids[0] == 101:
            ids = ids[1:]
        while ids[-1] == 0 or ids[-1] == 102:
            ids = ids[:-1]
        text = tokenizer.decode(ids)
        # print("text", text)
        lists = text.replace(" ", "").split("%")
        lens = lists[0].split("$")
        len_emb = []
        for sen_len in lens:
            for i in range(int(sen_len)):
                len_emb.append(int(sen_len) - i)
            len_emb.append(0)
        embs_list.append(len_emb[:-1] + [0] * (max_length - len(len_emb) + 1))
        # assert len(embs_list[0]) == max_length
        sentence_res.append(
            tokenizer.encode(lists[1])[:-1] + [0] * (max_length - len(len_emb) + 1)
        )
    print(embs_list, sentence_res)
    return torch.tensor(sentence_res, dtype=torch.long).to(device), [
        torch.tensor(embs_list, dtype=torch.long).to(device)
    ]


def generate_string(inputs):
    """
    [[sentece, yun], ...]
    """
    res_list = [[], [], [], []]
    i = 1
    for sentence in inputs:
        res_list[2].append(len(sentence[0]))
        res_list[0].append(sentence[1])
        res_list[1].append(len(inputs) - i)
        res_list[3].append(sentence[0])
        i += 1
    res = ""
    for part in res_list:
        for i in part:
            res += str(i) + "$"
        res = res[:-1] + "%"
    # print(res)
    return res[:-1]


def process_func(input_ids, max_length, device):
    batch_size = input_ids.shape[0]
    # print("ids", input_ids)
    embs_list = [[], [], []]
    sentence_res = []
    for i in range(batch_size):
        end = False
        ids = input_ids.cpu().tolist()[i]
        while ids[-1] == 0:
            ids = ids[:-1]
        if ids[-1] == 2:
            end = True
            ids = ids[1:-1]
        # print("ids", ids)
        text = tokenizer.decode(ids)
        # print("text", text)
        string_list = text.split("%")
        final_list = [string_list[i].split("$") for i in range(len(string_list) - 1)]
        # print("text:", text)
        lyric_list = string_list[-1].split("$")
        final_list.append([tokenizer.encode(sentence)[1:-1] for sentence in lyric_list])
        # print("final", final_list)
        sentence_num = len(final_list[-1])

        tone_emb = []
        for i in range(sentence_num):
            # print(final_list[0])
            try:
                tmp = [int(re.sub("\D", "", final_list[0][i])) + 2] * (
                    len(final_list[-1][i]) + 1
                )
            except Exception:
                tmp = [1] * (len(final_list[-1][i]) + 1)
            tone_emb.append(tmp)

        final_list[0] = tone_emb

        para_emb = []
        for i in range(sentence_num):
            tmp = [int(final_list[1][i])] * (len(final_list[-1][i]) + 1)
            para_emb.append(tmp)

        final_list[1] = para_emb

        len_emb = []
        for i in range(sentence_num):
            rest = int(final_list[2][i])
            tmp = [max(0, rest)]
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
                # print(final_list[j])
                tmp += i
            embs_list[j].append(tmp + [0] * (max_length - len(tmp)))

        tmp = [1]
        for i in final_list[-1]:
            tmp += i + [50000]
        sentence_id = tmp[:-1]

        # if end:
        sentence_res.append(sentence_id + [0] * (max_length - len(sentence_id)))
        # else:
        #     sentence_res.append(
        #         sentence_id + [2] + [0] * (max_length - 1 - len(sentence_id))
        #     )

    res_emb = [torch.tensor(i, dtype=torch.long).to(device) for i in embs_list]
    # print(torch.tensor(sentence_res, dtype=torch.long), res_emb)
    return torch.tensor(sentence_res, dtype=torch.long).to(device), res_emb


if __name__ == "__main__":
    import torch

    print(tokenizer.encode("[PAD]"))
    text = generate_string_single([["今天开始我要自己上厕所", 1], ["爸爸妈妈你们不要小看我", 1]])
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
    print(process_single(ids))
