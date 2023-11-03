import torch
from transformers import AutoTokenizer, GPT2Config
from copy import deepcopy
from models.SongnetPlus import SongLMHeadModel, SongLMConfig

tokenizer = AutoTokenizer.from_pretrained("bpe_gpt2_tokenizer")
text = "12$12$4$12%1$1$2$2%11$11$11$10%今天开始我要自己上厕所$爸爸妈妈你们不要小看我$宝宝巴士教我上厕所秘诀$我等不急了我要上厕所"
ids = tokenizer.encode(text)


def process_func(input_ids):
    batch_size = input_ids.shape[0]

    embs_list = [[], [], []]
    sentence_res = []
    for i in range(batch_size):
        ids = input_ids.tolist()[i]
        ids = ids[1:-1]
        text = tokenizer.decode(ids)
        string_list = text.split("%")
        final_list = [string_list[i].split("$") for i in range(len(string_list) - 1)]
        # print("text:", text)
        lyric_list = string_list[-1].split("$")
        final_list.append([tokenizer.encode(sentence)[1:-1] for sentence in lyric_list])

        sentence_num = len(final_list[-1])

        tone_emb = []
        for i in range(sentence_num):
            tmp = [int(final_list[0][i])] * (len(final_list[-1][i]) + 1)
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
                tmp += i
            embs_list[j].append(tmp)

        tmp = [1]
        for i in final_list[-1]:
            tmp += i + [50000]
        sentence_id = tmp[:-1]
        sentence_res.append(sentence_id)

    res_emb = [torch.tensor(i, dtype=torch.long) for i in embs_list]
    return torch.tensor(sentence_res, dtype=torch.long), res_emb


# print(process_func(ids))

gpt_config = GPT2Config()

config = SongLMConfig(gpt_config, 3)
model = SongLMHeadModel(config, 3, process_func)

out = model(torch.tensor(torch.tensor([ids, ids], dtype=torch.long)))
print(out.logits)

print("ids:", ids)
with torch.no_grad():
    ## 一个简单 topp 的设定
    output_tokens = model.generate(
        inputs=torch.tensor([ids], dtype=torch.long),
        # early_stopping=True,
        # min_length=180,
        max_length=30,
        # max_length=512,
        do_sample=True,
        # no_repeat_ngram_size=4,
        # num_beams=4,
        temperature=0.9,
        # num_return_sequences=bs,
        repetition_penalty=1.1,
        top_k=0,
        top_p=0.9,
        use_cache=True,
        # bad_words_ids=[[102]]
        # length_penalty=2.0
    )
    print(tokenizer.decode(output_tokens.tolist()[0]))
