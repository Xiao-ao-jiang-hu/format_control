import torch
from transformers import AutoTokenizer, GPT2Config
from copy import deepcopy
from models.SongnetPlus import SongLMHeadModel
from utils.songplus_utils import generate_string, process_func

tokenizer = AutoTokenizer.from_pretrained("bpe_gpt2_tokenizer")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
text = "12$12$4$12%1$1$2$2%11$11$11$10%今天开始我要自己上厕所$爸爸妈妈你们不要小看我$宝宝巴士教我上厕所秘诀$我等不急了我要上厕所"
ids = tokenizer.encode(
    text, max_length=1024, padding="max_length", add_special_tokens=True
)
id_no_0 = tokenizer.encode(text)

print(tokenizer.encode("$"))
# print(process_func(ids))

gpt_config = GPT2Config()

# config = SongLMConfig(gpt_config, 3)
# model = SongLMHeadModel(config, 3, process_func)

# out = model(torch.tensor(torch.tensor([ids, ids], dtype=torch.long)))
# print(out.logits)

model = SongLMHeadModel.from_pretrained(
    "/data22/private/wangsitu/model_params/result/songnet_pad/checkpoint1000",
    process_func=process_func,
    control_num=3,
).cpu()
# model = SongLMHeadModel(gpt_config, 3, process_func)
if tokenizer.pad_token is None:
    tokenizer.pad_token = torch.tensor([0], dtype=torch.long)

# tokenizer.pad_token_id = torch.tensor([0], dtype=torch.long)
print("ids:")
with torch.no_grad():
    ## 一个简单 topp 的设定
    output_tokens = model.generate(
        inputs=torch.tensor(
            [tokenizer.encode("12$12$4$12%1$1$2$2%11$11$11$10%今天开始我")[:-1]],
            dtype=torch.long,
        ),
        early_stopping=True,
        # min_length=180,
        max_length=100,
        # max_length=512,
        # do_sample=True,
        # no_repeat_ngram_size=4,
        # num_beams=4,
        # temperature=0.9,
        # num_return_sequences=bs,
        # repetition_penalty=1.1,
        top_k=0,
        top_p=0.9,
        use_cache=False,
        # pad_token_id=0,
        # bad_words_ids=[[50000]]
        # length_penalty=2.0
    )
    print(tokenizer.decode(output_tokens.tolist()[0]))
