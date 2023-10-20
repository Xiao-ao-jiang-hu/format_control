from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM
)
from add_len_layer_model import AddInfoModel
import torch
from tqdm import tqdm
import argparse

tokenizer = AutoTokenizer.from_pretrained("bpe_gpt2_tokenizer")
len_test = []
zeros = []
for i in range(20):
    zeros.append(0)
    
for i in range(50500):
    len_test.append(zeros)
    
for i in range(50500):
    if i != 1 and i != 9212 and i != 538 and i <50000:
        len_test[i][len(tokenizer.decode(i))] = 1
    else:
        len_test[i][0] = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.sep_token is None:
    tokenizer.sep_token = tokenizer.eos_token
eos_token = tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)[0]
sep_token = tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)[0]
eos_token_id = torch.tensor(eos_token, dtype=torch.long)
sep_token_id = torch.tensor(sep_token, dtype=torch.long)

## bert 的 sep 与 eos 均为 [SEP] - 102
model = AddInfoModel.from_pretrained("result/add_len_layer/final", pad_token_id=sep_token,  eos_token_id=eos_token, token_len=torch.tensor(len_test, dtype=torch.float32).to(device)).to(device).eval()
input_id = torch.tensor([tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1], dtype=torch.long).unsqueeze(0).to(device)
#attn_mask = torch.ones(input_ids.shape, dtype=torch.long)

class Generator():
    
    def generate_single_segment(self, input_ids, add_length:int):
        input_ids = input_ids.tolist()
        input_ids.append(add_length+50000)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            ## 一个简单 topp 的设定
            output_tokens = model.generate(
                inputs=input_ids,
                #early_stopping=True,
                # min_length=180,
                max_length=min(add_length+input_ids.shape[1]+10,1024),
                #max_length=512,
                do_sample=True,
                #no_repeat_ngram_size=4,
                # num_beams=4,
                temperature=0.9,
                #num_return_sequences=bs,
                repetition_penalty = 1.1,
                top_k=0,
                top_p=0.9, 
                use_cache=True,
                #bad_words_ids=[[102]]
                # length_penalty=2.0
            )
            #tokens = tokenizer.decode(output_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return output_tokens[0]

    def generate_sentence(self, begin, input_format:list):
        for i in range(len(input_format)):
            #print('begin:', begin)
            begin = self.generate_single_segment(begin, input_format[i])
            #print(begin)
            
            begin = begin.tolist()
            index = 0
            for j in range(len(begin)):
                if begin[j] == 538 or begin[j] == 9212:
                    index += 1
                    if index >= i+1:
                        begin = begin[:j]+[9212]
                        break
            
            begin = torch.tensor(begin, dtype=torch.long).unsqueeze(0).to(device)[0]
        return begin
            
    
    def generate_text(self, format_list):
        res0 = []
        res = self.generate_sentence(input_id[0], format_list).tolist()
        for j in res:
            if not(j >=50001 and j <= 50099):
                res0.append(j)
        
        generated_text = tokenizer.decode(res0, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ', '')
        return generated_text
        
        
            
generator = Generator()
for i in range(8):  
    print("No", i + 1 , ":", generator.generate_text([13,4,5,1,4,1,9,1,9,8,10]))
