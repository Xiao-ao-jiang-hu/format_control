from transformers import AutoTokenizer, GPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("bpe_gpt2_tokenizer")
lens = [len(tokenizer.decode(i)) for i in range(60000)]
lens[0], lens[1], lens[2], lens[3], lens[4] = tuple([0,0,0,0,0])

class Comp_tokenizer(GPT2Tokenizer):
    
    def tokenize(self, text):
        ids = self.encode(text, pad = False)
        tokens = tokenizer.decode(ids)
            
        return tokens
    
    def get_control(self, ids, pad = True, length = 1024):
        ids = ids[1:-2]
        idx_list = [-1]+[index for index, element in enumerate(ids) if (element == 9212 or element == 538)] + [len(ids)]
        result_list = []
        for i in range(len(idx_list) - 1):
            start = idx_list[i]
            end = idx_list[i+1]
            subset = ids[start+1:end]
            offset_list = []
            total_length = sum([lens[item] for item in subset])
            for j in range(len(subset)):
                offset_list.append(total_length - sum(lens[item] for item in subset[:j]))
                
            result_list+=offset_list.copy()+[0]
            
        result_list = [0]+result_list
        if pad:
            if len(result_list) >= length:
                result_list = result_list[:length]
            else:
                for i in range(length-len(result_list)):
                    result_list.append(0)
        return result_list
    
        
    def encode(self, text, pad = True, length = 1024):
        ids = tokenizer.encode(text)
        if pad:
            if len(ids) >= length:
                ids = ids[:length]
            else:
                for i in range(length-len(ids)):
                    ids.append(0)
        
        return ids
    
    def decode(self, ids):
        return tokenizer.decode(ids)



'''mt = Comp_tokenizer.from_pretrained("bpe_gpt2_tokenizer")
text = "今天开始我要自己上厕所，爸爸妈妈你们不要小看我。"
print(tokenizer.tokenize(text))
ids = mt.encode(text)

print(ids)
ct = mt.get_control(ids)

print(ct,len(ct))'''
