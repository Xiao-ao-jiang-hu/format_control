from g2pw.api import G2PWConverter
from pypinyin.contrib.tone_convert import to_finals
import zhconv
from collections import defaultdict
from transformers import AutoTokenizer, LogitsProcessorList, LogitsProcessor
from GPTSongYunAll import GPTSongTokenizer, GPTSongLMHeadModel
import torch
import copy

class VocabularySubsetLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.mask = torch.ones(tokenizer.vocab_size, dtype=torch.bool)
        self.mask[allowed_token_ids] = 0

    def __call__(self, input_ids, logits):
        logits.masked_fill_(self.mask.to(logits.device), float('-inf'))
        return logits

def is_chinese_char(char):
    """判断一个字符是否是中文字符"""
    codepoint = ord(char)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x2F00 <= codepoint <= 0x2FD5
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x2F800 <= codepoint <= 0x2FA1F
    )

def is_chinese_punctuation(char):
    """判断一个字符是否是中文标点符号"""
    codepoint = ord(char)
    return (
        0x3000 <= codepoint <= 0x303F
        or 0xFF01 <= codepoint <= 0xFF5E
        or char in {'\t', '\n', '\r'}
        or 0xFE50 <= codepoint <= 0xFE6F
    )

def is_valid(string):
    for c in string:
        if not is_chinese_char(c) or is_chinese_punctuation(c):
            return False
    return True

yun_mapper = {'a': 0, 'ia': 0, 'ua': 0, 
              'o': 1, 'uo': 1, 
              'e': 2, 'ie': 2, 've': 2, 
              'i': 3, 
              'u': 4, 
              'v': 5, 
              'ai': 6, 'uai': 6, 
              'ei': 7, 'uei': 7, #'ui': 7,
              'ao': 8, 'iao': 8, 
              'ou': 9, 'iou': 9, #'iu': 9,
              'an': 10, 'ian': 10, 'uan': 10, 'van': 10, 
              'en': 11, 'in': 11, 'uen': 11, 'vn': 11, #'un': 11,
              'ang': 12, 'iang': 12, 'uang': 12,
              'eng': 13, 'ing': 13, 'ueng': 13,
              'ong': 14, 'iong': 14,
              'er': 15}
def map_pinyin_to_ans(tone):
    if tone is None:
        return -1
    
    yun = to_finals(tone)
    if yun not in yun_mapper:
        print(tone)
        return -1
    yun = yun_mapper[yun]

    diao = tone[-1]
    if diao == '5':
        diao = 2
    elif diao == '3' or diao == '4':
        diao = 1
    elif diao == '1' or diao == '2':
        diao = 0
    return yun

class YunGenerator:
    def __init__(self, tokenizer):
        self.g2pw = G2PWConverter(
            model_dir='G2PWModel_v2/',
            style='pinyin',
            model_source=None,
            num_workers=4,
            batch_size=4,
            enable_non_tradional_chinese=True,
            turnoff_tqdm=True
        )
        self.tokenizer = tokenizer
        self.dic_chars = self.init_dict()
        self.chn_chars, self.yun_chars, self.yun_processor, self.length_dic, self.sep_char = self.get_sets()
    
    def init_dict(self):
        dic = defaultdict(set)
        for ele in self.g2pw.polyphonic_chars:
            py = map_pinyin_to_ans(self.g2pw._convert_bopomofo_to_pinyin(ele[1]))
            if py != -1:
                dic[zhconv.convert(ele[0], 'zh-cn')].add(py)
        for ele in self.g2pw.monophonic_chars:
            py = map_pinyin_to_ans(self.g2pw._convert_bopomofo_to_pinyin(ele[1]))
            if py != -1:
                dic[zhconv.convert(ele[0], 'zh-cn')].add(py)
        return dic

    def test(self, chr):
        for ele in self.yun_chars[chr]:
            print(self.tokenizer.convert_ids_to_tokens(ele), )
    
    def get_sets(self):
        vocab = self.tokenizer.get_vocab()
        chn_chars = []
        yun_chars = {i: [] for i in range(16)}
        length_dic = {}
        for token, id in vocab.items():
            if is_valid(token):
                chn_chars.append(id)
                length_dic[id] = len(token)
                for ele in self.dic_chars[token]:
                    yun_chars[ele].append(id)
        yun_chars[17] = chn_chars
        yun_processor = {}
        for key, ele in yun_chars.items():
            yun_processor[key] = VocabularySubsetLogitsProcessor(ele)
        return chn_chars, yun_chars, yun_processor, length_dic, vocab['$']
    
    def tone_detect(self, sentence_list):
        all_res = self.g2pw(sentence_list)
        final_res = []
        for res, sentence in zip(all_res, sentence_list):
            if not res or res[-1] is None:
                y = map_pinyin_to_ans(None)
            else:
                y = map_pinyin_to_ans(res[-1])
            final_res.append(y)
        return final_res
    
    def get_valid_ids(self, input_ids, yun):
        new_sens = []
        for i in range(len(input_ids)):
            new_sen = self.tokenizer.decode(input_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            new_sen = new_sen.split('$')[-1]
            new_sens.append(new_sen)
        tones = self.tone_detect(new_sens)
        return [char for char, tone in zip(input_ids, tones) if tone == yun - 1]

    
    def generate(self, model, lengths, yuns, prefix, **kwargs):
        if isinstance(yuns, int):
            yuns = [yuns] * len(lengths)
        assert len(yuns) == len(lengths)
        input_ids = [101] + [(ele + 200) * 100 + ele2 + 1 for ele, ele2 in zip(lengths, yuns)] + [105]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device='cuda').unsqueeze(0)
        regular_processor = self.yun_processor[17]
        sep_tensor = torch.tensor([self.sep_char], dtype=torch.long, device='cuda').unsqueeze(0)
        for i in range(len(lengths)):
            length = lengths[i] - 1
            print(length + len(input_ids[0]))
            input_ids = model.generate(input_ids=input_ids,
                                           logits_processor=[regular_processor],
                                           max_length=length + len(input_ids[0]), 
                                           min_length=length + len(input_ids[0]), 
                                           **kwargs)
            '''new_input_ids = model.generate(input_ids=input_ids,
                                            # logits_processor=[VocabularySubsetLogitsProcessor(valid_ids)],
                                            # logits_processor=[regular_processor],
                                            # logits_processor=[self.yun_processor[yuns[i]]],
                                            max_length=len(input_ids[0]) + 1, 
                                            min_length=len(input_ids[0]) + 1, 
                                            **new_kwargs)'''
            new_kwargs = copy.deepcopy(kwargs)
            num_sequences = 2
            new_kwargs['do_sample'] = False
            for j in range(5):
                new_kwargs['num_return_sequences'] = num_sequences
                new_kwargs['num_beams'] = num_sequences
                new_input_ids = model.generate(input_ids=input_ids,
                                            # logits_processor=[VocabularySubsetLogitsProcessor(valid_ids)],
                                            # logits_processor=[regular_processor],
                                            logits_processor=[self.yun_processor[yuns[i] - 1]],
                                            max_length=len(input_ids[0]) + 1, 
                                            min_length=len(input_ids[0]) + 1, 
                                            **new_kwargs)
                filtered_input_ids = self.get_valid_ids(new_input_ids, yuns[i])
                if len(filtered_input_ids) != 0:
                    input_ids = filtered_input_ids[0].unsqueeze(0)
                    break
                num_sequences *= 2
                if j == 4:
                    num_sequences = 1
                    new_kwargs['num_return_sequences'] = num_sequences
                    new_kwargs['num_beams'] = num_sequences
                    input_ids = model.generate(input_ids=input_ids,
                                            # logits_processor=[VocabularySubsetLogitsProcessor(valid_ids)],
                                            # logits_processor=[regular_processor],
                                            logits_processor=[self.yun_processor[i]],
                                            max_length=len(input_ids[0]) + 1, 
                                            min_length=len(input_ids[0]) + 1, 
                                            **new_kwargs)
            input_ids = torch.cat((input_ids, sep_tensor), axis=1)
        tokens = self.tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return tokens.split('<T>')[1].replace(' ', '').replace('\n', ' ').replace('$', '\n')
        
def get_model(path):
    tokenizer = GPTSongTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
    sep_token = tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)[0]
    sep_token_id = torch.tensor(sep_token, dtype=torch.long)
    model = GPTSongLMHeadModel.from_pretrained(path, pad_token_id=sep_token, eos_token_id=sep_token).to('cuda').eval()
    return tokenizer, model


model_path = '/data24/private/liwenhao/lyric/codes/output/GPT_YUN/checkpoint-90000'
tokenizer, model = get_model(model_path)
generator = YunGenerator(tokenizer)
while True:
    try:
        x = eval(input('input length>').strip())
        yun = eval(input('input yun (1-15)>').strip())
        if yun == -1:
            break
        print(x, yun)
    except :
        continue
    res = generator.generate(
        model=model,
        lengths=x,
        yuns=yun,
        prefix=None,
        early_stopping=True,
        do_sample=True,
        no_repeat_ngram_size=3,
        temperature=0.9,
        num_return_sequences=1,
        top_k=0,
        top_p=0.9, 
        use_cache=False,
        repetition_penalty=1.2)
    print(res)
