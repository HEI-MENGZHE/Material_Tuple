from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
import numpy as np
from transformers import BertTokenizer

BERT_PATH = "../MatSciBERT"
BERT_MAX_LEN = 512

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def gen_element_seq(token_len, token_text, element):
    #空字符串返回zeros
    if len(element) == 0:
        return np.zeros(token_len), np.zeros(token_len)
    else:
        token_element = tokenizer.tokenize(element)
        head_index = find_head_idx(token_text, token_element)
        if head_index == -1:
            print("element:", element)
            print("token_text", token_text)
            raise Exception("the defined element can not be found in the text")
        else:
            head = np.zeros(token_len)
            tail = np.zeros(token_len)
            tail_index = head_index + len(token_element) - 1
            head[head_index] = 1
            tail[tail_index] = 1

        return head, tail


class MaterSetDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = tokenizer
        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:500] #和输入进去的一样
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        ins_json_data = self.json_data[idx] #dict
        text = ins_json_data['text'] #解决长度问题，使用英文一般不会超出固定的长度，
        encodings = self.tokenizer.encode_plus(
            text,
            return_token_type_ids=False,
            max_length=BERT_MAX_LEN,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        token_text = self.tokenizer.tokenize(text)
        token_text_len = len(token_text)
        #如果是训练，则有元组,将元组内的元素的首尾坐标拿到手，若相应的元素是空，则dict的value为空字符，输出的序列为空序列
        if not self.is_test:
            materialdict = ins_json_data['set']
            order = ["M", "F", "VoF", "C", "VoC"]
            # 在二分类的问题中，我们想达成知道

            m_head, m_tail = gen_element_seq(token_len=len(token_text), token_text=token_text, element=materialdict[order[0]])
            f_head, f_tail = gen_element_seq(token_len=len(token_text), token_text=token_text, element=materialdict[order[1]])
            vof_head, vof_tail = gen_element_seq(token_len=len(token_text), token_text=token_text, element=materialdict[order[2]])
            c_head, c_tail = gen_element_seq(token_len=len(token_text), token_text=token_text, element=materialdict[order[3]])
            voc_head, voc_tail = gen_element_seq(token_len=len(token_text), token_text=token_text, element=materialdict[order[4]])

        else:
            m_head, m_tail = np.zeros(len(token_text)),np.zeros(len(token_text))
            f_head, f_tail = np.zeros(len(token_text)),np.zeros(len(token_text))
            vof_head, vof_tail = np.zeros(len(token_text)),np.zeros(len(token_text))
            c_head, c_tail = np.zeros(len(token_text)),np.zeros(len(token_text))
            voc_head, voc_tail = np.zeros(len(token_text)),np.zeros(len(token_text))

        return (encodings['input_ids'].flatten(), encodings['attention_mask'].flatten(), token_text_len, m_head, m_tail, f_head, f_tail,
                vof_head, vof_tail, c_head, c_tail, voc_head, voc_tail, ins_json_data['set'], token_text, text, ins_json_data['data_id'])


def cmed_collate_fn(batch): #自定义的batch生成器，将这些都转换为 tensor
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    (input_ids, attention_mask, token_text_len, m_head, m_tail, f_head, f_tail, vof_head, vof_tail, c_head, c_tail,
     voc_head, voc_tail, goldenset, token_text, text, data_id) = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(token_text_len) + 2
    batch_inputs_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_m_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_m_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_f_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_f_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_vof_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_vof_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_c_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_c_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_voc_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_voc_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    # batch_token_text = torch.Tensor(cur_batch,max_text_len).zero_()

    for i in range(cur_batch):
        offset = 1
        ori_pos = token_text_len[i]
        batch_inputs_ids[i, :token_text_len[i]+2].copy_(input_ids[i])
        batch_masks[i, :token_text_len[i]+2].copy_(attention_mask[i])
        batch_m_head[i, offset:offset + ori_pos].copy_(torch.from_numpy(m_head[i]))
        batch_m_tail[i, offset:offset + ori_pos].copy_(torch.from_numpy(m_tail[i]))
        batch_f_head[i, offset:offset + ori_pos].copy_(torch.from_numpy(f_head[i]))
        batch_f_tail[i, offset:offset + ori_pos].copy_(torch.from_numpy(f_tail[i]))
        batch_vof_head[i, offset:offset + ori_pos].copy_(torch.from_numpy(vof_head[i]))
        batch_vof_tail[i, offset:offset + ori_pos].copy_(torch.from_numpy(vof_tail[i]))
        batch_c_head[i, offset:offset + ori_pos].copy_(torch.from_numpy(c_head[i]))
        batch_c_tail[i, offset:offset + ori_pos].copy_(torch.from_numpy(c_tail[i]))
        batch_voc_head[i, offset:offset + ori_pos].copy_(torch.from_numpy(voc_head[i]))
        batch_voc_tail[i, offset:offset + ori_pos].copy_(torch.from_numpy(voc_tail[i]))
        # batch_token_text[i, :token_text_len[i]].copy_(torch.from_numpy(token_text[i]))

    return {'input_ids': batch_inputs_ids,
            'attention_mask': batch_masks,
            'm_head': batch_m_head,
            'f_head': batch_f_head,
            'vof_head': batch_vof_head,
            'c_head': batch_c_head,
            'voc_head': batch_voc_head,
            'm_tail': batch_m_tail,
            'f_tail': batch_f_tail,
            'vof_tail': batch_vof_tail,
            'c_tail': batch_c_tail,
            'voc_tail': batch_voc_tail,
            'token_text': token_text,
            'set': goldenset,
            'text': text,
            'data_id': data_id
            }


def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=cmed_collate_fn):
    dataset = MaterSetDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader


#直观来看这个函数是用来代替对batch的遍历
class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

# class CMEDDataset(Dataset):
#     def __init__(self, config, prefix, is_test, tokenizer):
#         self.config = config
#         self.prefix = prefix
#         self.is_test = is_test
#         self.tokenizer = tokenizer
#         if self.config.debug:
#             self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:500] #和输入进去的一样
#         else:
#             self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
#         self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]
#
#     def __len__(self):
#         return len(self.json_data)
#
#     def __getitem__(self, idx):
#         ins_json_data = self.json_data[idx]
#         text = ins_json_data['text']
#         # 为什么要在这里完成分词，这个分词我们要在英语中试一下
#         text = ' '.join(text.split()[:self.config.max_len])
#         tokens = self.tokenizer.tokenize(text)  # 输出分词后的列表
#         if len(tokens) > BERT_MAX_LEN:
#             tokens = tokens[: BERT_MAX_LEN]
#         text_len = len(tokens)
#
#         if not self.is_test:
#             s2ro_map = {}
#             for triple in ins_json_data['triple_list']:
#                 triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
#                 # 为什么要用[1：-1]
#                 sub_head_idx = find_head_idx(tokens, triple[0])
#                 obj_head_idx = find_head_idx(tokens, triple[2])
#                 if sub_head_idx != -1 and obj_head_idx != -1:
#                     sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
#                     if sub not in s2ro_map:
#                         s2ro_map[sub] = []
#                     s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))
#                     # s2ro key: sub head and tail; value: obj head and tail and relid
#             if s2ro_map:
#                 token_ids, segment_ids = self.tokenizer.encode(first=text)
#                 masks = segment_ids
#                 if len(token_ids) > text_len:
#                     token_ids = token_ids[:text_len]
#                     masks = masks[:text_len]
#                 token_ids = np.array(token_ids)
#                 masks = np.array(masks) + 1
#                 sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
#                 for s in s2ro_map:
#                     sub_heads[s[0]] = 1
#                     sub_tails[s[1]] = 1
#                 sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))  # 随便找一个
#                 sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
#                 sub_head[sub_head_idx] = 1
#                 sub_tail[sub_tail_idx] = 1
#                 obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros(
#                     (text_len, self.config.rel_num))
#                 for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):  # 这个是随机选择的key然后找value
#                     obj_heads[ro[0]][ro[2]] = 1  # obj_heads  text_len * rel_num : obj_head_index, rel_idx
#                     obj_tails[ro[1]][ro[2]] = 1
#                 return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
#                 ins_json_data['triple_list'], tokens
#             else:
#                 return None
#         else:
#             token_ids, segment_ids = self.tokenizer.encode(first=text)
#             masks = segment_ids
#             if len(token_ids) > text_len:
#                 token_ids = token_ids[:text_len]
#                 masks = masks[:text_len]
#             token_ids = np.array(token_ids)
#             masks = np.array(masks) + 1  # 为毛+1啊
#             sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
#             sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
#             obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
#             return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
#             ins_json_data['triple_list'], tokens
#
#
# def cmed_collate_fn(batch): #自定义的batch生成器
#     batch = list(filter(lambda x: x is not None, batch))
#     batch.sort(key=lambda x: x[2], reverse=True)
#     token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens = zip(
#         *batch)
#     cur_batch = len(batch)
#     max_text_len = max(text_len)
#     batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
#     batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
#     batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
#     batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
#     batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
#     batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
#     batch_obj_heads = torch.Tensor(cur_batch, max_text_len, 44).zero_()
#     batch_obj_tails = torch.Tensor(cur_batch, max_text_len, 44).zero_()
#
#     for i in range(cur_batch):
#         batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
#         batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
#         batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
#         batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
#         batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
#         batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
#         batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
#         batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))
#
#     return {'token_ids': batch_token_ids,
#             'mask': batch_masks,
#             'sub_heads': batch_sub_heads,
#             'sub_tails': batch_sub_tails,
#             'sub_head': batch_sub_head,
#             'sub_tail': batch_sub_tail,
#             'obj_heads': batch_obj_heads,
#             'obj_tails': batch_obj_tails,
#             'triples': triples,
#             'tokens': tokens}
#
#
# # 要理解get loader函数
# def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=cmed_collate_fn):
#     dataset = CMEDDataset(config, prefix, is_test, tokenizer)
#     if not is_test:
#         data_loader = DataLoader(dataset=dataset,
#                                  batch_size=config.batch_size,
#                                  shuffle=True,
#                                  pin_memory=True,
#                                  num_workers=num_workers,
#                                  collate_fn=collate_fn)
#     else:
#         data_loader = DataLoader(dataset=dataset,
#                                  batch_size=1,
#                                  shuffle=False,
#                                  pin_memory=True,
#                                  num_workers=num_workers,
#                                  collate_fn=collate_fn)
#     return data_loader
#
