from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
import numpy as np
from transformers import BertTokenizer
np.random.seed(42)

BERT_PATH = "../MatSciBERT"
BERT_MAX_LEN = 512

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

def gen_golden_label(cur_batch, e1_num, e2_num, pos_list):
    e_tensor = torch.Tensor(cur_batch, e1_num, e2_num).zero_()
    if len(pos_list) == 1:
        return e_tensor
    else:
        for i in range(len(pos_list)):
            for [e1_index, e2_index] in pos_list[i]:
                e_tensor[i, e1_index, e2_index] = 1
        return e_tensor


def gen_com_mapping(token_len, token_text, ele_list, id = None):
    res = []
    for ele in ele_list:
        mapping = gen_mapping(token_len, token_text, ele)

        res.append(mapping)
    return res


def gen_mapping(token_len, token_text, element):
    if len(element) == 0:
        # return np.zeros(token_len)
        raise Exception("the defined entity len is zero")
    else:
        token_element = tokenizer.tokenize(element)
        head_index = find_head_idx(token_text, token_element)
        if head_index == -1:
            print(token_text)
            print(token_element)
           
            raise Exception("the defined element can not be found in the text")
        else:
            head = [0 for i in range(token_len+2)]
            for i in range(head_index, head_index + len(token_element)):
                head[i+1] = 1
        return head


def get_index(e_list, e):
    if e == "":
        return -1
    for i in range(len(e_list)):
        if e_list[i] == e:
            return i
    raise Exception("the defined entity can not be found in the entity list")


def gen_neg(e_list1,e_list2,pos_list):
    neg = []
    for e1 in e_list1:
        for e2 in e_list2:
            if [e1,e2] not in pos_list:
                neg.append([e1,e2])
    if len(neg) == 0:
        return np.array(neg)
    else:
        index_list = np.random.choice(np.array(range(len(neg))), len(pos_list))
        neg = np.array(neg)
        return neg[index_list]


def get_unique_entity(e_type, e_set, text):
    entity_list = []
    if e_type == 'VoC':
        if "room temperature" in text:
            entity_list.append(("room temperature", 0))

    for single in e_set:
        # index = text.lower().find(single[e_type].lower())
        index = text.lower().find(single[e_type].lower())
        if index == -1:
            print("element:", single[e_type], "len", len(single[e_type]))
            print("text:", text)
            raise Exception("the defined element can not be found in the text")
        entity_list.append((single[e_type], index))

    entity_list.sort(key=lambda x: x[1])
    entity_list = [i[0] for i in entity_list]
    entity_list_1 = list(set(entity_list))
    entity_list_1.sort(key=entity_list.index)

    if "" in entity_list_1:
        entity_list_1.remove("")

    if entity_list_1 is None:
        return []
    else:
        return entity_list_1


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
            raise Exception("the defined element can not be found in the text")
        else:
            head = [0 for i in range(token_len+2)]
            tail = [0 for i in range(token_len+2)]
            tail_index = head_index + len(token_element) - 1
            head[head_index+1] = 1
            tail[tail_index+1] = 1

        return head, tail


class MaterSetDataset_allo(Dataset):
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
        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
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
        #为了完成训练，还需要什么？各个种类实体独特个数，需要各个种类实体的mapping集合，
        #在test中，data['set']不一定是正确的内容，要给出
        if not self.is_test:
            materialdict = ins_json_data['set']
        else:
            materialdict = ins_json_data['pred_set']

        order = ["M", "F", "VoF", "C", "VoC"]

        # 提取所有独一的实体
        unique_m = get_unique_entity(order[0], materialdict,text)

        unique_f = get_unique_entity(order[1], materialdict,text)
        unique_vof = get_unique_entity(order[2], materialdict,text)
        unique_c = get_unique_entity(order[3], materialdict,text)
        unique_voc = get_unique_entity(order[4], materialdict,text)

        # 生成各个实体的mapping，具体而言就是把uniquelist中的实体分词后对应的word_piece转化成长度为为sent_len+2的01 one-hot代码
        # print("id", ins_json_data["data_id"])
        # print("pred_set", ins_json_data["pred_set"])
        # print("material", materialdict)
        # print("unique_vof", unique_vof)
        m_mapping = gen_com_mapping(token_len=len(token_text), token_text=token_text, ele_list=unique_m)
        f_mapping = gen_com_mapping(token_len=len(token_text), token_text=token_text, ele_list=unique_f, id = ins_json_data['data_id'])
        vof_mapping = gen_com_mapping(token_len=len(token_text), token_text=token_text, ele_list=unique_vof)
        voc_mapping = gen_com_mapping(token_len=len(token_text), token_text=token_text, ele_list=unique_voc)
        c_mapping = [[]]
        # unique_num = np.array([len(i) for i in [unique_m, unique_f, unique_vof, unique_c, unique_voc]])
        # 就在这里，生成正负例
        if not self.is_test:
            f_vof_pos = []
            m_vof_pos = []
            voc_vof_pos = []

            for e_set in materialdict:
                e_f = e_set["F"]
                e_vof = e_set["VoF"]
                e_m = e_set["M"]
                e_voc = e_set["VoC"]
                # 这里e_f是空的，该怎么办
                m_index = get_index(unique_m, e_m) #uniquelist 是所有的列表，而e_m是其中一个元组的实体，这样做的目的是将set内对应位置的实体与uniquelist对应起来
                voc_index = get_index(unique_voc, e_voc)
                f_index = get_index(unique_f, e_f)
                vof_index = get_index(unique_vof, e_vof)
                # 但凡有一个是空的，都不能加入正例中
                if f_index != -1 and vof_index != -1:
                    f_vof_pos.append([f_index, vof_index])
                elif m_index != -1:
                    m_vof_pos.append([m_index, vof_index])
                elif voc_index != -1:
                    voc_vof_pos.append([voc_index, vof_index])
            # 负例的生成可能有点问题
            f_vof_neg, m_vof_neg, voc_vof_neg = gen_neg(unique_f, unique_vof, f_vof_pos), gen_neg(unique_m, unique_vof, m_vof_pos), gen_neg(unique_voc, unique_vof, voc_vof_pos)
            pred_set = {}
        else:
            #这里需要加一个pred_set
            pred_set = ins_json_data['pred_set']
            f_vof_neg, m_vof_neg, voc_vof_neg = np.zeros(len(token_text)), np.zeros(len(token_text)), np.zeros(len(token_text))
            f_vof_pos, m_vof_pos, voc_vof_pos = np.zeros(len(token_text)), np.zeros(len(token_text)), np.zeros(len(token_text))

        return (encodings['input_ids'].flatten(), encodings['attention_mask'].flatten(), token_text_len, ins_json_data['set'], pred_set, token_text, text, f_vof_pos, m_vof_pos, voc_vof_pos, f_vof_neg, m_vof_neg, voc_vof_neg,
                m_mapping, f_mapping, vof_mapping, voc_mapping, c_mapping, ins_json_data["data_id"])


def cmed_collate_fn(batch):
    # batch = list(filter(lambda x: x is not None, batch))
    # batch.sort(key=lambda x: x[2], reverse=True)
    (input_ids, attention_mask, token_text_len, goldenset, predset, token_text, text, f_vof_pos, m_vof_pos, voc_vof_pos, f_vof_neg, m_vof_neg, voc_vof_neg,
                m_mapping, f_mapping, vof_mapping, voc_mapping, c_mapping,id) = zip(*batch) # 解压缩
    cur_batch = len(batch)
    max_text_len = max(token_text_len) + 2
    batch_inputs_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    for i in range(cur_batch):
        batch_inputs_ids[i, :token_text_len[i]+2].copy_(input_ids[i])
        batch_masks[i, :token_text_len[i]+2].copy_(attention_mask[i])

    for i in range(cur_batch):
        for j in range(len(f_mapping[i])):

            cur_len = len(f_mapping[i][j])
            f_mapping[i][j] = f_mapping[i][j] + [0 for i in range(max_text_len - cur_len)]

        for p in range(len(m_mapping[i])):
            cur_len = len(m_mapping[i][p])
            m_mapping[i][p] = m_mapping[i][p] + [0 for i in range(max_text_len - cur_len)]

        for f in range(len(vof_mapping[i])):
            cur_len = len(vof_mapping[i][f])
            vof_mapping[i][f] = vof_mapping[i][f] + [0 for i in range(max_text_len - cur_len)]

        for c in range(len(voc_mapping[i])):
            cur_len = len(voc_mapping[i][c])
            voc_mapping[i][c] = voc_mapping[i][c] + [0 for i in range(max_text_len - cur_len)]

    f_mapping_max_num = []
    m_mapping_max_num = []
    vof_mapping_max_num = []
    voc_mapping_max_num = []
    for i in range(cur_batch):
        f_mapping_max_num.append(len(f_mapping[i]))
        m_mapping_max_num.append(len(m_mapping[i]))
        vof_mapping_max_num.append(len(vof_mapping[i]))
        voc_mapping_max_num.append(len(voc_mapping[i]))
    f_max_num = max(f_mapping_max_num)
    m_max_num = max(m_mapping_max_num)
    vof_max_num = max(vof_mapping_max_num)
    voc_max_num = max(voc_mapping_max_num)

    for i in range(cur_batch):
        for x in range(f_max_num - len(f_mapping[i])):
            f_mapping[i].append([0 for i in range(max_text_len)])
        for x in range(m_max_num - len(m_mapping[i])):
            m_mapping[i].append([0 for i in range(max_text_len)])
        for x in range(vof_max_num - len(vof_mapping[i])):
            vof_mapping[i].append([0 for i in range(max_text_len)])
        for x in range(voc_max_num - len(voc_mapping[i])):
            voc_mapping[i].append([0 for i in range(max_text_len)])
    # 转化pos和neg, f_vof pos and neg
    f_vof_label = gen_golden_label(cur_batch, f_max_num, vof_max_num, f_vof_pos) # b *
    m_vof_label = gen_golden_label(cur_batch, m_max_num, vof_max_num, m_vof_pos)
    voc_vof_label = gen_golden_label(cur_batch, voc_max_num, vof_max_num, voc_vof_pos)
    # batch_f_vof_pos = torch.Tensor(f_vof_pos)
    # batch_m_vof_pos = torch.Tensor(m_vof_pos)
    # batch_voc_vof_pos = torch.Tensor(voc_vof_pos)
    # batch_f_vof_neg = torch.Tensor(f_vof_neg)
    # batch_m_vof_neg = torch.Tensor(m_vof_neg)
    # batch_voc_vof_neg = torch.Tensor(voc_vof_neg)
    batch_m_mapping = torch.Tensor(m_mapping)
    batch_f_mapping = torch.Tensor(f_mapping)
    batch_vof_mapping = torch.Tensor(vof_mapping)
    batch_voc_mapping = torch.Tensor(voc_mapping)
    batch_c_mapping = torch.Tensor(c_mapping)
    if batch_m_mapping.shape[1] == 0:
        batch_m_mapping = torch.zeros((batch_m_mapping.shape[0],0,max_text_len))
    if batch_f_mapping.shape[1] == 0:
        batch_f_mapping = torch.zeros((batch_f_mapping.shape[0],0,max_text_len))
    if batch_vof_mapping.shape[1] == 0:
        batch_vof_mapping = torch.zeros((batch_vof_mapping.shape[0],0,max_text_len))
    if batch_voc_mapping.shape[1] == 0:
        batch_voc_mapping = torch.zeros((batch_voc_mapping.shape[0],0,max_text_len))

    return {'input_ids': batch_inputs_ids,
            'attention_mask': batch_masks,
            'token_text': token_text,
            'set': goldenset,
            'pred_set':predset,
            'text': text,
            # 'f_vof_pos': batch_f_vof_pos,
            # 'm_vof_pos': batch_m_vof_pos,
            # 'voc_vof_pos': batch_voc_vof_pos,
            # 'f_vof_neg': batch_f_vof_neg,
            # 'm_vof_neg': batch_m_vof_neg,
            # 'voc_vof_neg': batch_voc_vof_neg,
            'f_vof_label': torch.Tensor(f_vof_label),
            'm_vof_label': torch.Tensor(m_vof_label),
            'voc_vof_label': torch.Tensor(voc_vof_label),
            'm_mapping': batch_m_mapping,
            'f_mapping': batch_f_mapping,
            'vof_mapping': batch_vof_mapping,
            'voc_mapping': batch_voc_mapping,
            'c_mapping': batch_c_mapping,
            'data_id':id
            }


def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=cmed_collate_fn):
    dataset = MaterSetDataset_allo(config, prefix, is_test, tokenizer)
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
