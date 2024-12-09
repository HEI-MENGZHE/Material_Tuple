import torch.optim as optim
from torch import nn
import os
import data_loader
import data_loader_allocation
import torch.nn.functional as F
import torch
import numpy as np
import json
import time
import re
from transformers import BertTokenizer

BERT_PATH = "../MatSciBERT"
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

class Framework(object):
    def __init__(self, con):
        self.config = con
        self.offset = 1

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
        # initialize the model
        ori_model = model_pattern(self.config)
        ori_model.cuda()

        # define the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr=self.config.learning_rate)

        # whether use multi GPU
        if self.config.multi_gpu:
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model

        # define the loss function
        def loss(gold, pred, mask):
            # [seq_len]
            pred = pred.squeeze(-1)
            los = F.binary_cross_entropy(pred, gold, reduction='none')
            if los.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            los = torch.sum(los * mask) / torch.sum(mask)
            return los

        # check the checkpoint dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        # get the data loader
        train_data_loader = data_loader.get_loader(self.config, prefix=self.config.train_prefix)
        dev_data_loader = data_loader.get_loader(self.config, prefix=self.config.dev_prefix, is_test=True)
        print("size of dataloader", len(train_data_loader))
        # other
        model.train()
        global_step = 0
        loss_sum = 0

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.max_epoch):
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()
            while data is not None:
                (pred_m_heads, pred_m_tails, pred_f_heads, pred_f_tails, pred_vof_heads, pred_vof_tails,
                 pred_c_heads, pred_c_tails, pred_voc_heads, pred_voc_tails) = model(data)

                m_heads_loss = loss(data['m_head'], pred_m_heads, data['attention_mask'])
                m_tails_loss = loss(data['m_tail'], pred_m_tails, data['attention_mask'])
                f_heads_loss = loss(data['f_head'], pred_f_heads, data['attention_mask'])
                f_tails_loss = loss(data['f_tail'], pred_f_tails, data['attention_mask'])
                vof_heads_loss = loss(data['vof_head'], pred_vof_heads, data['attention_mask'])
                vof_tails_loss = loss(data['vof_tail'], pred_vof_tails, data['attention_mask'])
                c_heads_loss = loss(data['c_head'], pred_c_heads, data['attention_mask'])
                c_tails_loss = loss(data['c_tail'], pred_c_tails, data['attention_mask'])
                voc_heads_loss = loss(data['voc_head'], pred_voc_heads, data['attention_mask'])
                voc_tails_loss = loss(data['voc_tail'], pred_voc_tails, data['attention_mask'])

                total_loss = (m_heads_loss + m_tails_loss) + (f_heads_loss + f_tails_loss) + (vof_heads_loss + vof_tails_loss) + (c_heads_loss + c_tails_loss) + (voc_heads_loss + voc_tails_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += total_loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                                 format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))
                    loss_sum = 0
                    start_time = time.time()

                data = train_data_prefetcher.next()

            if (epoch + 1) % self.config.test_epoch == 0:
                eval_start_time = time.time()
                model.eval()
                # call the test function
                precision, recall, f1_score = self.test(dev_data_loader, model)
                model.train()
                self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.
                             format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

                if f1_score >= best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall
                    self.logging("saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
                                 format(best_epoch, best_f1_score, precision, recall))
                    # save the best model
                    path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                    if not self.config.debug:
                        torch.save(ori_model.state_dict(), path)

            # manually release the unused cache
            torch.cuda.empty_cache()

        self.logging("finish training")
        # self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
        #              format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))
        self.logging("best epoch: {}, best f1: {}, precision: {}, recall: {}, total time: {}s".
                     format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))

    def train_allocation(self, model_pattern):
        # initialize the model
        ori_model = model_pattern(self.config)
        ori_model.cuda()

        # define the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr=self.config.learning_rate)

        # whether use multi GPU
        if self.config.multi_gpu:
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model

        # define the loss function
        def loss(gold, pred, mask):
            # [seq_len]
            pred = pred.squeeze(-1).flatten(start_dim=1, end_dim=-1)
            gold = gold.squeeze(-1).flatten(start_dim=1, end_dim=-1)
            los = F.binary_cross_entropy(pred, gold)
            # if los.shape != mask.shape:
            #     mask = mask.unsqueeze(-1)
            # los = torch.sum(los * mask) / torch.sum(mask)
            if torch.any(torch.isnan(los)):
                return 0
            return los

        # check the checkpoint dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        # get the data loader
        train_data_loader = data_loader_allocation.get_loader(self.config, prefix=self.config.train_prefix)
        dev_data_loader = data_loader_allocation.get_loader(self.config, prefix=self.config.dev_prefix, is_test=True)
        print("size of dataloader", len(train_data_loader))
        # other
        model.train()
        global_step = 0
        loss_sum = 0

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.max_epoch):
            train_data_prefetcher = data_loader_allocation.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()
            # for data in train_data_prefetcher:
            while data is not None:
                (score_f_vof, score_m_vof, score_voc_vof) = model(data)
                # mask = torch.empty(data['f_vof_label'].shape)
                mask = []
                f_vof_loss = loss(data['f_vof_label'], score_f_vof, mask)
                m_vof_loss = loss(data['m_vof_label'], score_m_vof, mask)
                voc_vof_loss = loss(data['voc_vof_label'], score_voc_vof, mask)
                total_loss = f_vof_loss + m_vof_loss + voc_vof_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += total_loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                                 format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))
                    loss_sum = 0
                    start_time = time.time()

                data = train_data_prefetcher.next()

            if (epoch + 1) % self.config.test_epoch == 0:
                eval_start_time = time.time()
                model.eval()
                # call the test function
                precision, recall, f1_score = self.test_allocation(dev_data_loader, model)
                model.train()
                self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.
                             format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

                if f1_score >= best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall
                    self.logging("saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
                                 format(best_epoch, best_f1_score, precision, recall))
                    # save the best model
                    path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                    if not self.config.debug:
                        torch.save(ori_model.state_dict(), path)

            # manually release the unused cache
            torch.cuda.empty_cache()

        self.logging("finish allocation training")
        # self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
        #              format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))
        self.logging("best epoch: {}, best f1: {}, precision: {}, recall: {}, total time: {}s".
                     format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))

    def test_allocation(self, test_data_loader, model):
        # train 返回的是各个实体的对应情况，test就要返回这些对应的对错的正确程度
        precision, recall, f1_score = 0, 0, 0
        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        correct_num, predict_num, gold_num = 0, 0, 0

        def get_unique_entity(e_type, e_set):
            entity_list = []
            for single in e_set:
                entity_list.append(single[e_type])
            entity_list = list(set(entity_list))
            if "" in entity_list:
                entity_list.remove("")

            if entity_list is None:
                return []
            else:
                return entity_list

        while data is not None:
            with torch.no_grad():
                (score_f_vof, score_m_vof, score_voc_vof) = model(data)
                materialdict = data['set']
                order = ["M", "F", "VoF", "C", "VoC"]

                # 提取所有特别的实体
                m_list = get_unique_entity(order[0], materialdict)
                f_list = get_unique_entity(order[1], materialdict)
                vof_list = get_unique_entity(order[2], materialdict)
                c_list = get_unique_entity(order[3], materialdict)
                voc_list = get_unique_entity(order[4], materialdict)

                # batch * e1 * e2 * 1
                if len(voc_list) != 0:
                    voc_vof = score_f_vof[0].cpu()
                    res_voc = np.argmax(voc_vof, axis=0)
                else:
                    res_voc = None

                m_vof = score_m_vof[0].cpu()
                f_vof = score_f_vof[0].cpu()
                res_f = np.argmax(f_vof, axis=0)
                res_m = np.argmax(m_vof, axis=0)
                pred = []
                for vof_index in range(len(vof_list)):
                    m = m_list[res_m[vof_index]]
                    f = f_list[res_f[vof_index]]
                    if res_voc != None:
                        voc = voc_list[res_voc[vof_index]]
                    else:
                        voc = ""
                    if len(c_list) != 0:
                        c = c_list[0]
                    else:
                        c = ""
                    pred.append([m, f, vof_list[vof_index], c, voc])
                pred = set(pred)
                label = data['set']
                gold = []
                for i in label:
                    gold.append(tuple(i.values()))
                gold = set(gold)

                correct_num += len(pred & gold)
                predict_num += len(pred)
                gold_num += len(gold)
                print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
                # 这里写一个输出的函数
                precision = correct_num / (predict_num + 1e-10)
                recall = correct_num / (gold_num + 1e-10)
                f1_score = 2 * precision * recall / (precision + recall + 1e-10)
            # 这个时候就是要稳住心神，现在还差什么，差改test的代码，我们把framework复制一下
        return precision, recall, f1_score

    def get_out(self, m_set, token, text):
        head, tail = m_set[1], m_set[2]
        head_text = token[head-self.offset].lstrip("##")
        tail_text = token[tail-self.offset].lstrip('##')
        special_token = ['[', ']', '(', ')', '{', '}']
        for i in special_token:
            if i in head_text:
                head_text = head_text.replace(i, '\\' + i)
            if i in tail_text:
                tail_text = tail_text.replace(i, '\\' + i)
        if head == tail:
            return token[head-self.offset].lstrip("##")
        else:
            pattern = re.compile(r'{}.*?{}'.format(head_text, tail_text), re.I)
            token_list = token[head: tail]
            result_pool = pattern.findall(text)

            real_can = []
            fin_can = []
            for can in result_pool:
                for tok in token_list:
                    if tok.lstrip('##') not in can.lower():
                        break
                    real_can.append(can)

            for realcan in real_can:
                flag = 0
                can_token = tokenizer.tokenize(realcan)
                for t in can_token:
                    if t.lower() not in m_set[0]:
                        flag = 1
                if flag == 0:
                    fin_can.append(realcan)
                else:
                    continue

            if len(fin_can) > 0:
                fin_can = list(set(fin_can))
                return fin_can[0] #可能有点问题
            else:
                return ""

    def get_list(self, heads, tails, token):
        #这个函数的目的是或得head和tail列表之后的实体
        #head 不能为0,这里的heads和tails是实际的还是模型中的？
        result_list = []
        for head in heads:
            if head == 0 or head - self.offset == len(token):
                continue
            tail = tails[tails >= head]
            if len(tail) > 0:
                tail = tail[0]
                if tail - self.offset == len(token):
                    continue
                if tail == head:
                    m = token[head - self.offset]
                else:
                    m = token[head - self.offset: tail - self.offset + 1]
                result_list.append((m, head, tail))
        return result_list

    def test(self, test_data_loader, model, output=False, h_bar=0.5, t_bar=0.5):
        if output:
            # check the result dir
            if not os.path.exists(self.config.result_dir):
                # print(os.path.abspath(self.config.result_dir))
                os.makedirs(self.config.result_dir)

            path = os.path.join(self.config.result_dir, self.config.result_save_name)

            fw = open(path, 'w')

        orders = ['M', 'F', 'VoF', 'C', 'VoC']

        def to_tup(set_list):
            ret = []

            for single_dict in set_list:
                # print(single_dict)
                single_set = tuple(single_dict.values())
                ret.append(single_set)
            return set(ret)

        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        correct_num, predict_num, gold_num = 0, 0, 0

        while data is not None:
            with torch.no_grad():
                token_ids = data['input_ids']
                tokens = data['token_text'][0]
                mask = data['attention_mask']
                text = data['text'][0]

                encoded_text = model.get_encoded_text(token_ids, mask) #拿到表示
                # [batch_size, seq_len, 1]
                pred_m_heads, pred_m_tails = model.get_m(encoded_text)
                m_heads, m_tails = np.where(pred_m_heads.cpu()[0] > h_bar)[0], np.where(pred_m_tails.cpu()[0] > t_bar)[0] #where 返回索引
                ms = self.get_list(m_heads, m_tails, tokens)

                if ms:
                    tuple_list = []
                    # f
                    # m_head_mapping = torch.Tensor(len(ms), 1, encoded_text.size(1)).zero_()  # [m_num, 1, seq_len]
                    # m_tail_mapping = torch.Tensor(len(ms), 1, encoded_text.size(1)).zero_()
                    # for m_idx, m in enumerate(ms):
                    #     m_head_mapping[m_idx][0][m[1]+1] = 1  # 从真实的index转化为模型中的
                    #     m_tail_mapping[m_idx][0][m[2]+1] = 1  # [m_num, 1, seq_len]
                    # m_tail_mapping = m_tail_mapping.to(repeated_encoded_text)
                    # m_head_mapping = m_head_mapping.to(repeated_encoded_text)
                    # pred_f_heads, pred_f_tails = model.get_f(m_head_mapping, m_tail_mapping, repeated_encoded_text) #这个时候不知道对应的谁是谁了
                    for m_idx, m in enumerate(ms):  # 对其中的一个m
                        repeated_encoded_text = encoded_text.repeat(len(ms), 1, 1)
                        m_head_mapping = torch.Tensor(len(ms), 1, encoded_text.size(1)).zero_()  # [m_num, 1, seq_len]
                        m_tail_mapping = torch.Tensor(len(ms), 1, encoded_text.size(1)).zero_()
                        m_head_mapping[m_idx][0][m[1] + 1] = 1  # 从真实的index转化为模型中的
                        m_tail_mapping[m_idx][0][m[2] + 1] = 1  # [m_num, 1, seq_len]
                        m_tail_mapping = m_tail_mapping.to(repeated_encoded_text)
                        m_head_mapping = m_head_mapping.to(repeated_encoded_text)
                        pred_f_heads, pred_f_tails = model.get_f(m_head_mapping, m_tail_mapping, repeated_encoded_text) #如果放在这里，得到的就是该

                        m_out = self.get_out(m, tokens, text)
                        f_heads, f_tails = np.where(pred_f_heads.cpu()[m_idx] > 0.35)[0], np.where(
                            pred_f_tails.cpu()[m_idx] > 0.35)[0]  # 这是f的新坐标，当然这可能有很多个
                        fs = self.get_list(f_heads, f_tails, tokens)

                        if fs:

                            # f_tail_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                            # f_head_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                            # m_tail_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                            # m_head_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                            # for f_idx, f in enumerate(fs):
                            #     f_head_mapping_vof[f_idx][0][f[1]+1] = 1
                            #     f_tail_mapping_vof[f_idx][0][f[2]+1] = 1
                            #     m_head_mapping_vof[f_idx][0][m[1]+1] = 1
                            #     m_tail_mapping_vof[f_idx][0][m[2]+1] = 1
                            # f_tail_mapping_vof = f_tail_mapping_vof.to(repeated_encoded_text_vof)
                            # f_head_mapping_vof = f_head_mapping_vof.to(repeated_encoded_text_vof)
                            # m_head_mapping_vof = m_head_mapping_vof.to(repeated_encoded_text_vof)
                            # m_tail_mapping_vof = m_tail_mapping_vof.to(repeated_encoded_text_vof)
                            # pred_vof_heads, pred_vof_tails = model.get_vof(m_head_mapping_vof, m_tail_mapping_vof, f_head_mapping_vof, f_tail_mapping_vof, repeated_encoded_text_vof)

                            for f_idx, f in enumerate(fs):
                                repeated_encoded_text_vof = encoded_text.repeat(len(fs), 1, 1)
                                f_tail_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                                f_head_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                                m_tail_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                                m_head_mapping_vof = torch.Tensor(len(fs), 1, encoded_text.size(1)).zero_()
                                f_head_mapping_vof[f_idx][0][f[1] + 1] = 1
                                f_tail_mapping_vof[f_idx][0][f[2] + 1] = 1
                                m_head_mapping_vof[f_idx][0][m[1] + 1] = 1
                                m_tail_mapping_vof[f_idx][0][m[2] + 1] = 1
                                f_tail_mapping_vof = f_tail_mapping_vof.to(repeated_encoded_text_vof)
                                f_head_mapping_vof = f_head_mapping_vof.to(repeated_encoded_text_vof)
                                m_head_mapping_vof = m_head_mapping_vof.to(repeated_encoded_text_vof)
                                m_tail_mapping_vof = m_tail_mapping_vof.to(repeated_encoded_text_vof)
                                pred_vof_heads, pred_vof_tails = model.get_vof(m_head_mapping_vof, m_tail_mapping_vof,
                                                                               f_head_mapping_vof, f_tail_mapping_vof,
                                                                               repeated_encoded_text_vof)
                                # print("!!!pred_vof_head:", pred_vof_heads)
                                # print("!!!pred_vof_tail:", pred_vof_tails)
                                f_out = self.get_out(f, tokens, text)
                                vof_heads, vof_tails = \
                                np.where(pred_vof_heads.cpu()[f_idx].numpy() == np.max(pred_vof_heads.cpu()[f_idx].numpy()))[0], \
                                np.where(pred_vof_tails.cpu()[f_idx].numpy() == np.max(pred_vof_tails.cpu()[f_idx].numpy()))[0]
                                # vof_heads, vof_tails = np.where(pred_vof_heads.cpu()[f_idx] > 0.25)[0], np.where(pred_vof_tails.cpu()[f_idx] > 0.25)[0]
                                vofs = self.get_list(vof_heads, vof_tails, tokens)

                                if vofs:
                                    repeated_encoded_text_voc = encoded_text.repeat(len(vofs), 1, 1)
                                    # vof_head_mapping = torch.Tensor(len(vofs), 1, encoded_text.size(1)).zero_()
                                    # vof_tail_mapping = torch.Tensor(len(vofs), 1, encoded_text.size(1)).zero_()
                                    # for vof_idx, vof in enumerate(vofs):
                                    #     vof_head_mapping[vof_idx][0][vof[1]] = 1
                                    #     vof_tail_mapping[vof_idx][0][vof[2]] = 1
                                    # vof_head_mapping = vof_head_mapping.to(repeated_encoded_text_vof)
                                    # vof_tail_mapping = vof_tail_mapping.to(repeated_encoded_text_vof)
                                    # pred_voc_heads, pred_voc_tails = model.get_voc(vof_head_mapping, vof_tail_mapping, repeated_encoded_text_voc)
                                    for vof_idx, vof in enumerate(vofs):
                                        vof_head_mapping = torch.Tensor(len(vofs), 1, encoded_text.size(1)).zero_()
                                        vof_tail_mapping = torch.Tensor(len(vofs), 1, encoded_text.size(1)).zero_()
                                        vof_head_mapping[vof_idx][0][vof[1]] = 1
                                        vof_tail_mapping[vof_idx][0][vof[2]] = 1
                                        vof_head_mapping = vof_head_mapping.to(repeated_encoded_text_vof)
                                        vof_tail_mapping = vof_tail_mapping.to(repeated_encoded_text_vof)
                                        pred_voc_heads, pred_voc_tails = model.get_voc(vof_head_mapping,
                                                                                       vof_tail_mapping,
                                                                                       repeated_encoded_text_voc)

                                        vof_out = self.get_out(vof, tokens, text)
                                        voc_heads, voc_tails = np.where(pred_voc_heads.cpu()[vof_idx] > 0.35)[0], np.where(pred_voc_tails.cpu()[vof_idx] > 0.35)[0]
                                        vocs = self.get_list(voc_heads, voc_tails, tokens)

                                        if vocs:
                                            repeated_encoded_text_c = encoded_text.repeat(len(vocs), 1, 1)
                                            # voc_head_mapping = torch.Tensor(len(vocs), 1, encoded_text.size(1)).zero_()
                                            # voc_tail_mapping = torch.Tensor(len(vocs), 1, encoded_text.size(1)).zero_()
                                            # for voc_idx, voc in enumerate(vocs):
                                            #     voc_head_mapping[voc_idx][0][voc[1]] = 1
                                            #     voc_tail_mapping[voc_idx][0][voc[2]] = 1
                                            # voc_head_mapping = voc_head_mapping.to(repeated_encoded_text_c)
                                            # voc_tail_mapping = voc_tail_mapping.to(repeated_encoded_text_c)
                                            # pred_c_heads, pred_c_tails = model.get_c(voc_head_mapping, voc_tail_mapping, repeated_encoded_text_c)
                                            for voc_idx, voc in enumerate(vocs):
                                                voc_head_mapping = torch.Tensor(len(vocs), 1,
                                                                                encoded_text.size(1)).zero_()
                                                voc_tail_mapping = torch.Tensor(len(vocs), 1,
                                                                                encoded_text.size(1)).zero_()
                                                voc_head_mapping[voc_idx][0][voc[1]] = 1
                                                voc_tail_mapping[voc_idx][0][voc[2]] = 1
                                                voc_head_mapping = voc_head_mapping.to(repeated_encoded_text_c)
                                                voc_tail_mapping = voc_tail_mapping.to(repeated_encoded_text_c)
                                                pred_c_heads, pred_c_tails = model.get_c(voc_head_mapping,
                                                                                         voc_tail_mapping,
                                                                                         repeated_encoded_text_c)

                                                voc_out = self.get_out(voc, tokens, text)
                                                c_heads, c_tails = np.where(pred_c_heads.cpu()[voc_idx] > 0.35)[0], np.where(pred_c_tails.cpu()[voc_idx] > 0.35)[0]
                                                cs = self.get_list(c_heads, c_tails, tokens)

                                                if cs:
                                                    for c in cs:
                                                        c_out = self.get_out(c, tokens, text)
                                                        tuple_list.append((m_out, f_out, vof_out, c_out, voc_out))
                                                else:
                                                    tuple_list.append((m_out, f_out, vof_out, "", voc_out))
                                        else:
                                            tuple_list.append((m_out, f_out, vof_out, "", ""))
                                else:
                                    tuple_list.append((m_out, f_out, "", "", ""))
                        else:
                            tuple_list.append((m_out, "", "", "", ""))
                    tuple_set = set()

                    for (m_i, f_i, vof_i, c_i, voc_i) in tuple_list:
                        tuple_set.add((m_i, f_i, vof_i, c_i, voc_i))
                    pred_list = list(tuple_set)
                else:
                    pred_list = []

                pred_triples = set(pred_list)
                print("!!!!!pred_triples", pred_triples)

                gold_triples = to_tup(data['set'][0])
                print("!!!!!gold_triples", gold_triples)

                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                if output:
                    result = json.dumps({
                        # 'text': ' '.join(tokens),
                        'triple_list_gold': [
                            dict(zip(orders, triple)) for triple in gold_triples
                        ],
                        'triple_list_pred': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ],
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                        ],
                        'lack': [
                            dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                        ]
                    }, ensure_ascii=False)
                    fw.write(result + '\n')

                data = test_data_prefetcher.next()

        print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
        #这里写一个输出的函数
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, f1_score

    def testall(self, model_pattern):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        precision, recall, f1_score = self.test(test_data_loader, model, True)
        print("f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".format(f1_score, precision, recall))