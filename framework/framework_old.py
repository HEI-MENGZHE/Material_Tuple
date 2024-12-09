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
                precision, recall, f1_score, _ = self.test(dev_data_loader, model)
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


    def get_out(self, m_set, token, text):
        head, tail = m_set[1], m_set[2] #这里拿的是头尾文本的坐标，
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
            head_list = []
            start = 0
            head_g = head_text
            tail_g = tail_text
            while text.lower().find(head_g, start) != -1:
                head_in_text = text.lower().find(head_g, start)
                start = head_in_text + 1
                head_list.append(head_in_text)

            tail_list = []
            start1 = 0
            while text.lower().find(tail_g, start1) != -1:
                tail_in_text = text.lower().find(tail_g, start1)
                start1 = tail_in_text + 1
                tail_list.append(tail_in_text)

            result_list = []
            for head_ in head_list:
                for tail_ in [i for i in tail_list if i > head_]:
                    result_list.append((head_, tail_))
            if len(result_list) == 0:
                return ""
            else:
                result_list.sort(key=lambda x: (x[1] - x[0]))
                true_list = []
                word_list = token[head-self.offset:tail]
                for can in result_list:
                    temp = 0
                    can_res = text[can[0]:can[1] + len(tail_g)]
                    # print(can_res)
                    for word_piece in word_list:
                        if word_piece.lstrip("##") not in can_res.lower():
                            temp = 1
                            break
                    if temp == 1:
                        continue
                    else:
                        true_list.append(can)
                if len(true_list) == 0:
                    return text[result_list[0][0]:result_list[0][1] + len(tail_g)]
                    # print("result_list", result_list)
                    # print("text", text)
                    # raise Exception("can_list has entity but true_list does not")
                else:
                    true_result = text[true_list[0][0]:true_list[0][1] + len(tail_g)]
                    return true_result
    def get_mapping(self, heads, tails, token):
        result_list = []
        token_len = len(token)
        mapping = [0 for i in range(token_len + 2)]
        for head in heads:
            if head == 0 or head - self.offset == len(token):
                continue
            tail = tails[tails >= head]
            if len(tail) > 0:
                tail = tail[0]
                if tail-1 == len(token):
                    continue
                if tail == head:
                    mapping[head] = 1
                else:
                    for i in range(head, tail+1):
                        mapping[i] = 1
            result_list.append(mapping)
        return result_list

    def get_F1(self, list_pred1, list_pred2, list_pred3, list_pred4, list_pred5, list_gold1, list_gold2, list_gold3, list_gold4, list_gold5):

        list_pred1 = set(list_pred1)
        list_pred2 = set(list_pred2)
        list_pred3 = set(list_pred3)
        list_pred4 = set(list_pred4)
        list_pred5 = set(list_pred5)
        correct_m = len(list_pred1 & list_gold1)
        correct_f = len(list_pred2 & list_gold2)
        correct_vof = len(list_pred3 & list_gold3)
        correct_c = len(list_pred4 & list_gold4)
        correct_voc = len(list_pred5 & list_gold5)

        predict_m = len(list_pred1)
        predict_f = len(list_pred2)
        predict_vof = len(list_pred3)
        predict_c = len(list_pred4)
        predict_voc = len(list_pred5)

        gold_m = len(list_gold1)
        gold_f = len(list_gold2)
        gold_vof = len(list_gold3)
        gold_c = len(list_gold4)
        gold_voc = len(list_gold5)

        return correct_m, correct_f, correct_vof, correct_c, correct_voc

    def get_list(self, heads, tails, token):
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
        res_list = []

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
        data_dict = {}
        clock = 1
        data_dict['set'] = data['set'][0]
        data_dict['text'] = data['text'][0]
        data_dict['data_id'] = clock
        correct_num, predict_num, gold_num = 0, 0, 0
        correct_t, predict_t, gold_t = 0, 0, 0
        correct_1, correct_2, correct_3, correct_4, correct_5 = 0,0,0,0,0
        predict_1, predict_2, predict_3, predict_4, predict_5 = 0,0,0,0,0
        gold_1, gold_2, gold_3, gold_4, gold_5 = 0,0,0,0,0

        while data is not None:
            with torch.no_grad():
                token_ids = data['input_ids']
                tokens = data['token_text'][0]
                mask = data['attention_mask']
                data_id = data['data_id']
                text = data['text'][0]
                m_pred_list = []
                f_pred_list = []
                vof_pred_list = []
                c_pred_list = []
                voc_pred_list = []
                m_h, m_t = 0.2, 0.2
                f_h, f_t = 0.3, 0.2
                vof_h, vof_t = 0.38, 0.25
                voc_h, voc_t = 0.15, 0.15
                encoded_text = model.get_encoded_text(token_ids, mask) #拿到表示
                # [batch_size, seq_len, 1]
                pred_m_heads, pred_m_tails = model.get_m(encoded_text)
                m_heads, m_tails = np.where(pred_m_heads.cpu()[0] > m_h)[0], np.where(pred_m_tails.cpu()[0] > m_t)[0] #where 返回索引
                ms = self.get_list(m_heads, m_tails, tokens)
                m_mapping = self.get_mapping(m_heads, m_tails, tokens)
                if ms:
                    tuple_list = []

                    for m_idx, m in enumerate(ms):  # 对其中的一个m
                        repeated_encoded_text = encoded_text.repeat(len(ms), 1, 1)
                        m_head_mapping = torch.Tensor(len(ms), 1, encoded_text.size(1)).zero_()  # [m_num, 1, seq_len]
                        m_tail_mapping = torch.Tensor(len(ms), 1, encoded_text.size(1)).zero_()
                        m_head_mapping[m_idx][0][m[1] + 1] = 1  # 从真实的index转化为模型中的
                        m_tail_mapping[m_idx][0][m[2] + 1] = 1  # [m_num, 1, seq_len]
                        m_tail_mapping = m_tail_mapping.to(repeated_encoded_text)
                        m_head_mapping = m_head_mapping.to(repeated_encoded_text)
                        pred_f_heads, pred_f_tails = model.get_f(m_head_mapping, m_tail_mapping, repeated_encoded_text)
                        # print("!!!!!!!",pred_f_heads)
                        m_out = self.get_out(m, tokens, text)
                        m_pred_list.append(m_out)
                        f_heads, f_tails = np.where(pred_f_heads.cpu()[m_idx] > f_h)[0], np.where(pred_f_tails.cpu()[m_idx] > f_t)[0]  # 这是f的新坐标，当然这可能有很多个
                        fs = self.get_list(f_heads, f_tails, tokens)
                        f_mapping = self.get_mapping(f_heads, f_tails, tokens)
                        if fs:

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
                                f_out = self.get_out(f, tokens, text)
                                f_pred_list.append(f_out)
                                # vof_heads, vof_tails = \
                                # np.where(pred_vof_heads.cpu()[f_idx].numpy() == np.max(pred_vof_heads.cpu()[f_idx].numpy()))[0], \
                                # np.where(pred_vof_tails.cpu()[f_idx].numpy() == np.max(pred_vof_tails.cpu()[f_idx].numpy()))[0]
                                vof_heads, vof_tails = np.where(pred_vof_heads.cpu()[f_idx] > vof_h)[0], np.where(pred_vof_tails.cpu()[f_idx] > vof_t)[0]
                                if len(vof_heads) > 10:
                                    vof_heads = np.where(pred_vof_heads.cpu()[f_idx] > 0.99)[0]

                                vofs = self.get_list(vof_heads, vof_tails, tokens)

                                vof_mapping = self.get_mapping(vof_heads, vof_tails, tokens)

                                if vofs:
                                    repeated_encoded_text_voc = encoded_text.repeat(len(vofs), 1, 1)

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
                                        vof_pred_list.append(vof_out)
                                        voc_heads, voc_tails = np.where(pred_voc_heads.cpu()[vof_idx] > voc_h)[0], np.where(pred_voc_tails.cpu()[vof_idx] > voc_t)[0]

                                        vocs = self.get_list(voc_heads, voc_tails, tokens)

                                        if vocs:
                                            repeated_encoded_text_c = encoded_text.repeat(len(vocs), 1, 1)
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
                                                voc_pred_list.append(voc_out)
                                                if "room temperature" in text and "room temperature" not in voc_pred_list:
                                                    voc_pred_list.append("room temperature")
                                                c_heads, c_tails = np.where(pred_c_heads.cpu()[voc_idx] > 0.35)[0], np.where(pred_c_tails.cpu()[voc_idx] > 0.35)[0]
                                                cs = self.get_list(c_heads, c_tails, tokens)

                                                if cs:
                                                    for c in cs:
                                                        c_out = self.get_out(c, tokens, text)
                                                        c_pred_list.append(c_out)
                                                        tuple_list.append((m_out, f_out, vof_out, c_out, voc_out))

                                                else:
                                                    tuple_list.append((m_out, f_out, vof_out, "", voc_out))

                                        else:
                                            # if "room temperature" in text:
                                            #     tuple_list.append((m_out, f_out, vof_out, "", "room temperature"))
                                            # else:
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

                # 在train.allocation 训练之前我们需要更多的信息来调整数据抽取的效果。
                true_m_list = []
                true_f_list = []
                true_vof_list = []
                true_c_list = []
                true_voc_list = []
                for i in data['set']:
                    true_m_list.append(i['M'])
                    true_f_list.append(i['F'])
                    true_vof_list.append(i['VoF'])
                    true_c_list.append(i['C'])
                    if len(i['VoC']) == 0:
                        continue
                    else:
                        true_voc_list.append(i['VoC'])

                # if "" in true_voc_list:
                #     true_voc_list.remove("")

                true_vof_list = set(true_vof_list)
                true_f_list = set(true_f_list)
                true_m_list = set(true_m_list)
                true_c_list = set(true_c_list)
                true_voc_list = set(true_voc_list)
                m_pred_list = set(m_pred_list)
                f_pred_list = set(f_pred_list)
                vof_pred_list = set(vof_pred_list)
                c_pred_list = set(c_pred_list)
                voc_pred_list = set(voc_pred_list)

                # print("true_voc_list_len:",len(true_voc_list))
                print("true_voc_list:",true_voc_list)
                print("pred_voc",voc_pred_list)
                correct_m, correct_f, correct_vof, correct_c, correct_voc = self.get_F1(m_pred_list, f_pred_list,
                                                                                        vof_pred_list, c_pred_list,
                                                                                        voc_pred_list, true_m_list,
                                                                                        true_f_list, true_vof_list,
                                                                                        true_c_list, true_voc_list)

                pred_triples = set(pred_list)
                # print("data_id:", data_id)
                # print("!pred_triples", pred_triples)

                gold_triples = to_tup(data['set'])
                # data['set'] 在处理num>2的情况，需要加[0]
                # print("!gold_triples", gold_triples)
                # 有没有办法换一种方法搞测试计算？
                # if len(pred_triples)>1:
                # if len(pred_triples & gold_triples) != len(gold_triples):
                #     print("data_id", data['data_id'][0])
                #     print("pred", pred_triples)
                #     print("gold", gold_triples)
                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                correct_1 += correct_m
                correct_2 += correct_f
                correct_3 += correct_vof
                correct_4 += correct_c
                correct_5 += correct_voc
                # correct_5 = 0
                correct_t = correct_1 + correct_2 + correct_3 +  correct_5 + correct_t
                predict_1 += len(m_pred_list)
                predict_2 += len(f_pred_list)
                predict_3 += len(vof_pred_list)
                predict_4 += len(c_pred_list)
                # predict_5 = 0
                predict_5 += len(voc_pred_list)
                predict_t = predict_1 + predict_2 + predict_3 +  predict_5 + predict_t
                gold_1 += len(true_m_list)
                gold_2 += len(true_f_list)
                gold_3 += len(true_vof_list)
                gold_4 += len(true_c_list)
                # gold_5 = 0
                gold_5 += len(true_voc_list)
                gold_t = gold_t + gold_1 + gold_2 + gold_3 +  gold_5
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
                result_dict = []
                key_list = ["M", "F", "VoF", "C", "VoC"]
                for i in pred_list:
                    result_dict.append(dict(zip(key_list,i)))
                data_dict['pred_set'] = result_dict
                res_list.append(data_dict)
                data = test_data_prefetcher.next()
                clock += 1
                data_dict = {}

                if data is not None:
                    data_dict['set'] = data['set'][0]
                    data_dict['text'] = data['text'][0]
                    data_dict['data_id'] = clock

        print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        precision_1 = correct_1 / (predict_1 + 1e-10)
        recall_1 = correct_1 / (gold_1 + 1e-10)
        f1_score_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-10)

        precision_2 = correct_2 / (predict_2 + 1e-10)
        recall_2 = correct_2 / (gold_2 + 1e-10)
        f1_score_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-10)

        precision_3 = correct_3 / (predict_3 + 1e-10)
        recall_3 = correct_3 / (gold_3 + 1e-10)
        f1_score_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3 + 1e-10)

        precision_4 = correct_4 / (predict_4 + 1e-10)
        recall_4 = correct_4 / (gold_4 + 1e-10)
        f1_score_4 = 2 * precision_4 * recall_4 / (precision_4 + recall_4 + 1e-10)

        precision_5 = correct_5 / (predict_5 + 1e-10)
        recall_5 = correct_5 / (gold_5 + 1e-10)
        f1_score_5 = 2 * precision_5 * recall_5 / (precision_5 + recall_5 + 1e-10)

        precision_t = correct_t / (predict_t + 1e-10)
        recall_t = correct_t / (gold_t + 1e-10)
        f1_score_t = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-10)
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        # print("indicator in full set, F1: {:4.2f}, P: {:4.2f}, R: {:4.2f}".format(f1_score,precision,recall))
        print("indicator in m, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_1,precision_1,recall_1))
        print("indicator in f, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_2,precision_2,recall_2))
        print("indicator in vof, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_3,precision_3,recall_3))
        print("indicator in c, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_4,precision_4,recall_4))
        print("indicator in voc, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_5,precision_5,recall_5))
        print("indicator in total, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_t,precision_t,recall_t))
        # print("indicator in single set, F1: {:4.2f}, P: {:4.2f}, R: {:4.2f}".format(F1_ext,p_ext,r_ext))
        # print("indicator in single set, F1: {:4.2f}, P: {:4.2f}, R: {:4.2f}".format(F1_ext,p_ext,r_ext))

        return precision, recall, f1_score, res_list

    def testall(self, model_pattern):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
        print("model used:", path)
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        # 我应该要拿到一个json文件
        precision, recall, f1_score, res_list = self.test(test_data_loader, model, True)
        write_path = os.path.join("./data/MaterialSet_allocation/", self.config.test_prefix + '.json')
        # with open(write_path, "w") as f_normal:
        #     json.dump(res_list, f_normal)
        print("f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}".format(f1_score, precision, recall))


    def test_allocation(self, test_data_loader, model):
        test_data_prefetcher = data_loader_allocation.DataPreFetcher(test_data_loader)
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
                materialdict = data['pred_set'][0] # test函数生成的数据需要包含set与golden set，set用来得到unique list,这两个是不一样的
                order = ["M", "F", "VoF", "C", "VoC"]

                # 提取所有特别的实体, 这里有点问题，他们希望获得的输入是mapping而不是实体
                m_list = get_unique_entity(order[0], materialdict)
                f_list = get_unique_entity(order[1], materialdict)
                vof_list = get_unique_entity(order[2], materialdict)
                c_list = get_unique_entity(order[3], materialdict)
                voc_list = get_unique_entity(order[4], materialdict)

                # 在训练的时候，确实很有可能发生由于数量抽取不正确导致的非方阵情况，但是这种情况并不一定代表不正确，
                # 那么在非方阵的情况下，为什么会出现之前所谓的超出索引范围呢？
                # 首先应该了解，难道是由于get_entity生成的unique name list 与 索引产生了冲突？
                # batch * e1 * e2 * 1
                # print("voc_vof",score_voc_vof[0].squeeze(2).cpu().shape)
                # print("m_vof",score_m_vof[0].squeeze(2).cpu().shape)
                # print("f_vof",score_f_vof[0].squeeze(2).cpu().shape)
                # print("voc_list", len(voc_list))
                # print("m_list",len(m_list))
                # print("f_list",len(f_list))
                # np.argmax 如果 axis = 1 输出的结果维数与第一维相同， 也就是说，使用axis=1得到的是各个非vof的元素对应的最合适的vof
                # 但是如果我们从vof入手的话，应该是得到最适合vof的各个非vof元素
                # 所以我们应该生成np.argmax(x_vof,0), 然后得到这些非vof元素的坐标，然后得到文本实体


                if len(voc_list) != 0:
                    voc_vof = score_voc_vof[0].squeeze(2).cpu()
                    res_voc = np.argmax(voc_vof, axis=0).tolist()
                else:
                    res_voc = None

                if len(m_list) != 0:
                    m_vof = score_m_vof[0].squeeze(2).cpu()
                    res_m = np.argmax(m_vof, axis=0).tolist()
                else:
                    res_m = None

                if len(f_list) != 0:
                    f_vof = score_f_vof[0].squeeze(2).cpu()
                    res_f = np.argmax(f_vof, axis=0).tolist()
                else:
                    res_f = None

                pred = []

                if len(vof_list) != 0:
                    for vof_index in range(len(vof_list)):
                        vof = vof_list[vof_index]
                        if len(m_list) != 0:
                            m_index = res_m[vof_index]
                            m = m_list[m_index]
                        else:
                            m = ""
                        if len(f_list) != 0:
                            f_index = res_f[vof_index]
                            f = f_list[f_index]
                        else:
                            f = ""
                        if len(voc_list) != 0:
                            voc_index = res_voc[vof_index]
                            voc = voc_list[voc_index]
                        else:
                            voc = ""
                        if len(c_list) != 0:
                            c = c_list[0]
                        else:
                            c = ""
                        pred.append((m.lower(), f.lower(), vof.lower(), c.lower(), voc.lower()))
                        # pred.append((m, f, vof, c, voc))

                pred = set(pred)

                label = data['set'][0]
                # num = 1 去掉[0]？
                gold = []
                for i in label:
                    res_list = [j.lower() for j in i.values()]
                    # res_list = i.values()
                    gold.append(tuple(res_list))
                gold = set(gold)

                id = data['data_id'][0]

                if len(pred & gold) != len(gold):
                    print("data_id", id)
                    print("pred", pred)
                    print("gold", gold)

                # if id in [1,4,5,7,18,19,28,31,34,35,40,21]:
                #     print("m_list", m_list)
                #     # print("vof_list", vof_list)
                #     # print("m_vof",m_vof)
                # if id in [9,33]:
                #     print("f_list", f_list)

                correct_num += len(pred & gold)
                predict_num += len(pred)
                gold_num += len(gold)
                data = test_data_prefetcher.next()

        print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num,
                                                                               gold_num))
        # 这里写一个输出的函数
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        return precision, recall, f1_score

    def testall_allocation(self, model_pattern):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
        print("model used:", path)
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.eval()
        test_data_loader = data_loader_allocation.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        precision, recall, f1_score = self.test_allocation(test_data_loader, model)

        print("f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}".format(f1_score, precision, recall))
