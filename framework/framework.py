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
torch.manual_seed(0)
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
        if self.config.fine_tuned_type != 0:
            fined_path = os.path.join(self.config.checkpoint_dir, self.config.fine_tune_model_name)
            print("fine tuned model used:", fined_path)
            ori_model.load_state_dict(torch.load(fined_path))
        else:
            print("we do not use fine tune")

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

                if recall >= best_recall:
                # if f1_score >= best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall
                    self.logging("!! BEST RECALL CHANGE, saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
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

        if self.config.fine_tuned_type != 0:
            fined_path = os.path.join(self.config.checkpoint_dir, self.config.fine_tune_model_name)
            print("fine tuned model used:", fined_path)
            ori_model.load_state_dict(torch.load(fined_path))
        else:
            print("we do not use fine tune")
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

                if f1_score >= best_f1_score: #由于是抽取阶段，我们用召回率
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

                encoded_text = model.get_encoded_text(token_ids, mask) #拿到表示
                # m
                pred_m_heads, pred_m_tails = model.get_m(encoded_text)

                mh, mt = 0.5,0.5
                fh, ft = 0.5, 0.5
                vofh, voft = 0.5, 0.5
                ch, ct = 0.5, 0.5
                voch, voct = 0.5, 0.5

                # print("fh,ft:",vofh, voft)
                m_heads, m_tails = np.where(pred_m_heads.cpu()[0] > mh)[0], np.where(pred_m_tails.cpu()[0] > mt)[0]

                ms = self.get_list(m_heads, m_tails, tokens)
                ms.sort(key=lambda x:x[1])

                # f
                pred_f_heads, pred_f_tails = model.get_f(encoded_text)
                f_heads, f_tails = np.where(pred_f_heads.cpu()[0] > fh)[0], \
                np.where(pred_f_tails.cpu()[0] > ft)[0]
                fs = self.get_list(f_heads, f_tails, tokens)
                fs.sort(key=lambda x:x[1])

                # vof !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pred_vof_heads, pred_vof_tails = model.get_vof(encoded_text)
                vof_heads, vof_tails = np.where(pred_vof_heads.cpu()[0] > vofh)[0], \
                np.where(pred_vof_tails.cpu()[0] > voft)[0]
                vofs = self.get_list(vof_heads, vof_tails, tokens)
                vofs.sort(key=lambda x:x[1])
                # c
                pred_c_heads, pred_c_tails = model.get_c(encoded_text)
                c_heads, c_tails = np.where(pred_c_heads.cpu()[0] > ch)[0], \
                np.where(pred_c_tails.cpu()[0] > ct)[0]
                cs = self.get_list(c_heads, c_tails, tokens)
                cs.sort(key=lambda x:x[1])
                # voc
                pred_voc_heads, pred_voc_tails = model.get_voc(encoded_text)
                voc_heads, voc_tails = np.where(pred_voc_heads.cpu()[0] > voch)[0], \
                np.where(pred_voc_tails.cpu()[0] > voct)[0]
                vocs = self.get_list(voc_heads, voc_tails, tokens)
                vocs.sort(key=lambda x:x[1])

                for m in ms:
                    m_out = self.get_out(m, tokens, text)
                    if m_out != "":
                        m_pred_list.append(m_out)
                for f in fs:
                    f_out = self.get_out(f, tokens, text)
                    if f_out != "":
                        f_pred_list.append(f_out)
                for vof in vofs:
                    vof_out = self.get_out(vof, tokens, text)
                    if vof_out != "":
                        vof_pred_list.append(vof_out)
                for c in cs:
                    c_out = self.get_out(c, tokens, text)
                    if c_out != "":
                        c_pred_list.append(c_out)
                for voc in vocs:
                    voc_out = self.get_out(voc, tokens, text)
                    if voc_out != "":
                        voc_pred_list.append(voc_out)
                # print("voc_pred_list：", voc_pred_list)

                #test voc and c
                if "temperature" in text and len(vocs) != 0:
                    c_pred_list.append("temperature")

                tuple_list = []
                if m_pred_list:
                    for m_out in m_pred_list:
                        if f_pred_list:
                            for f_out in f_pred_list:
                                if vof_pred_list:
                                    for vof_out in vof_pred_list:
                                        if voc_pred_list:
                                            for voc_out in voc_pred_list:
                                                if c_pred_list:
                                                    c_out = c_pred_list[0]
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

                true_m_list = []
                true_f_list = []
                true_vof_list = []
                true_c_list = []
                true_voc_list = []

                # print("看一看是什么样子的",data['set'])
                if len(data['set'][0]) > 4: # data['set'] -> [{}],
                    # print("看一看是什么样子的 1", data['set'])
                    for i in data['set']:
                        if i['M'] != "":
                            true_m_list.append(i['M'])
                        if i['F'] != "":
                            true_f_list.append(i['F'])
                        if i['VoF'] != "":
                            true_vof_list.append(i['VoF'])

                        if i['C'] != "":
                            true_c_list.append(i['C'])
                        if i['VoC'] != "":
                            # print("pred_voc_list", true_voc_list)
                            true_voc_list.append(i['VoC'])

                elif len(data['set'][0]) <= 4:
                    # print("看一看是什么样子的 2", data['set'])
                    for i in data['set'][0]:
                        if i['M'] != "":
                            true_m_list.append(i['M'])
                        if i['F'] != "":
                            true_f_list.append(i['F'])
                        if i['VoF'] != "":
                            true_vof_list.append(i['VoF'])
                        if i['C'] != "":
                            true_c_list.append(i['C'])
                        if i['VoC'] != "":
                            # print("pred_voc_list", true_voc_list)
                            true_voc_list.append(i['VoC'])


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
                # print("true_voc_list：", true_voc_list)
                # print("true_vof_list：", true_vof_list)

                temp = 0
                for i in true_c_list:
                    if i not in c_pred_list:
                        temp = 1
                for i in c_pred_list:
                    if i not in true_c_list:
                        temp = 1
                if temp:
                    print("true_vof_list:", true_c_list)
                    print("pred_vof_list", c_pred_list)

                correct_m, correct_f, correct_vof, correct_c, correct_voc = self.get_F1(m_pred_list, f_pred_list, vof_pred_list, c_pred_list, voc_pred_list, true_m_list, true_f_list, true_vof_list, true_c_list, true_voc_list)

                pred_triples = set(pred_list)
                # print("data_id:", data_id)
                # print("!pred_triples", pred_triples)
                if len(data['set'][0]) > 4:
                    gold_triples = to_tup(data['set'])
                elif len(data['set'][0]) <= 4:
                    gold_triples = to_tup(data['set'][0])
                # data['set'] 在处理num>2的情况，需要加[0]
                # print("!gold_triples", gold_triples)
                # 有没有办法换一种方法搞测试计算？


                # if len(pred_triples & gold_triples) != len(gold_triples):
                #     print("data_id", data['data_id'][0])
                #     print("input text:", text)
                #     # if len(m_pred_list & true_m_list) != len(m_pred_list) or (len(m_pred_list) == 0 and len(true_m_list)!=0):
                #     print("pred_m", m_pred_list)
                #     print("gold_m", true_m_list)
                #     # if len(f_pred_list & true_f_list) != len(f_pred_list) or (len(f_pred_list) == 0 and len(true_f_list) !=0):
                #     print("pred_f", f_pred_list)
                #     print("gold_f", true_f_list)
                #     # if len(vof_pred_list & true_vof_list) != len(vof_pred_list) or (len(vof_pred_list) == 0 and len(true_vof_list) != 0):
                #     print("pred_vof", vof_pred_list)
                #     print("gold_vof", true_vof_list)
                #
                # if len(c_pred_list & true_c_list) != len(c_pred_list) or (len(c_pred_list) == 0 and len(true_c_list) != {""}):
                #     print("pred_c", c_pred_list)
                #     print("gold_c", true_c_list)
                #
                #     # if len(voc_pred_list & true_voc_list) != len(voc_pred_list) or (len(voc_pred_list) == 0 and len(true_voc_list) != {""}):
                #     print("pred_voc", voc_pred_list)
                #     print("gold_voc", true_voc_list)
                #         # print("pred", pred_triples)
                #     print("gold", gold_triples)


                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                correct_1 += correct_m
                correct_2 += correct_f
                correct_3 += correct_vof
                correct_4 += correct_c
                correct_5 += correct_voc
                correct_t = correct_1 + correct_2 + correct_3 + correct_4 + correct_5 + correct_t
                predict_1 += len(m_pred_list)
                predict_2 += len(f_pred_list)
                predict_3 += len(vof_pred_list)
                predict_4 += len(c_pred_list)
                predict_5 += len(voc_pred_list)
                predict_t = predict_1 + predict_2 + predict_3 + predict_4 + predict_5 + predict_t
                gold_1 += len(true_m_list)
                gold_2 += len(true_f_list)
                gold_3 += len(true_vof_list)
                gold_4 += len(true_c_list)
                gold_5 += len(true_voc_list)
                gold_t = gold_t + gold_1 + gold_2 + gold_3 + gold_4 + gold_5
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
        # print("indicator in full set, F1: {:4.2f}, P: {:4.2f}, R: {:4.2f}".format(f1_score,precision,recall))
        print("indicator in m, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_1,precision_1,recall_1))
        print("indicator in f, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_2,precision_2,recall_2))
        print("indicator in vof, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_3,precision_3,recall_3))
        print("indicator in c, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_4,precision_4,recall_4))
        print("indicator in voc, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_5,precision_5,recall_5))
        print("indicator in total, F1: {:4.3f}, P: {:4.3f}, R: {:4.3f}".format(f1_score_t,precision_t,recall_t))
        # print("indicator in single set, F1: {:4.2f}, P: {:4.2f}, R: {:4.2f}".format(F1_ext,p_ext,r_ext))
        # return precision, recall, f1_score, res_list
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
        # 使用增强的数据集
        write_path = os.path.join("./data/MaterialSet_allocation/", self.config.test_prefix + '_enhance.json')
        with open(write_path, "w") as f_normal:
            json.dump(res_list, f_normal)
        print("f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}".format(f1_score, precision, recall))


    def test_allocation(self, test_data_loader, model):
        test_data_prefetcher = data_loader_allocation.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        correct_num, predict_num, gold_num = 0, 0, 0

        # def get_unique_entity(e_type, e_set):
        #     entity_list = []
        #     for single in e_set:
        #         entity_list.append(single[e_type])
        #     entity_list = list(set(entity_list))
        #     if "" in entity_list:
        #         entity_list.remove("")
        #
        #     if entity_list is None:
        #         return []
        #     else:
        #         return entity_list

        def get_unique_entity(e_type, e_set, text):
            entity_list = []
            if e_type == 'VoC':
                if "room temperature" in text:
                    entity_list.append(("room temperature", 0))
            # if e_type == 'C':
            #     if "room temperature" in text:

            for single in e_set:
                # index = text.find(single[e_type])
                index = text.lower().find(single[e_type].lower())
                if index == -1:
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

        def trans(a):
            n = 3
            b = torch.eye(a.shape[0])

            return a + torch.mul(a, b) * (n - 1)

        def gen_alternate(x):
            row1 = x[0, :].unsqueeze(0)  # 确保是2D张量
            row2 = x[1, :].unsqueeze(0)

            # 使用列表推导式交替复制两行
            rows = [row1, row2, row1, row2]

            # 使用torch.cat沿着第0维（行）连接这些行
            x_alternating = torch.cat(rows, dim=0)

            return x_alternating

        while data is not None:
            with torch.no_grad():
                (score_f_vof, score_m_vof, score_voc_vof) = model(data)
                materialdict = data['pred_set'][0] # test函数生成的数据需要包含set与golden set，set用来得到unique list,这两个是不一样的
                order = ["M", "F", "VoF", "C", "VoC"]
                text = data['text'][0]
                # 提取所有特别的实体, 这里有点问题，他们希望获得的输入是mapping而不是实体
                m_list = get_unique_entity(order[0], materialdict, text)
                f_list = get_unique_entity(order[1], materialdict, text)
                vof_list = get_unique_entity(order[2], materialdict, text)
                c_list = get_unique_entity(order[3], materialdict, text)
                voc_list = get_unique_entity(order[4], materialdict, text)

                only_rule = 0
                # 考虑四元组的规则抽取问题，隐含的条件往往是，可能重复的两个或者两个以上的实体种类只在文中出现了一次，导致匹配矩阵不再是方阵，
                # 所以我们需要将方阵补齐，然后进行规则加强
                sign_1, sign_2 = 0, 0
                if len(vof_list) == 4:
                    if len(m_list) == 2 and len(voc_list) == 2:
                        sign_1 = 1
                        m_list = m_list + m_list
                        voc_list = [voc_list[0] for i in voc_list] + [voc_list[1] for i in voc_list]
                        m_vof = score_m_vof[0].squeeze(2).cpu()
                        m_vof_new = torch.cat((m_vof, m_vof), dim=0)
                        voc_vof_new = gen_alternate(score_voc_vof[0].squeeze(2).cpu())

                    if len(m_list) == 2 and len(f_list) == 2:
                        sign_2 = 1
                        m_list = [m_list[0] for i in m_list] + [m_list[1] for i in m_list]
                        f_list = f_list + f_list
                        m_vof_new = gen_alternate(score_m_vof[0].squeeze(2).cpu())
                        f_vof = score_f_vof[0].squeeze(2).cpu()
                        f_vof_new = torch.cat((f_vof,f_vof), dim=0)
                        print("m_vof_new", m_vof_new)

                if len(voc_list) != 0:
                    if sign_1 == 1:
                        voc_vof = voc_vof_new
                    else:
                        voc_vof = score_voc_vof[0].squeeze(2).cpu()

                    if only_rule == 0:
                        if voc_vof.shape[0] == voc_vof.shape[1]:
                            res_voc = np.argmax(trans(voc_vof), axis=0).tolist()
                        else:
                            res_voc = np.argmax(voc_vof, axis=0).tolist()
                    else:
                        if voc_vof.shape[0] == voc_vof.shape[1]: # 如果是方阵，就是用单位矩阵作为输入，如果不能就是
                            I = torch.eye(voc_vof.shape[0])
                            res_voc = np.argmax(I, axis=0).tolist()
                        else:
                            random_voc_vof = torch.rand(voc_vof.shape[0], voc_vof.shape[1])
                            res_voc = np.argmax(random_voc_vof, axis=0).tolist()
                else:
                    res_voc = None

                if len(m_list) != 0:
                    if sign_1 + sign_2 >= 1:
                        m_vof = m_vof_new
                    else:
                        m_vof = score_m_vof[0].squeeze(2).cpu()

                    if only_rule == 0:
                        if m_vof.shape[0] == m_vof.shape[1]:
                            res_m = np.argmax(trans(m_vof), axis=0).tolist()
                        else:
                            res_m = np.argmax(m_vof, axis=0).tolist()
                    else:
                        if m_vof.shape[0] == m_vof.shape[1]:
                            res_m = np.argmax(torch.eye(m_vof.shape[0]), axis=0).tolist()
                        else:
                            random_m_vof = torch.rand(m_vof.shape[0], m_vof.shape[1])
                            res_m = np.argmax(random_m_vof, axis=0).tolist()
                else:
                    res_m = None

                if len(f_list) != 0:
                    if sign_2 == 1:
                        f_vof = f_vof_new
                    else:
                        f_vof = score_f_vof[0].squeeze(2).cpu()

                    if only_rule == 0:
                        if f_vof.shape[0] == f_vof.shape[1]:
                            res_f = np.argmax(trans(f_vof), axis=0).tolist()
                        else:
                            res_f = np.argmax(f_vof, axis=0).tolist()
                    else:
                        if f_vof.shape[0] == f_vof.shape[1]:
                            res_f = np.argmax(torch.eye(f_vof.shape[0]), axis=0).tolist()
                        else:
                            random_f_vof = torch.rand(f_vof.shape[0], f_vof.shape[1])
                            res_f = np.argmax(random_f_vof, axis=0).tolist()
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
                    print("text", data['text'][0])
                    print("pred", pred)
                    print("gold", gold)
                    # print("voc_vof",score_voc_vof[0].squeeze(2).cpu())
                    # print("m_vof",score_m_vof[0].squeeze(2).cpu())
                    # print("f_vof",score_f_vof[0].squeeze(2).cpu())
                    print("m_list:",m_list)
                    print("f_list:",f_list)
                    print("vof_list:",vof_list)
                    print("voc_list:",voc_list)


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
