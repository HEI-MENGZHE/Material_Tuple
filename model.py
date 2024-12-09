from torch import nn
import numpy as np
from transformers import BertTokenizer,BertModel
import torch
import torch.nn.functional as F
#11/27

class MaterialSet(nn.Module):
    def __init__(self, config):
        super(MaterialSet, self).__init__()
        self.config = config
        self.bert_dim = 768
        self.bert_encoder = BertModel.from_pretrained("../MatSciBERT")
        self.m_heads_linear = nn.Linear(self.bert_dim, 1) #M
        self.c_heads_linear = nn.Linear(self.bert_dim, 1) #C
        self.m_tails_linear = nn.Linear(self.bert_dim, 1)
        self.c_tails_linear = nn.Linear(self.bert_dim, 1)

        self.f_heads_linear = nn.Linear(self.bert_dim, 1) #F
        self.f_tails_linear = nn.Linear(self.bert_dim, 1)

        self.vof_heads_linear = nn.Linear(self.bert_dim, 1) #vof
        self.vof_tails_linear = nn.Linear(self.bert_dim, 1)
        self.voc_heads_linear = nn.Linear(self.bert_dim, 1) #voc
        self.voc_tails_linear = nn.Linear(self.bert_dim, 1)


    def get_m(self, encoded_text):
        # [batch_size, seq_len, 1]
        pred_m_heads = self.m_heads_linear(encoded_text)
        pred_m_heads = torch.sigmoid(pred_m_heads)
        # [batch_size, seq_len, 1]
        pred_m_tails = self.m_tails_linear(encoded_text)
        pred_m_tails = torch.sigmoid(pred_m_tails)

        return pred_m_heads, pred_m_tails

    def get_c(self, encoded_text):
        # C的抽取不依赖于M
        # voc_head = torch.matmul(voc_head_mapping, encoded_text)
        # voc_tail = torch.matmul(voc_tail_mapping, encoded_text)
        # voc = (voc_head + voc_tail) / 2

        # encoded_text = encoded_text + voc
        encoded_text = encoded_text
        # [batch_size, seq_len, 1]
        pred_c_heads = self.c_heads_linear(encoded_text)
        pred_c_heads = torch.sigmoid(pred_c_heads)
        # [batch_size, seq_len, 1]
        pred_c_tails = self.c_tails_linear(encoded_text)
        pred_c_tails = torch.sigmoid(pred_c_tails)

        return pred_c_heads, pred_c_tails

    # def get_f(self, m_head_mapping, m_tail_mapping, encoded_text):
    def get_f(self, encoded_text):
        # F的抽取似乎也不依赖与M，但是我们还是这样做了，后面可以试一下
        # # [batch_size, 1, bert_dim]
        # m_head = torch.matmul(m_head_mapping, encoded_text) # m_head_mapping : [batch_size, 1, seq_len] encoded_text: [batch_size, seq_len, bert_dim]
        # # [batch_size, 1, bert_dim]
        # m_tail = torch.matmul(m_tail_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # m = (m_head + m_tail) / 2 #这种将首尾相加的方法可行吗
        # [batch_size, seq_len, bert_dim]
        # encoded_text = encoded_text + m
        # 放弃使用联合抽取的方法
        encoded_text = encoded_text
        # [batch_size, seq_len, 1]
        pred_f_heads = self.f_heads_linear(encoded_text)
        pred_f_heads = torch.sigmoid(pred_f_heads)
        # [batch_size, seq_len, 1]
        pred_f_tails = self.f_tails_linear(encoded_text)
        pred_f_tails = torch.sigmoid(pred_f_tails)
        return pred_f_heads, pred_f_tails

    # def get_vof(self, m_head_mapping, m_tail_mapping, f_head_mapping, f_tail_mapping, encoded_text):
    def get_vof(self, encoded_text):
        # [batch_size, 1, bert_dim]
        # m_head = torch.matmul(m_head_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # m_tail = torch.matmul(m_tail_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # m = (m_head + m_tail) / 2 # 不太懂为什么要除以2

        # [batch_size, 1, bert_dim]
        # f_head = torch.matmul(f_head_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # f_tail = torch.matmul(f_tail_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        # f = (f_head + f_tail) / 2
        # [batch_size, seq_len, bert_dim]
        # encoded_text = encoded_text + m + f
        # encoded_text = encoded_text + f
        encoded_text = encoded_text
        # [batch_size, seq_len, rel_num]
        pred_vof_heads = self.vof_heads_linear(encoded_text)
        pred_vof_heads = torch.sigmoid(pred_vof_heads)
        # [batch_size, seq_len, rel_num]
        pred_vof_tails = self.vof_tails_linear(encoded_text)
        pred_vof_tails = torch.sigmoid(pred_vof_tails)
        return pred_vof_heads, pred_vof_tails

    # def get_voc(self, vof_head_mapping, vof_tail_mapping, encoded_text):
    def get_voc(self, encoded_text):
        # [batch_size, 1, bert_dim]
        # c_head = torch.matmul(c_head_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # c_tail = torch.matmul(c_tail_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # c = (c_head + c_tail) / 2

        # [batch_size, 1, bert_dim]
        # vof_head = torch.matmul(vof_head_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # vof_tail = torch.matmul(vof_tail_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        # vof = (vof_head + vof_tail) / 2

        # [batch_size, seq_len, bert_dim]
        # encoded_text = encoded_text + vof
        encoded_text = encoded_text
        # [batch_size, seq_len, rel_num]
        pred_voc_heads = self.voc_heads_linear(encoded_text)
        pred_voc_heads = torch.sigmoid(pred_voc_heads)
        # [batch_size, seq_len, rel_num]
        pred_voc_tails = self.voc_tails_linear(encoded_text)
        pred_voc_tails = torch.sigmoid(pred_voc_tails)
        return pred_voc_heads, pred_voc_tails

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text


    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['input_ids']
        # [batch_size, seq_len]
        mask = data['attention_mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size, seq_len, 1]
        pred_m_heads, pred_m_tails = self.get_m(encoded_text)
        # [batch_size, seq_len, 1]
        # pred_c_heads, pred_c_tails = self.get_c(encoded_text)
        # [batch_size, 1, seq_len]
        m_head_mapping = data['m_head'].unsqueeze(1)
        # [batch_size, 1, seq_len]
        m_tail_mapping = data['m_tail'].unsqueeze(1)
        # [batch_size, 1, seq_len]
        f_head_mapping = data['f_head'].unsqueeze(1)
        # 这些mapping都是 1*seq_len
        f_tail_mapping = data['f_tail'].unsqueeze(1)
        c_head_mapping = data['c_head'].unsqueeze(1)
        vof_head_mapping = data['vof_head'].unsqueeze(1)
        c_tail_mapping = data['c_tail'].unsqueeze(1)
        vof_tail_mapping = data['vof_tail'].unsqueeze(1)
        voc_head_mapping = data['voc_head'].unsqueeze(1)
        voc_tail_mapping = data['voc_tail'].unsqueeze(1)
        # [batch_size, seq_len, 1]
        # pred_f_heads, pred_f_tails = self.get_f(m_head_mapping, m_tail_mapping, encoded_text)
        # pred_vof_heads, pred_vof_tails = self.get_vof(m_head_mapping, m_tail_mapping, f_head_mapping, f_tail_mapping, encoded_text)
        # pred_voc_heads, pred_voc_tails = self.get_voc(vof_head_mapping, vof_tail_mapping, encoded_text)
        # pred_c_heads, pred_c_tails = self.get_c(voc_head_mapping, voc_tail_mapping, encoded_text)

        pred_f_heads, pred_f_tails = self.get_f(encoded_text)
        pred_vof_heads, pred_vof_tails = self.get_vof(encoded_text)
        pred_voc_heads, pred_voc_tails = self.get_voc(encoded_text)
        pred_c_heads, pred_c_tails = self.get_c(encoded_text)
        return pred_m_heads, pred_m_tails, pred_f_heads, pred_f_tails, pred_vof_heads, pred_vof_tails, pred_c_heads, pred_c_tails, pred_voc_heads, pred_voc_tails


class MaterialSet_allocation(nn.Module):
    def __init__(self, config):
        super(MaterialSet_allocation, self).__init__()
        self.config = config
        if self.config.traintype == "all":
            self.layer_num = 6
        elif self.config.traintype == "wo_att":
            self.layer_num = 2
        else:
            self.layer_num = 4
        self.bert_dim = 768
        self.hid_dim = 768
        self.dropout = 0.5

        self.sqrt_d = np.sqrt(self.bert_dim)
        self.bert_encoder = BertModel.from_pretrained("../MatSciBERT")
        self.linear_v_vof = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.bert_dim * self.layer_num, self.hid_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, 1),
            nn.Sigmoid()
        )
        self.linear_m_vof = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.bert_dim * self.layer_num, self.hid_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, 1),
            nn.Sigmoid()
        )
        self.linear_voc_vof = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.bert_dim * self.layer_num, self.hid_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, 1),
            nn.Sigmoid()
        )


    def get_combined_embedding(self, e1_embedding, e2_embedding):
        # e1_embedding: batch * entity_num * 768
        # e12_score: batch * e1_num * e2_num
        e12_score = torch.sum(e1_embedding.unsqueeze(2) * e2_embedding.unsqueeze(1), dim=-1) * (
                    1 / self.sqrt_d)
        # 这个-5是什么意思？
        # batch * e1_num * e2_num * 1 到达行归一化和列归一，
        e12e2_softmax = F.softmax(e12_score, dim=1).unsqueeze(-1) # 要列归一化
        e22e1_softmax = F.softmax(e12_score, dim=2).unsqueeze(-1).permute(0,2,1,3) # 要行归一化
        # b * 1 * e2 * 768   b * e2 * e1 * 1
        # torch.Size([8, 4, 1, 768])
        # torch.Size([8, 4, 3, 1])
        # torch.Size([8, 3, 768])
        e1_e2Awared = torch.sum(e2_embedding.unsqueeze(2) * e22e1_softmax, dim=1)
        # b * 1 * e1 * 768   b * e1 * e2 * 1
        # torch.Size([8, 1, 3, 768])
        # torch.Size([8, 3, 4, 1])
        # torch.Size([8, 4, 768])
        e2_e1Awared = torch.sum(e1_embedding.unsqueeze(2) * e12e2_softmax, dim=1)
        A_e12e2 = e1_e2Awared.unsqueeze(2).repeat(1, 1, e2_e1Awared.shape[1], 1)
        A_e22e1 = e2_e1Awared.unsqueeze(1).repeat(1, e1_e2Awared.shape[1], 1, 1)
        # bidirectional attention
        A_e1 = e1_embedding.unsqueeze(2).repeat(1,1,e2_embedding.shape[1],1)
        A_e2 = e2_embedding.unsqueeze(1).repeat(1,e1_embedding.shape[1],1,1)
        # argumentation embedding
        e12e1_softmax = F.softmax(e1_embedding.matmul(e1_embedding.transpose(-1,-2)), dim=-1)
        A_e12e1 = e12e1_softmax.matmul(e1_embedding).unsqueeze(2).repeat(1, 1, e2_embedding.shape[1], 1)
        # e2 to e2 attention
        e22e2_softmax = F.softmax(e2_embedding.matmul(e2_embedding.transpose(-1,-2)), dim=-1)
        A_e22e2 = e22e2_softmax.matmul(e2_embedding).unsqueeze(1).repeat(1, e1_embedding.shape[1], 1, 1)

        if self.config.traintype == "all":
            # proposed model
            latent = torch.cat((A_e1, A_e2, A_e12e2, A_e22e1, A_e12e1, A_e22e2), dim=-1)
        elif self.config.traintype == "wo_att":
            # no attention
            latent = torch.cat((A_e1, A_e2), dim=-1)
        elif self.config.traintype == "wo_self":
            # no self_attention
            latent = torch.cat((A_e1, A_e2, A_e12e2, A_e22e1), dim=-1)
        elif self.config.traintype == "wo_inter":
            # no inter_attention
            latent = torch.cat((A_e1, A_e2, A_e12e1, A_e22e2), dim=-1)
        else:
            latent = torch.zeros(self.bert_dim)

        return latent

    def get_entity_embedding(self, encoded_text, e1_mapping, e2_mapping):
        # e1_mapping: batch * entity_num * max_sent_len
        # encode_text: batch * max_sent_len * 768
        e1_embedding = encoded_text.transpose(1,2).matmul(e1_mapping.transpose(1,2)).transpose(1,2)
        e2_embedding = encoded_text.transpose(1,2).matmul(e2_mapping.transpose(1,2)).transpose(1,2)
        # e1_embedding: batch * entity_num * 768
        return e1_embedding, e2_embedding

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    def forward(self, data):
        encoded_text = self.get_encoded_text(data["input_ids"], data["attention_mask"])
        # mapping是嵌套数组，中间是实体的位置，
        # pos, neg是mapping对应次序实体的两两组合，
        # 现在的任务就是将给的实体位置和表示，得到组合表示

        # 先得到两种实体f和vof的表示情况。mapping是空的不影响模型的矩阵运算，在dataloader中必须把mapping也统一维数。
        f_embedding, vof_embedding = self.get_entity_embedding(encoded_text, data['f_mapping'], data['vof_mapping'])
        m_embedding, voc_embedding = self.get_entity_embedding(encoded_text, data['m_mapping'], data['voc_mapping'])

        # 分为三个二分类问题，首先是f和vof
        latent_f_vof = self.get_combined_embedding(f_embedding, vof_embedding)
        latent_m_vof = self.get_combined_embedding(m_embedding, vof_embedding)
        latent_voc_vof = self.get_combined_embedding(voc_embedding, vof_embedding)

        score_f_vof = self.linear_v_vof(latent_f_vof)
        score_m_vof = self.linear_m_vof(latent_m_vof)
        score_voc_vof = self.linear_voc_vof(latent_voc_vof)
        # 到这里forward函数的任务就结束了，然后我们需要考虑使用什么损失函数，正负例如何确定。将pos和neg转化为b * e1_max_num * e2_max_num * 1

        return score_f_vof, score_m_vof, score_voc_vof

