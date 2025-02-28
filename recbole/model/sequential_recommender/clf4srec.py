# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import math
import random
from torch.nn import Parameter
import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.fft as fft
import torch.nn.functional as F
from kan import KANLinear, FastKAN


class CLF4SRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(CLF4SRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.batch_size = config['train_batch_size']
        self.lmd = config['lmd']
        self.lmd_tf = config['lmd_tf']
        self.tau = config['tau']
        self.sim = config['sim']

        self.tau_plus = config['tau_plus']
        self.beta = config['beta']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.step = config['step']
        self.device = config['device']
        self.g_weight = config['g_weight']
        self.noise_base = config['noise_base']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.fft_layer = BandedFourierLayer(self.hidden_size, self.hidden_size, 0, 1, length=self.max_seq_length)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        self.gnn = GNN(self.hidden_size, self.step)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_3 = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_out = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.kan = FastKAN(self.hidden_size, self.hidden_size)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output_t = trm_output[-1]
        output_t = self.gather_indexes(output_t, item_seq_len - 1)

        output_f = self.fft_layer(input_emb)
        trm_output_f = self.trm_encoder(output_f, extended_attention_mask, output_all_encoded_layers=True)
        output_f = trm_output_f[-1]
        output_f = self.gather_indexes(output_f, item_seq_len - 1)

        return output_t, output_f  # [B H]
    
    def forward_gcn(self, item_seq, item_seq_len):
        alias_inputs, A, items, mask = self._get_slice(item_seq)

        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.hidden_size)

        hidden = self.item_embedding(items)
        # hidden_t = self.linear_1(hidden)

        noise = self.gaussian_noise(hidden, self.noise_base)
        mask1 = item_seq.gt(0).unsqueeze(dim=2)
        noise1 = noise * mask1
        hidden = hidden + noise1
        
        res_hidden = self.gnn(A, hidden)

        def last_hidden(x):
            seq_hidden = torch.gather(x, dim=1, index=alias_inputs)
            # fetch the last hidden state of last timestamp
            ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
            l1 = self.linear_1(ht).view(ht.size(0), 1, ht.size(1))
            l2 = self.linear_2(seq_hidden)

            alpha = self.linear_3(torch.sigmoid(l1 + l2))
            a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
            seq_output = self.linear_out(torch.cat([a, ht], dim=1))
            return seq_output
        
        seq_output, seq_first_output, seq_final_output = [last_hidden(x) for x in res_hidden]
        # seq_output_t = self.kan(seq_output_t)
        
        return seq_output, seq_first_output, seq_final_output  # [B H]

    def my_fft(self, seq):
        f = torch.fft.rfft(seq, dim=1)
        amp = torch.absolute(f)
        phase = torch.angle(f)
        return amp, phase
    
    def calculate_loss(self, interaction):
        # cal loss
        loss = torch.tensor(0.0).to(self.device)
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        loss += self.tff_loss(interaction, item_seq, item_seq_len) * 0.5
        loss += self.gcl_loss(interaction, item_seq, item_seq_len, self.g_weight, self.forward_gcn) * 0.5
        return loss

    def tff_loss(self, interaction, item_seq, item_seq_len):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output_t, seq_output_f = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output_t * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output_t * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'

            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(seq_output_t, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # NCE

        # Time ssl
        aug_item_seq, aug_len = interaction['aug'], interaction['aug_len']
        aug_seq_output_t, aug_seq_output_f = self.forward (aug_item_seq, aug_len)
        nce_logits_t, nce_labels_t = self.info_nce(aug_seq_output_t, seq_output_t, temp=self.tau,
                                                   batch_size=seq_output_t.shape[0],
                                                   sim=self.sim) 
        nce_loss_t = self.nce_fct(nce_logits_t, nce_labels_t)

        # Time-Frequency ssl
        nce_logits_t_f, nce_labels_t_f = self.info_nce(seq_output_f, seq_output_t, temp=self.tau,
                                                       batch_size=seq_output_t.shape[0],
                                                       sim=self.sim)
        nce_loss_t_f = self.nce_fct(nce_logits_t_f, nce_labels_t_f)

        # Frequency ssl
        f_aug_seq_output_amp, f_aug_seq_output_phase = self.my_fft(seq_output_t)
        f_seq_output_amp, f_seq_output_phase = self.my_fft(seq_output_f)

        # Amp ssl
        nce_logits_amp, nce_labels_amp = self.info_nce(f_aug_seq_output_amp, f_seq_output_amp, temp=self.tau,
                                                       batch_size=seq_output_t.shape[0],
                                                       sim=self.sim)
        nce_loss_amp = self.nce_fct(nce_logits_amp, nce_labels_amp)

        # Phase ssl
        nce_logits_phase, nce_labels_phase = self.info_nce(f_aug_seq_output_phase, f_seq_output_phase, temp=self.tau,
                                                           batch_size=seq_output_t.shape[0],
                                                           sim=self.sim)
        nce_loss_phase = self.nce_fct(nce_logits_phase, nce_labels_phase)

        return loss + self.lmd/2 * (
                    self.lmd_tf * nce_loss_t + (1 - self.lmd_tf)/3 * (nce_loss_t_f + nce_loss_phase + nce_loss_amp))
    
    def gcl_loss(self, interaction,item_seq, item_seq_len, cff, forward):
        loss = torch.tensor(0.0).to(self.device)
        
        seq_output, seq_first_output, seq_final_output = forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            # Calculate Bayesian Personalized Ranking (BPR) losses, using positive and negative items and embeddings to calculate scores, and then calculate losses
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type == 'CE'
            # 'CE'(cross entropy), then calculate the loss by embedding the item, using the softmax function
            # item_emb get all
            test_item_emb = self.item_embedding.weight
            # seq_output and all item_emb do similarity calculation # tensor.transpose exchange matrix of two dimensions, transpose(dim0, dim1)
            # torch.matmul is multiplication of tensor,
            # Input can be high dimensional. 2D is normal matrix multiplication like tensor.mm.
            # When there are multiple dimensions, the extra one dimension is brought out as batch, and the other parts do matrix multiplication
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss) or torch.isinf(loss):
            print("rec_loss:", loss)
            loss = 1e-8

        # loss *= cff 
        loss += 0.01 * self.infonce(self.tau, self.kan(seq_output), self.kan(seq_first_output))

        return loss
    
    def gaussian_noise(self, source, noise_base=0.1, dtype=torch.float32):
        x = noise_base + torch.zeros_like(source, dtype=dtype, device=source.device)
        noise = torch.normal(mean=torch.tensor([0.0]).to(source.device), std=x).to(source.device)
        return noise

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels
    
    def infonce(self, temperature, feature1, feature2):
        """
        Computes the InfoNCE loss.
        
        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim], 
                                     where features[:batch_size] are the representations of 
                                     the first set of augmented images, and features[batch_size:] 
                                     are the representations of the second set.
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        features = torch.cat([feature1, feature2])

        # Normalize features to have unit norm
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature

        # Get batch size
        batch_size = features.shape[0] // 2
        
        # Construct labels where each sample's positive pair is in the other view
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output_t, seq_output_f = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output_t, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output_t, seq_output_f = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output_t, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        # Inside the function, a standard deviation (stdv) is first calculated, which is based on the inverse square of embedding_size.
        # Then, all the parameters of the model are iterated over, initializing the data (i.e., the weights) for each parameter to a value drawn from the uniform distribution, ranging from -stdv to stdv.
        # This initialization helps to set the parameters to small random values at the beginning of training for better model training and optimization.
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
            if i == 0:
                first_hidden = hidden
            elif i == self.step - 1:
                final_hidden = hidden
        return hidden, first_hidden, final_hidden




    