#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

import torch.autograd as autograd
from torch.autograd import Variable
from numpy.random import RandomState

import json
import torch.autograd.profiler as profiler
# from pytorch_memlab import LineProfiler
import math

class DensE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 entity_embedding_has_mod=False, relation_embedding_has_mod=False):
        super(DensE, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 1.2
        self.rel_high_bound = 2.0
        
        self.use_abs_norm = True
        self.allow_minus_mod = True
        self.use_entity_phase = False
        self.use_real_part = False
        
        self.criterion = 'he'
        
        if self.criterion == 'glorot':
            mod_range = 1. / np.sqrt(2 * (self.hidden_dim + self.hidden_dim))
        elif self.criterion == 'he':
            mod_range = 1. / np.sqrt(2 * self.hidden_dim)
        
        if self.allow_minus_mod:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range * 2.]), 
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range]), 
                requires_grad=False
            )
            
        self.gamma1 = nn.Parameter(
            torch.Tensor([(self.rel_high_bound + self.epsilon) * mod_range * self.hidden_dim]), 
            requires_grad=False
        )
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.unit_mod = nn.Parameter(
            torch.Tensor([1.]), 
            requires_grad=False
        )
        
        self.zero_ent_phase = nn.Parameter(
            torch.Tensor([0.]), 
            requires_grad=False
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.entity_embedding_has_mod = entity_embedding_has_mod
        self.relation_embedding_has_mod = relation_embedding_has_mod
                   
        self.entity_x = nn.Embedding(self.nentity, self.hidden_dim)
        self.entity_y = nn.Embedding(self.nentity, self.hidden_dim)   
        self.entity_z = nn.Embedding(self.nentity, self.hidden_dim)
        
        self.relation_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_z = nn.Embedding(self.nrelation, self.hidden_dim)
        
        self.init_weights()
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['DensE']:
            raise ValueError('model %s not supported' % model_name)
        if self.use_real_part:
            try:
                assert(self.use_abs_norm == True)
            except:
                raise ValueError('use_abs_norm should be true if you only use real part')
        if (not self.entity_embedding_has_mod) and self.relation_embedding_has_mod:
            raise ValueError('when relation has mod, entity must have mod')
                
            
    def init_weights(self):

        rel_w, rel_x, rel_y, rel_z = self.relation_init(self.nrelation, self.hidden_dim)
        rel_w, rel_x, rel_y, rel_z = torch.from_numpy(rel_w), torch.from_numpy(rel_x), torch.from_numpy(rel_y), torch.from_numpy(rel_z)
        self.relation_w.weight.data = rel_w.type_as(self.relation_w.weight.data)
        self.relation_x.weight.data = rel_x.type_as(self.relation_x.weight.data)
        self.relation_y.weight.data = rel_y.type_as(self.relation_y.weight.data)
        self.relation_z.weight.data = rel_z.type_as(self.relation_z.weight.data)
        
        e_x, e_y, e_z = self.entity_init(self.nentity, self.hidden_dim)
        e_x, e_y, e_z = torch.from_numpy(e_x), torch.from_numpy(e_y), torch.from_numpy(e_z)
        self.entity_x.weight.data = e_x.type_as(self.entity_x.weight.data)
        self.entity_y.weight.data = e_y.type_as(self.entity_y.weight.data)
        self.entity_z.weight.data = e_z.type_as(self.entity_z.weight.data)
        
    def relation_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features
        
        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init rel_mod is: ', s)

        kernel_shape = (n_entries, features)
            
        rel_mod = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rotate_phase = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=kernel_shape)
        theta = np.random.uniform(low=0, high=np.pi, size=kernel_shape)
        phi = np.random.uniform(low=0, high=2*np.pi, size=kernel_shape)
        
        rel_w = rel_mod * np.cos(rotate_phase/2)
        rel_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
        rel_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
        rel_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

        return rel_w, rel_x, rel_y, rel_z

    def entity_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init x, y, z is: ', s)

        # rng = RandomState(456)
        kernel_shape = (n_entries, features)
            
        x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return x, y, z
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # batch_size, negative_sample_size = sample.size(0), 1
            
            head_x = self.entity_x(sample[:, 0]).unsqueeze(1)
            head_y = self.entity_y(sample[:, 0]).unsqueeze(1)
            head_z = self.entity_z(sample[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(sample[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(sample[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(sample[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(sample[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(sample[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(sample[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(sample[:, 1]).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_x = self.entity_x(head_part)
            head_y = self.entity_y(head_part)
            head_z = self.entity_z(head_part)
            
            tail_x = self.entity_x(tail_part[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(tail_part[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(tail_part[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(tail_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(tail_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(tail_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(tail_part[:, 1]).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_x = self.entity_x(head_part[:, 0]).unsqueeze(1)
            head_y = self.entity_y(head_part[:, 0]).unsqueeze(1)
            head_z = self.entity_z(head_part[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(tail_part)
            tail_y = self.entity_y(tail_part)
            tail_z = self.entity_z(tail_part)
            
            rel_w = self.relation_w(head_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(head_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(head_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(head_part[:, 1]).unsqueeze(1)
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'DensE': self.DensE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_x, head_y, head_z, 
                                                rel_w, rel_x, rel_y, rel_z, 
                                                tail_x, tail_y, tail_z, 
                                                mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def _quat_mul(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        
        return (A, B, C, D)

    def rotate(self, x, y, z, rel_w, rel_x, rel_y, rel_z):
        A, B, C, D = self._quat_mul(rel_w, rel_x, rel_y, rel_z, 0, x, y, z)
        return self._quat_mul(A, B, C, D, rel_w, -1.0*rel_x, -1.0*rel_y, -1.0*rel_z)
        # return self._quat_mul(A, B, C, D, rel_w, rel_x, rel_y, rel_z)

    def DensE(self, head_x, head_y, head_z, 
                   rel_w, rel_x, rel_y, rel_z, 
                   tail_x, tail_y, tail_z, 
                   mode):
        pi = 3.14159265358979323846
        assert(self.use_entity_phase == False)
        assert(self.use_real_part == False)
        
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
        
        compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
        compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
        compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z


        if self.relation_embedding_has_mod:
            compute_tail_x = denominator * compute_tail_x
            compute_tail_y = denominator * compute_tail_y
            compute_tail_z = denominator * compute_tail_z
        
        delta_x = (compute_tail_x - tail_x)
        delta_y = (compute_tail_y - tail_y)
        delta_z = (compute_tail_z - tail_z)
        
        score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
        score1 = score1.norm(dim = 0)
        
        x = -x
        y = -y
        z = -z
        compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
        compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
        compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z

        if self.relation_embedding_has_mod:
            compute_head_x = compute_head_x / denominator
            compute_head_y = compute_head_y / denominator
            compute_head_z = compute_head_z / denominator
        
        delta_x2 = (compute_head_x - head_x)
        delta_y2 = (compute_head_y - head_y)
        delta_z2 = (compute_head_z - head_z)
        
        score2 = torch.stack([delta_x2, delta_y2, delta_z2], dim = 0)
        score2 = score2.norm(dim = 0)     
        
        score1 = score1.mean(dim=2)
        score2 = score2.mean(dim=2)

#         score1 = score1.sum(dim=2)
#         score2 = score2.sum(dim=2)
        
        score = (score1 + score2) / 2
        
        score = self.gamma.item() - score
            
        return score, score1, score2, torch.abs(delta_x)

    @staticmethod
    def train_step(model, optimizer, train_iterator, step, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode=mode) # 全是负样本分数 shape: batch_size, neg_size
        
        if step % 500 == 0:
            print(negative_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, head_mod, tail_mod, rel_mod = model(positive_sample) # 正样本分数 shape: batch_size, 1     

        if step % 500 == 0:
            print(positive_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_x.weight.data.norm(p = 3)**3 + 
                model.entity_y.weight.data.norm(p = 3)**3 + 
                model.entity_z.weight.data.norm(p = 3)**3 
            ) / args.batch_size

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
#             'train_hit1': train_hit1
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode)
#                         print(filter_bias, filter_bias.shape, filter_bias.sum())
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics



class HopfEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 entity_embedding_has_mod=False, relation_embedding_has_mod=False):
        super(HopfEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 1.2
        self.rel_high_bound = 2.0
        
        self.use_abs_norm = True
        self.allow_minus_mod = True
        self.use_entity_phase = False
        self.use_real_part = False
        
        self.criterion = 'he'
        
        if self.criterion == 'glorot':
            mod_range = 1. / np.sqrt(2 * (self.hidden_dim + self.hidden_dim))
        elif self.criterion == 'he':
            mod_range = 1. / np.sqrt(2 * self.hidden_dim)
        
        if self.allow_minus_mod:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range * 2.]), 
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range]), 
                requires_grad=False
            )
            
        self.gamma1 = nn.Parameter(
            torch.Tensor([(self.rel_high_bound + self.epsilon) * mod_range * self.hidden_dim]), 
            requires_grad=False
        )
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.unit_mod = nn.Parameter(
            torch.Tensor([1.]), 
            requires_grad=False
        )
        
        self.zero_ent_phase = nn.Parameter(
            torch.Tensor([0.]), 
            requires_grad=False
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.entity_embedding_has_mod = entity_embedding_has_mod
        self.relation_embedding_has_mod = relation_embedding_has_mod
                   
        self.entity_x = nn.Embedding(self.nentity, self.hidden_dim)
        self.entity_y = nn.Embedding(self.nentity, self.hidden_dim)   
        self.entity_z = nn.Embedding(self.nentity, self.hidden_dim)
        
        self.relation_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_z = nn.Embedding(self.nrelation, self.hidden_dim)
        
        self.init_weights()
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['HopfE']:
            raise ValueError('model %s not supported' % model_name)
        if self.use_real_part:
            try:
                assert(self.use_abs_norm == True)
            except:
                raise ValueError('use_abs_norm should be true if you only use real part')
        if (not self.entity_embedding_has_mod) and self.relation_embedding_has_mod:
            raise ValueError('when relation has mod, entity must have mod')
                
            
    def init_weights(self):

        rel_w, rel_x, rel_y, rel_z = self.relation_init(self.nrelation, self.hidden_dim)
        rel_w, rel_x, rel_y, rel_z = torch.from_numpy(rel_w), torch.from_numpy(rel_x), torch.from_numpy(rel_y), torch.from_numpy(rel_z)
        self.relation_w.weight.data = rel_w.type_as(self.relation_w.weight.data)
        self.relation_x.weight.data = rel_x.type_as(self.relation_x.weight.data)
        self.relation_y.weight.data = rel_y.type_as(self.relation_y.weight.data)
        self.relation_z.weight.data = rel_z.type_as(self.relation_z.weight.data)
        
        e_x, e_y, e_z = self.entity_init(self.nentity, self.hidden_dim)
        e_x, e_y, e_z = torch.from_numpy(e_x), torch.from_numpy(e_y), torch.from_numpy(e_z)
        self.entity_x.weight.data = e_x.type_as(self.entity_x.weight.data)
        self.entity_y.weight.data = e_y.type_as(self.entity_y.weight.data)
        self.entity_z.weight.data = e_z.type_as(self.entity_z.weight.data)
        
    def relation_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features
        
        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init rel_mod is: ', s)

        kernel_shape = (n_entries, features)
            
        rel_mod = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rotate_phase = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=kernel_shape)
        theta = np.random.uniform(low=0, high=np.pi, size=kernel_shape)
        phi = np.random.uniform(low=0, high=2*np.pi, size=kernel_shape)
        
        rel_w = rel_mod * np.cos(rotate_phase/2)
        rel_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
        rel_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
        rel_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

        return rel_w, rel_x, rel_y, rel_z

    def entity_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init x, y, z is: ', s)

        # rng = RandomState(456)
        kernel_shape = (n_entries, features)
            
        x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return x, y, z
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # batch_size, negative_sample_size = sample.size(0), 1
            
            head_x = self.entity_x(sample[:, 0]).unsqueeze(1)
            head_y = self.entity_y(sample[:, 0]).unsqueeze(1)
            head_z = self.entity_z(sample[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(sample[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(sample[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(sample[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(sample[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(sample[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(sample[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(sample[:, 1]).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_x = self.entity_x(head_part)
            head_y = self.entity_y(head_part)
            head_z = self.entity_z(head_part)
            
            tail_x = self.entity_x(tail_part[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(tail_part[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(tail_part[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(tail_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(tail_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(tail_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(tail_part[:, 1]).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_x = self.entity_x(head_part[:, 0]).unsqueeze(1)
            head_y = self.entity_y(head_part[:, 0]).unsqueeze(1)
            head_z = self.entity_z(head_part[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(tail_part)
            tail_y = self.entity_y(tail_part)
            tail_z = self.entity_z(tail_part)
            
            rel_w = self.relation_w(head_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(head_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(head_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(head_part[:, 1]).unsqueeze(1)
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'HopfE': self.HopfE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_x, head_y, head_z, 
                                                rel_w, rel_x, rel_y, rel_z, 
                                                tail_x, tail_y, tail_z, 
                                                mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def _quat_mul(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        
        return (A, B, C, D)

    def rotate(self, x, y, z, rel_w, rel_x, rel_y, rel_z):
        A, B, C, D = self._quat_mul(rel_w, rel_x, rel_y, rel_z, 0, x, y, z)
        return self._quat_mul(A, B, C, D, rel_w, -1.0*rel_x, -1.0*rel_y, -1.0*rel_z)
        # return self._quat_mul(A, B, C, D, rel_w, rel_x, rel_y, rel_z)

    def HopfE(self, head_x, head_y, head_z, 
                   rel_w, rel_x, rel_y, rel_z, 
                   tail_x, tail_y, tail_z, 
                   mode):
        pi = 3.14159265358979323846
        assert(self.use_entity_phase == False)
        assert(self.use_real_part == False)
        
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
        
        # compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
        # compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
        # compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
        _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z)


        if self.relation_embedding_has_mod:
            compute_tail_x = denominator * compute_tail_x
            compute_tail_y = denominator * compute_tail_y
            compute_tail_z = denominator * compute_tail_z
        
        delta_x = (compute_tail_x - tail_x)
        delta_y = (compute_tail_y - tail_y)
        delta_z = (compute_tail_z - tail_z)
        
        score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
        score1 = score1.norm(dim = 0)
        
        x = -x
        y = -y
        z = -z
        # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
        # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
        # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
        _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z)

        if self.relation_embedding_has_mod:
            compute_head_x = compute_head_x / denominator
            compute_head_y = compute_head_y / denominator
            compute_head_z = compute_head_z / denominator
        
        delta_x2 = (compute_head_x - head_x)
        delta_y2 = (compute_head_y - head_y)
        delta_z2 = (compute_head_z - head_z)
        
        score2 = torch.stack([delta_x2, delta_y2, delta_z2], dim = 0)
        score2 = score2.norm(dim = 0)     
        
        score1 = score1.mean(dim=2)
        score2 = score2.mean(dim=2)

        
        score = (score1 + score2) / 2
        
        score = self.gamma.item() - score
            
        return score, score1, score2, torch.abs(delta_x)

    @staticmethod
    def train_step(model, optimizer, train_iterator, step, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode=mode) # 全是负样本分数 shape: batch_size, neg_size
        
        if step % 500 == 0:
            print(negative_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, head_mod, tail_mod, rel_mod = model(positive_sample) # 正样本分数 shape: batch_size, 1     

        if step % 500 == 0:
            print(positive_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_x.weight.data.norm(p = 3)**3 + 
                model.entity_y.weight.data.norm(p = 3)**3 + 
                model.entity_z.weight.data.norm(p = 3)**3 
            ) / args.batch_size

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics


class HopfSemanticsEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 entity_embedding_has_mod=False, relation_embedding_has_mod=False, params=None):
        super(HopfSemanticsEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 1.2
        self.rel_high_bound = 2.0
        
        self.use_abs_norm = True
        self.allow_minus_mod = True
        self.use_entity_phase = False
        self.use_real_part = False
        
        self.criterion = 'he'
        
        if self.criterion == 'glorot':
            mod_range = 1. / np.sqrt(2 * (self.hidden_dim + self.hidden_dim))
        elif self.criterion == 'he':
            mod_range = 1. / np.sqrt(2 * self.hidden_dim)
        
        if self.allow_minus_mod:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range * 2.]), 
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range]), 
                requires_grad=False
            )
            
        self.gamma1 = nn.Parameter(
            torch.Tensor([(self.rel_high_bound + self.epsilon) * mod_range * self.hidden_dim]), 
            requires_grad=False
        )
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.unit_mod = nn.Parameter(
            torch.Tensor([1.]), 
            requires_grad=False
        )
        
        self.zero_ent_phase = nn.Parameter(
            torch.Tensor([0.]), 
            requires_grad=False
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.entity_embedding_has_mod = entity_embedding_has_mod
        self.relation_embedding_has_mod = relation_embedding_has_mod
                   
        self.entity_x = nn.Embedding(self.nentity, self.hidden_dim)
        self.entity_y = nn.Embedding(self.nentity, self.hidden_dim)   
        self.entity_z = nn.Embedding(self.nentity, self.hidden_dim)
        
        self.relation_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_z = nn.Embedding(self.nrelation, self.hidden_dim)
        
        self.init_weights()
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['HopfSemanticsE']:
            raise ValueError('model %s not supported' % model_name)
        if self.use_real_part:
            try:
                assert(self.use_abs_norm == True)
            except:
                raise ValueError('use_abs_norm should be true if you only use real part')
        if (not self.entity_embedding_has_mod) and self.relation_embedding_has_mod:
            raise ValueError('when relation has mod, entity must have mod')
        
        if params is None:
            params={    
                'embeddings_path': '../../data/FB15K237_context/limit1_v3/embeddings.npy',
                'char_vocab_path': '../../data/FB15K237_context/limit1_v3/char2idx.json',
                'char_feature_size': 50,
                'char_embed_dim': 50,
                'max_word_len_entity': 10,
                'conv_filter_size': 3,
                'drop_rate': 0.0,
                'max_sent_len': 16,
                'entity_indices_file': '../../data/FB15K237_context/limit1_v3/entity_context_indices.json',
                'word2idx_path': '../../data/FB15K237_context/limit1_v3/word2idx.json',
                'all_word_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/word_indices_h.npy',
                'all_char_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/char_indices_h.npy',
                'mask_file_h': '../../data/FB15K237_context/limit1_v3/mask_h.npy',
                'all_word_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/word_indices_t.npy',
                'all_char_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/char_indices_t.npy',
                'mask_file_t': '../../data/FB15K237_context/limit1_v3/mask_t.npy',
                'padding': 1,
                'checkpoint_json_path': './result_hopfe_rot_2/HopfE.json',
                'wassertein_approx': False,
                'method_to_induce_semantics': 'mult'
            }
        self.embeddings = np.load(params['embeddings_path'])
        with open(params['char_vocab_path'], 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.all_word_token_ids_h = np.load(params['all_word_token_ids_file_h'])
        self.all_char_token_ids_h = np.load(params['all_char_token_ids_file_h'])
        self.all_word_token_ids_t = np.load(params['all_word_token_ids_file_t'])
        self.all_char_token_ids_t = np.load(params['all_char_token_ids_file_t'])
        self.PADDING = params['padding']
        self.max_sent_len = params['max_sent_len']
        self.conv_filter_size = params['conv_filter_size']
        self.max_char_len = params['max_word_len_entity']
        self.char_embed_dim = params['char_embed_dim']
        self.drop_rate = params['drop_rate']
        self.mask_h = np.load(params['mask_file_h'])
        self.mask_t = np.load(params['mask_file_t'])
        self.char_feature_size = params['char_feature_size']

        self.semanticE = CbowE({'embeddings': self.embeddings, 
            'char_vocab': self.char_vocab, 
            'char_feature_size': self.char_feature_size, 
            'char_embed_dim': self.char_embed_dim, 
            'max_word_len_entity': self.max_char_len, 
            'conv_filter_size': self.conv_filter_size, 
            'drop_rate': self.drop_rate})
        # self.linear_proj = nn.Linear(50, 2*self.hidden_dim)
        self.linear_proj = nn.Linear(100, 2*self.hidden_dim)

        self.CUDA = torch.cuda.is_available()
        self.avg_semantics = True
        self.add_semantics_after_map = True
        self.wassertein_approx = params.get('wasserstein_approx', False)
        self.num_heads = 1
        self.linear_proj2 = nn.Linear(50, 4*self.hidden_dim)
        self.param_ent_s_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)
        self.param_ent_x_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)

        # options: [hopf, mult, mlp, init]
        self.method_to_induce_semantics = params.get('method_to_induce_semantics', 'mult')
        self.linear_proj_mlp = nn.Linear(3*self.hidden_dim+50, 3*self.hidden_dim)
        self.linear_proj_init = nn.Linear(50, 3*self.hidden_dim)

    def init_weights(self):

        rel_w, rel_x, rel_y, rel_z = self.relation_init(self.nrelation, self.hidden_dim)
        rel_w, rel_x, rel_y, rel_z = torch.from_numpy(rel_w), torch.from_numpy(rel_x), torch.from_numpy(rel_y), torch.from_numpy(rel_z)
        self.relation_w.weight.data = rel_w.type_as(self.relation_w.weight.data)
        self.relation_x.weight.data = rel_x.type_as(self.relation_x.weight.data)
        self.relation_y.weight.data = rel_y.type_as(self.relation_y.weight.data)
        self.relation_z.weight.data = rel_z.type_as(self.relation_z.weight.data)
        
        e_x, e_y, e_z = self.entity_init(self.nentity, self.hidden_dim)
        e_x, e_y, e_z = torch.from_numpy(e_x), torch.from_numpy(e_y), torch.from_numpy(e_z)
        self.entity_x.weight.data = e_x.type_as(self.entity_x.weight.data)
        self.entity_y.weight.data = e_y.type_as(self.entity_y.weight.data)
        self.entity_z.weight.data = e_z.type_as(self.entity_z.weight.data)
        
    def relation_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features
        
        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init rel_mod is: ', s)

        kernel_shape = (n_entries, features)
            
        rel_mod = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rotate_phase = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=kernel_shape)
        theta = np.random.uniform(low=0, high=np.pi, size=kernel_shape)
        phi = np.random.uniform(low=0, high=2*np.pi, size=kernel_shape)
        
        rel_w = rel_mod * np.cos(rotate_phase/2)
        rel_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
        rel_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
        rel_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

        return rel_w, rel_x, rel_y, rel_z

    def entity_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init x, y, z is: ', s)

        # rng = RandomState(456)
        kernel_shape = (n_entries, features)
            
        x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return x, y, z
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # batch_size, negative_sample_size = sample.size(0), 1
            
            head_x = self.entity_x(sample[:, 0]).unsqueeze(1)
            head_y = self.entity_y(sample[:, 0]).unsqueeze(1)
            head_z = self.entity_z(sample[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(sample[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(sample[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(sample[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(sample[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(sample[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(sample[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(sample[:, 1]).unsqueeze(1)

            self.batch_h = sample[:, 0]
            self.batch_r = sample[:, 1]
            self.batch_t = sample[:, 2]
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_x = self.entity_x(head_part)
            head_y = self.entity_y(head_part)
            head_z = self.entity_z(head_part)
            
            tail_x = self.entity_x(tail_part[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(tail_part[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(tail_part[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(tail_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(tail_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(tail_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(tail_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part
            self.batch_r = tail_part[:, 1]
            self.batch_t = tail_part[:, 2]
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_x = self.entity_x(head_part[:, 0]).unsqueeze(1)
            head_y = self.entity_y(head_part[:, 0]).unsqueeze(1)
            head_z = self.entity_z(head_part[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(tail_part)
            tail_y = self.entity_y(tail_part)
            tail_z = self.entity_z(tail_part)
            
            rel_w = self.relation_w(head_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(head_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(head_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(head_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part[:, 0]
            self.batch_r = head_part[:, 1]
            self.batch_t = tail_part
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'HopfSemanticsE': self.HopfSemanticsE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_x, head_y, head_z, 
                                                rel_w, rel_x, rel_y, rel_z, 
                                                tail_x, tail_y, tail_z, 
                                                mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def _quat_mul(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        
        return (A, B, C, D)

    def rotate(self, x, y, z, rel_w, rel_x, rel_y, rel_z):
        A, B, C, D = self._quat_mul(rel_w, rel_x, rel_y, rel_z, 0, x, y, z)
        return self._quat_mul(A, B, C, D, rel_w, -1.0*rel_x, -1.0*rel_y, -1.0*rel_z)
        # return self._quat_mul(A, B, C, D, rel_w, rel_x, rel_y, rel_z)

    def _calc_semantic_score_optim_after_map(self, Hs, Ts, Hs2, Ts2, H, T, set_sizes_h, set_sizes_t):

        # single_mode = True
        # if len(Hs.shape)==4:
        #     single_mode = False
        #     neg_samples = Hs.shape[1]
        #     Hs = Hs.reshape((-1,Hs.shape[2],Hs.shape[3]))
        #     set_sizes_h = set_sizes_h * neg_samples
        # if len(Ts.shape)==4:
        #     single_mode = False
        #     neg_samples = Ts.shape[1]
        #     Ts = Ts.reshape((-1,Ts.shape[2],Ts.shape[3]))
        #     set_sizes_t = set_sizes_t * neg_samples

        # s_b = Rs[:,:,0]
        # x_b = Rs[:,:,1]
        # y_b = Rs[:,:,2]
        # z_b = Rs[:,:,3]
        # denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        # s_b = s_b / denominator_b
        # x_b = x_b / denominator_b
        # y_b = y_b / denominator_b
        # z_b = z_b / denominator_b

        if self.avg_semantics:
            Hs2 = torch.mean(Hs2.reshape(Hs2.shape[0]//self.PADDING,self.PADDING,Hs2.shape[1],Hs2.shape[2],Hs2.shape[3]), dim=1)
            Ts2 = torch.mean(Ts2.reshape(Ts2.shape[0]//self.PADDING,self.PADDING,Ts2.shape[1],Ts2.shape[2],Ts2.shape[3]), dim=1)

        H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        
        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]
        # denominator_a = torch.sqrt(s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
        # s_a = s_a / denominator_a
        # x_a = x_a / denominator_a
        # y_a = y_a / denominator_a
        # z_a = z_a / denominator_a

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]
        # denominator_c = torch.sqrt(s_c ** 2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
        # s_c = s_c / denominator_c
        # x_c = x_c / denominator_c
        # y_c = y_c / denominator_c
        # z_c = z_c / denominator_c

        # Rs0 = torch.cat((s_b.unsqueeze(-1),x_b.unsqueeze(-1),y_b.unsqueeze(-1),z_b.unsqueeze(-1)), dim=-1)

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        # A, B, C, D = self._quat_mul(H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # A, B, C, D = self._quat_mul(H[:,:,0], 1+H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # num = torch.sqrt(H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        # den = torch.sqrt(1+2*H[:,:,1] + H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
        num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
        den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        norm = torch.div(num,den)
        Ah = A*norm
        Bh = B*norm
        Ch = C*norm
        Dh = D*norm        
        # Hs[:,:,0] = A
        # Hs[:,:,1] = B
        # Hs[:,:,2] = C
        # Hs[:,:,3] = D
        Hs0 = torch.cat((Ah.unsqueeze(-1),Bh.unsqueeze(-1),Ch.unsqueeze(-1),Dh.unsqueeze(-1)), dim=-1)

        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        # A, B, C, D = self._quat_mul(T[:,:,0], T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # A, B, C, D = self._quat_mul(T[:,:,0], 1+ T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # num = torch.sqrt(T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        # den = torch.sqrt(1+2*T[:,:,1] + T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
        num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
        den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        norm = torch.div(num,den)
        At = A*norm
        Bt = B*norm
        Ct = C*norm
        Dt = D*norm        
        # Ts[:,:,0] = A
        # Ts[:,:,1] = B
        # Ts[:,:,2] = C
        # Ts[:,:,3] = D
        Ts0 = torch.cat((At.unsqueeze(-1),Bt.unsqueeze(-1),Ct.unsqueeze(-1),Dt.unsqueeze(-1)), dim=-1)

        if self.wassertein_approx:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])

            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            Hs1 = torch.cat([Ahr,Bhr,Chr,Dhr], dim=-1)
            Ts1 = torch.cat([Atr,Btr,Ctr,Dtr], dim=-1)

            # find the wass dist
            dist, P, C = self.sinkhorn(Hs1, Ts1)

            # dist = dist.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])    
            delta_min1 = torch.min(P, dim=-1)
            dr3 = delta_min1[1].unsqueeze(-1).repeat([1,1,self.hidden_dim])

            # Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3)
            Btrm = torch.gather(Btr,dim=1,index=dr3)
            Ctrm = torch.gather(Ctr,dim=1,index=dr3)
            Dtrm = torch.gather(Dtr,dim=1,index=dr3)

            d1 = Ahr-Atrm
            d2 = Bhr-Btrm
            d3 = Chr-Ctrm
            d4 = Dhr-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            d7 = torch.mean(d7,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

            # return dist
        else:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
                Hs2 = Hs2.repeat([1,Ts2.shape[1],1,1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])
                Ts2 = Ts2.repeat([1,Hs2.shape[1],1,1])

            # Map with the semantics quaternion
            # TODO: construct the sem_quat quaternion using another method
            # sem_quat = torch.cat((Hs2,Ts2), dim=-1)
            sem_quat_h = Hs2
            denominator_sem_q_h = torch.sqrt(sem_quat_h[:,:,:,0] ** 2 + sem_quat_h[:,:,:,1] ** 2 + sem_quat_h[:,:,:,2] ** 2 + sem_quat_h[:,:,:,3] ** 2)
            sem_quat_h_w = sem_quat_h[:,:,:,0] / denominator_sem_q_h
            sem_quat_h_x = sem_quat_h[:,:,:,1] / denominator_sem_q_h
            sem_quat_h_y = sem_quat_h[:,:,:,2] / denominator_sem_q_h
            sem_quat_h_z = sem_quat_h[:,:,:,3] / denominator_sem_q_h
            Hr = torch.cat((Ah.unsqueeze(-1), Bh.unsqueeze(-1), Ch.unsqueeze(-1), Dh.unsqueeze(-1)), dim=-1)
            Ah2, Bh2, Ch2, Dh2 = self._quat_mul(Hr[:,:,:,0], Hr[:,:,:,1], Hr[:,:,:,2], Hr[:,:,:,3], sem_quat_h_w, sem_quat_h_x, sem_quat_h_y, sem_quat_h_z)


            sem_quat_t = Ts2
            denominator_sem_q_t = torch.sqrt(sem_quat_t[:,:,:,0] ** 2 + sem_quat_t[:,:,:,1] ** 2 + sem_quat_t[:,:,:,2] ** 2 + sem_quat_t[:,:,:,3] ** 2)
            sem_quat_t_w = sem_quat_t[:,:,:,0] / denominator_sem_q_t
            sem_quat_t_x = sem_quat_t[:,:,:,1] / denominator_sem_q_t
            sem_quat_t_y = sem_quat_t[:,:,:,2] / denominator_sem_q_t
            sem_quat_t_z = sem_quat_t[:,:,:,3] / denominator_sem_q_t
            Tr = torch.cat((At.unsqueeze(-1), Bt.unsqueeze(-1), Ct.unsqueeze(-1), Dt.unsqueeze(-1)), dim=-1)
            At2, Bt2, Ct2, Dt2 = self._quat_mul(Tr[:,:,:,0], Tr[:,:,:,1], Tr[:,:,:,2], Tr[:,:,:,3], sem_quat_t_w, sem_quat_t_x, sem_quat_t_y, sem_quat_t_z)


            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah2.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh2.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch2.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh2.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At2.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt2.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct2.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt2.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            # find the pairwise dist
            # dw = torch.cdist(Ahr,Atr).reshape(Ah.shape[0]//self.num_heads,Ah.shape[1],self.num_heads,self.num_heads)
            # dx = torch.cdist(Bhr,Btr).reshape(Bh.shape[0]//self.num_heads,Bh.shape[1],self.num_heads,self.num_heads)
            # dy = torch.cdist(Chr,Ctr).reshape(Ch.shape[0]//self.num_heads,Ch.shape[1],self.num_heads,self.num_heads)
            # dz = torch.cdist(Dhr,Dtr).reshape(Dh.shape[0]//self.num_heads,Dh.shape[1],self.num_heads,self.num_heads)
            dw = torch.cdist(Ahr,Atr)
            dx = torch.cdist(Bhr,Btr)
            dy = torch.cdist(Chr,Ctr)
            dz = torch.cdist(Dhr,Dtr)
                       
            delta = dw + dx + dy + dz

            # take the min, min or max, min
            delta_min1 = torch.min(delta, dim=-1)
            delta_min2 = torch.min(delta_min1[0], dim=-1)
            dr1 = delta_min1[1]
            dr2 = delta_min2[1].unsqueeze(-1)
            # dr3 = torch.cat((dr2,dr1[dr2]), dim=-1)
            dr1g = torch.gather(dr1,dim=1,index=dr2)
            dr3 = torch.cat((dr2,dr1g), dim=-1)

            Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Btrm = torch.gather(Btr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Ctrm = torch.gather(Ctr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dtrm = torch.gather(Dtr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))

            d1 = Ahrm-Atrm
            d2 = Bhrm-Btrm
            d3 = Chrm-Ctrm
            d4 = Dhrm-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

    def _calc_semantic_score_optim(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t):

        # single_mode = True
        # if len(Hs.shape)==4:
        #     single_mode = False
        #     neg_samples = Hs.shape[1]
        #     Hs = Hs.reshape((-1,Hs.shape[2],Hs.shape[3]))
        #     set_sizes_h = set_sizes_h * neg_samples
        # if len(Ts.shape)==4:
        #     single_mode = False
        #     neg_samples = Ts.shape[1]
        #     Ts = Ts.reshape((-1,Ts.shape[2],Ts.shape[3]))
        #     set_sizes_t = set_sizes_t * neg_samples

        # s_b = Rs[:,:,0]
        # x_b = Rs[:,:,1]
        # y_b = Rs[:,:,2]
        # z_b = Rs[:,:,3]
        # denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        # s_b = s_b / denominator_b
        # x_b = x_b / denominator_b
        # y_b = y_b / denominator_b
        # z_b = z_b / denominator_b

        if self.avg_semantics:
            Hs = torch.mean(Hs.reshape(Hs.shape[0]//self.PADDING,self.PADDING,Hs.shape[1],Hs.shape[2],Hs.shape[3]), dim=1)
            Ts = torch.mean(Ts.reshape(Ts.shape[0]//self.PADDING,self.PADDING,Ts.shape[1],Ts.shape[2],Ts.shape[3]), dim=1)
        else:
            H = torch.repeat_interleave(H, set_sizes_h, dim=0)
            T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        
        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]
        # denominator_a = torch.sqrt(s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
        # s_a = s_a / denominator_a
        # x_a = x_a / denominator_a
        # y_a = y_a / denominator_a
        # z_a = z_a / denominator_a

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]
        # denominator_c = torch.sqrt(s_c ** 2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
        # s_c = s_c / denominator_c
        # x_c = x_c / denominator_c
        # y_c = y_c / denominator_c
        # z_c = z_c / denominator_c

        # Rs0 = torch.cat((s_b.unsqueeze(-1),x_b.unsqueeze(-1),y_b.unsqueeze(-1),z_b.unsqueeze(-1)), dim=-1)

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        # A, B, C, D = self._quat_mul(H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # A, B, C, D = self._quat_mul(H[:,:,0], 1+H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # num = torch.sqrt(H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        # den = torch.sqrt(1+2*H[:,:,1] + H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
        num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
        den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        norm = torch.div(num,den)
        A = A*norm
        B = B*norm
        C = C*norm
        D = D*norm        
        # Hs[:,:,0] = A
        # Hs[:,:,1] = B
        # Hs[:,:,2] = C
        # Hs[:,:,3] = D
        Hs0 = torch.cat((A.unsqueeze(-1),B.unsqueeze(-1),C.unsqueeze(-1),D.unsqueeze(-1)), dim=-1)

        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        # A, B, C, D = self._quat_mul(T[:,:,0], T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # A, B, C, D = self._quat_mul(T[:,:,0], 1+ T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # num = torch.sqrt(T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        # den = torch.sqrt(1+2*T[:,:,1] + T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
        num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
        den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        norm = torch.div(num,den)
        A = A*norm
        B = B*norm
        C = C*norm
        D = D*norm        
        # Ts[:,:,0] = A
        # Ts[:,:,1] = B
        # Ts[:,:,2] = C
        # Ts[:,:,3] = D
        Ts0 = torch.cat((A.unsqueeze(-1),B.unsqueeze(-1),C.unsqueeze(-1),D.unsqueeze(-1)), dim=-1)

        if not self.avg_semantics:
            Hs0 = Hs0.unsqueeze(dim=1).reshape(Hs0.shape[0]//self.PADDING,self.PADDING,Hs0.shape[1],Hs0.shape[2],Hs.shape[3])
            Ts0 = Ts0.unsqueeze(dim=1).reshape(Ts0.shape[0]//self.PADDING,self.PADDING,Ts0.shape[1],Ts0.shape[2],Hs.shape[3])

            Hs0 = torch.mean(Hs0, dim=1)
            Ts0 = torch.mean(Ts0, dim=1)

        delta_w = Hs0[:,:,:,0] - Ts0[:,:,:,0]
        delta_x = Hs0[:,:,:,1] - Ts0[:,:,:,1]
        delta_y = Hs0[:,:,:,2] - Ts0[:,:,:,2]
        delta_z = Hs0[:,:,:,3] - Ts0[:,:,:,3]

        # if not single_mode:
        #     delta_w = delta_w.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_x = delta_x.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_y = delta_y.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_z = delta_z.unsqueeze(-1).reshape((-1,neg_samples))

        # return delta_w, delta_x, delta_y, delta_z
        score = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
        score = score.norm(dim = 0)
        score = score.mean(dim=2)
        return score

        # Rs = torch.repeat_interleave(Rs0, set_sizes_h, dim=0)
        # # Rs = Rs0
        # # Rotate in 4-D using the relation quaternion
        # A, B, C, D = self._quat_mul(Hs0[:,:,0], Hs0[:,:,1], Hs0[:,:,2], Hs0[:,:,3], Rs[:,:,0], Rs[:,:,1], Rs[:,:,2], Rs[:,:,3])
        # # Hs[:,:,0] = A
        # # Hs[:,:,1] = B
        # # Hs[:,:,2] = C
        # # Hs[:,:,3] = D
        # Hs1 = torch.cat((A.unsqueeze(-1),B.unsqueeze(-1),C.unsqueeze(-1),D.unsqueeze(-1)), dim=-1)

        # '''# Repeat elements along Hs t times
        # # Repeat set elements along Ts h times
        # set_sizes_h2 = torch.repeat_interleave(set_sizes_t, set_sizes_h)
        # set_sizes_t2 = torch.repeat_interleave(set_sizes_h, set_sizes_t)
        # Hs2 = torch.repeat_interleave(Hs1, set_sizes_h2, dim=0)
        # Ts2 = torch.repeat_interleave(Ts0, set_sizes_t2, dim=0)
        # gather_batch_indices = []
        # cur_batch_indices = []
        # j = 0
        # for i in range(Ts0.shape[0]):
        #     cur_batch_indices.append(i)
        #     if len(cur_batch_indices)==set_sizes_t[j]:
        #         cur_batch_indices = cur_batch_indices*set_sizes_h[j]
        #         gather_batch_indices.extend(cur_batch_indices)
        #         cur_batch_indices = []
        #         j += 1
        # if self.CUDA:
        #     gather_batch_indices = torch.tensor(gather_batch_indices).cuda()
        # else:
        #     gather_batch_indices = torch.tensor(gather_batch_indices)
        #     # gather_indices = torch.ones(Ts2.shape)*gather_batch_indices 
        # gather_indices = gather_batch_indices.unsqueeze(-1).unsqueeze(-1).repeat([1,Ts2.shape[1],Ts2.shape[2]])
        # Ts2 = torch.gather(Ts2, 0, gather_indices)
        # set_sizes_gather = set_sizes_h*set_sizes_t'''
        # Ts2 = Ts0
        # Hs2 = Hs1

        # score_r = (Hs2[:,:,0] * Ts2[:,:,0] + Hs2[:,:,1] * Ts2[:,:,1] + Hs2[:,:,2] * Ts2[:,:,2] + Hs2[:,:,3] * Ts2[:,:,3])
        # score_r = -torch.sum(score_r, -1)
        # score_r = score_r.view([score_r.shape[0]//self.PADDING,self.PADDING]).unsqueeze(1)
        # pooled_score = torch.nn.MaxPool1d(self.PADDING, stride=self.PADDING)(score_r).squeeze()
        # return pooled_score


    def _calc_semantic_score_optim_mlp(self, Hs, Ts, Hs2, Ts2, H, T, set_sizes_h, set_sizes_t):

        # assert self.avg_semantics==True, "Semantics must be averaged for mlp option"

        # if self.avg_semantics:
        #     Hs2 = torch.mean(Hs2.reshape(Hs2.shape[0]//self.PADDING,self.PADDING,Hs2.shape[1],Hs2.shape[2],Hs2.shape[3]), dim=1)
        #     Ts2 = torch.mean(Ts2.reshape(Ts2.shape[0]//self.PADDING,self.PADDING,Ts2.shape[1],Ts2.shape[2],Ts2.shape[3]), dim=1)

        # H = self.linear_proj_mlp(torch.cat((H,Hs2), dim=-2)).reshape((H.shape[0],H.shape[1],self.hidden_dim,3))
        # T = self.linear_proj_mlp(torch.cat((T,Ts2), dim=-2)).reshape((T.shape[0],T.shape[1],self.hidden_dim,3))

        H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        

        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
        num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
        den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        norm = torch.div(num,den)
        Ah = A*norm
        Bh = B*norm
        Ch = C*norm
        Dh = D*norm
        Hs0 = torch.cat((Ah.unsqueeze(-1),Bh.unsqueeze(-1),Ch.unsqueeze(-1),Dh.unsqueeze(-1)), dim=-1)

        A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
        num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
        den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        norm = torch.div(num,den)
        At = A*norm
        Bt = B*norm
        Ct = C*norm
        Dt = D*norm       
        Ts0 = torch.cat((At.unsqueeze(-1),Bt.unsqueeze(-1),Ct.unsqueeze(-1),Dt.unsqueeze(-1)), dim=-1)

        delta_w = Hs0[:,:,:,0] - Ts0[:,:,:,0]
        delta_x = Hs0[:,:,:,1] - Ts0[:,:,:,1]
        delta_y = Hs0[:,:,:,2] - Ts0[:,:,:,2]
        delta_z = Hs0[:,:,:,3] - Ts0[:,:,:,3]

        score = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
        score = score.norm(dim = 0)
        score = score.mean(dim=2)
        return score

    def HopfSemanticsE(self, head_x, head_y, head_z, 
                   rel_w, rel_x, rel_y, rel_z, 
                   tail_x, tail_y, tail_z, 
                   mode):
        pi = 3.14159265358979323846
        assert(self.use_entity_phase == False)
        assert(self.use_real_part == False)
        
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
    
        # if not self.add_semantics_after_map: 
        if self.method_to_induce_semantics=='hopf':   
            # compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
            # compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
            # compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
            _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z)

            if self.relation_embedding_has_mod:
                compute_tail_x = denominator * compute_tail_x
                compute_tail_y = denominator * compute_tail_y
                compute_tail_z = denominator * compute_tail_z

            words_a, chars_a, mask_h, set_sizes_h = self.get_entity_properties_from_npy(self.batch_h, pos='head')
            words_b, chars_b, mask_t, set_sizes_t = self.get_entity_properties_from_npy(self.batch_t, pos='tail')
            
            if self.CUDA:
                words_a, chars_a, mask_h, set_sizes_h = torch.from_numpy(words_a).cuda(), torch.from_numpy(chars_a).cuda(), torch.from_numpy(mask_h).cuda(), torch.from_numpy(set_sizes_h).cuda()
                words_b, chars_b, mask_t, set_sizes_t = torch.from_numpy(words_b).cuda(), torch.from_numpy(chars_b).cuda(), torch.from_numpy(mask_t).cuda(), torch.from_numpy(set_sizes_t).cuda()
            else:
                words_a, chars_a, mask_h, set_sizes_h = torch.from_numpy(words_a), torch.from_numpy(chars_a), torch.from_numpy(mask_h), torch.from_numpy(set_sizes_h)
                words_b, chars_b, mask_t, set_sizes_t = torch.from_numpy(words_b), torch.from_numpy(chars_b), torch.from_numpy(mask_t), torch.from_numpy(set_sizes_t)
            semantics_a = self.semanticE(words_a, chars_a, mask_h)
            if len(semantics_a.shape)==2:        
                semantics_a = self.linear_proj(semantics_a).reshape((semantics_a.shape[0],self.hidden_dim,2))
                norm_a = torch.sqrt(torch.sum(semantics_a**2, dim=-1)).unsqueeze(-1).repeat([1,1,semantics_a.shape[-1]])
                semantics_a = torch.div(semantics_a,norm_a)
                semantics_a = semantics_a.unsqueeze(dim=1)
            else:
                semantics_a = self.linear_proj(semantics_a).reshape((semantics_a.shape[0],semantics_a.shape[1],self.hidden_dim,2))
                norm_a = torch.sqrt(torch.sum(semantics_a**2, dim=-1)).unsqueeze(-1).repeat([1,1,1,semantics_a.shape[-1]])
                semantics_a = torch.div(semantics_a,norm_a)
            
            semantics_b = self.semanticE(words_b, chars_b, mask_t)
            if len(semantics_b.shape)==2:
                semantics_b = self.linear_proj(semantics_b).reshape((semantics_b.shape[0],self.hidden_dim,2))
                norm_b = torch.sqrt(torch.sum(semantics_b**2, dim=-1)).unsqueeze(-1).repeat([1,1,semantics_b.shape[-1]])
                semantics_b = torch.div(semantics_b,norm_b)
                semantics_b = semantics_b.unsqueeze(dim=1)
            else:
                semantics_b = self.linear_proj(semantics_b).reshape((semantics_b.shape[0],semantics_b.shape[1],self.hidden_dim,2))
                norm_b = torch.sqrt(torch.sum(semantics_b**2, dim=-1)).unsqueeze(-1).repeat([1,1,1,semantics_b.shape[-1]])
                semantics_b = torch.div(semantics_b,norm_b)            

            # structural_score = self._calc_structural_score(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
            # if self.semantic_score_fn == 'pooled':
            #     semantic_score = self._calc_pooled_semantic_score_optim(semantics_a, semantics_b, Rs, H, T, set_sizes_h, set_sizes_t)
            # else:    
            #     semantic_score = self._calc_semantic_score_optim(semantics_a, semantics_b, Rs, H, T, set_sizes_h, set_sizes_t)
            if self.CUDA:
                H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1).cuda(),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
                T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1).cuda(),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
            else:
                H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
                T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
            # delta_w, delta_x, delta_y, delta_z = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            score1 = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            
            delta_x = (compute_tail_x - tail_x)
            # delta_y = (compute_tail_y - tail_y)
            # delta_z = (compute_tail_z - tail_z)
            
            # score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
            # score1 = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
            # score1 = score1.norm(dim = 0)
            
            x = -x
            y = -y
            z = -z
            # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
            # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
            # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
            _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z)

            if self.relation_embedding_has_mod:
                compute_head_x = compute_head_x / denominator
                compute_head_y = compute_head_y / denominator
                compute_head_z = compute_head_z / denominator
            
            if self.CUDA:
                T = torch.cat((torch.zeros(compute_head_x.shape).cuda().unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
                H = torch.cat((torch.zeros(head_x.shape).cuda().unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
            else:
                T = torch.cat((torch.zeros(compute_head_x.shape).unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
                H = torch.cat((torch.zeros(head_x.shape).unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
            # delta_w2, delta_x2, delta_y2, delta_z2 = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            score2 = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)

            # delta_x2 = (compute_head_x - head_x)
            # delta_y2 = (compute_head_y - head_y)
            # delta_z2 = (compute_head_z - head_z)
            
            # score2 = torch.stack([delta_w2, delta_x2, delta_y2, delta_z2], dim = 0)
            # score2 = score2.norm(dim = 0)     
            
            # score1 = score1.mean(dim=2)
            # score2 = score2.mean(dim=2)

            # score1 = score1.sum(dim=2)
            # score2 = score2.sum(dim=2)
        elif self.method_to_induce_semantics=='mult':
            # compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
            # compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
            # compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
            _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z)

            if self.relation_embedding_has_mod:
                compute_tail_x = denominator * compute_tail_x
                compute_tail_y = denominator * compute_tail_y
                compute_tail_z = denominator * compute_tail_z

            words_a, chars_a, mask_h, _ = self.get_entity_properties_from_npy(self.batch_h, pos='head')
            words_b, chars_b, mask_t, _ = self.get_entity_properties_from_npy(self.batch_t, pos='tail')
            
            if self.CUDA:
                words_a, chars_a, mask_h = torch.from_numpy(words_a).cuda(), torch.from_numpy(chars_a).cuda(), torch.from_numpy(mask_h).cuda()
                words_b, chars_b, mask_t = torch.from_numpy(words_b).cuda(), torch.from_numpy(chars_b).cuda(), torch.from_numpy(mask_t).cuda()
            else:
                words_a, chars_a, mask_h = torch.from_numpy(words_a), torch.from_numpy(chars_a), torch.from_numpy(mask_h)
                words_b, chars_b, mask_t = torch.from_numpy(words_b), torch.from_numpy(chars_b), torch.from_numpy(mask_t)
            semantics_a = self.semanticE(words_a, chars_a, mask_h)
            if len(semantics_a.shape)==2:        
                semantics_a = self.linear_proj2(semantics_a).reshape((semantics_a.shape[0],self.hidden_dim,4))
                norm_a = torch.sqrt(torch.sum(semantics_a**2, dim=-1)).unsqueeze(-1).repeat([1,1,semantics_a.shape[-1]])
                semantics_a = torch.div(semantics_a,norm_a)
                semantics_a = semantics_a.unsqueeze(dim=1)
            else:
                semantics_a = self.linear_proj2(semantics_a).reshape((semantics_a.shape[0],semantics_a.shape[1],self.hidden_dim,4))
                norm_a = torch.sqrt(torch.sum(semantics_a**2, dim=-1)).unsqueeze(-1).repeat([1,1,1,semantics_a.shape[-1]])
                semantics_a = torch.div(semantics_a,norm_a)
            
            semantics_b = self.semanticE(words_b, chars_b, mask_t)
            if len(semantics_b.shape)==2:
                semantics_b = self.linear_proj2(semantics_b).reshape((semantics_b.shape[0],self.hidden_dim,4))
                norm_b = torch.sqrt(torch.sum(semantics_b**2, dim=-1)).unsqueeze(-1).repeat([1,1,semantics_b.shape[-1]])
                semantics_b = torch.div(semantics_b,norm_b)
                semantics_b = semantics_b.unsqueeze(dim=1)
            else:
                semantics_b = self.linear_proj2(semantics_b).reshape((semantics_b.shape[0],semantics_b.shape[1],self.hidden_dim,4))
                norm_b = torch.sqrt(torch.sum(semantics_b**2, dim=-1)).unsqueeze(-1).repeat([1,1,1,semantics_b.shape[-1]])
                semantics_b = torch.div(semantics_b,norm_b)            

            param_s_a = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_s_a = param_s_a.permute([0,2,1]).reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[2]//self.num_heads, param_s_a.shape[1])).permute([0,2,1])
            else:
                param_s_a = param_s_a.reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[1]//self.num_heads))
            param_x_a = self.param_ent_x_a(self.batch_h)    
            if len(self.batch_h.shape) == 2:
                param_x_a = param_x_a.permute([0,2,1]).reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[2]//self.num_heads, param_x_a.shape[1])).permute([0,2,1])
            else:
                param_x_a = param_x_a.reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[1]//self.num_heads))
            norm_a = torch.sqrt(param_s_a**2 + param_x_a**2)
            param_s_a = param_s_a / norm_a
            param_x_a = param_x_a / norm_a
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)

            param_s_c = self.param_ent_s_a(self.batch_t)
            if len(self.batch_t.shape) == 2:
                param_s_c = param_s_c.permute([0,2,1]).reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[2]//self.num_heads, param_s_c.shape[1])).permute([0,2,1])
            else:
                param_s_c = param_s_c.reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[1]//self.num_heads))        
            param_x_c = self.param_ent_x_a(self.batch_t)    
            if len(self.batch_t.shape) == 2:
                param_x_c = param_x_c.permute([0,2,1]).reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[2]//self.num_heads, param_x_c.shape[1])).permute([0,2,1])
            else:
                param_x_c = param_x_c.reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[1]//self.num_heads))        
            norm_c = torch.sqrt(param_s_c**2 + param_x_c**2)
            param_s_c = param_s_c / norm_c
            param_x_c = param_x_c / norm_c
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)           
            if len(self.batch_t.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)

            # structural_score = self._calc_structural_score(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
            # if self.semantic_score_fn == 'pooled':
            #     semantic_score = self._calc_pooled_semantic_score_optim(semantics_a, semantics_b, Rs, H, T, set_sizes_h, set_sizes_t)
            # else:    
            #     semantic_score = self._calc_semantic_score_optim(semantics_a, semantics_b, Rs, H, T, set_sizes_h, set_sizes_t)
            if self.CUDA:
                H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1).cuda(),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
                T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1).cuda(),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
            else:
                H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
                T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
            # delta_w, delta_x, delta_y, delta_z = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            score1 = self._calc_semantic_score_optim_after_map(param_a, param_b, semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            
            delta_x = (compute_tail_x - tail_x)
            # delta_y = (compute_tail_y - tail_y)
            # delta_z = (compute_tail_z - tail_z)
            
            # score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
            # score1 = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
            # score1 = score1.norm(dim = 0)
            
            x = -x
            y = -y
            z = -z
            # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
            # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
            # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
            _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z)

            if self.relation_embedding_has_mod:
                compute_head_x = compute_head_x / denominator
                compute_head_y = compute_head_y / denominator
                compute_head_z = compute_head_z / denominator
            
            if self.CUDA:
                T = torch.cat((torch.zeros(compute_head_x.shape).cuda().unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
                H = torch.cat((torch.zeros(head_x.shape).cuda().unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
            else:
                T = torch.cat((torch.zeros(compute_head_x.shape).unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
                H = torch.cat((torch.zeros(head_x.shape).unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
            # delta_w2, delta_x2, delta_y2, delta_z2 = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            score2 = self._calc_semantic_score_optim_after_map(param_a, param_b, semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
        elif self.method_to_induce_semantics=='mlp':
            assert self.avg_semantics==True, "Semantics must be averaged for mlp option"

            words_a, chars_a, mask_h, _ = self.get_entity_properties_from_npy(self.batch_h, pos='head')
            words_b, chars_b, mask_t, _ = self.get_entity_properties_from_npy(self.batch_t, pos='tail')
            
            if self.CUDA:
                words_a, chars_a, mask_h = torch.from_numpy(words_a).cuda(), torch.from_numpy(chars_a).cuda(), torch.from_numpy(mask_h).cuda()
                words_b, chars_b, mask_t = torch.from_numpy(words_b).cuda(), torch.from_numpy(chars_b).cuda(), torch.from_numpy(mask_t).cuda()
            else:
                words_a, chars_a, mask_h = torch.from_numpy(words_a), torch.from_numpy(chars_a), torch.from_numpy(mask_h)
                words_b, chars_b, mask_t = torch.from_numpy(words_b), torch.from_numpy(chars_b), torch.from_numpy(mask_t)
            Hs2 = self.semanticE(words_a, chars_a, mask_h)
            Ts2 = self.semanticE(words_b, chars_b, mask_t)

            Hs2 = torch.mean(Hs2.reshape(Hs2.shape[0]//self.PADDING,self.PADDING,Hs2.shape[1],Hs2.shape[2]), dim=1)
            Ts2 = torch.mean(Ts2.reshape(Ts2.shape[0]//self.PADDING,self.PADDING,Ts2.shape[1],Ts2.shape[2]), dim=1)

            H = self.linear_proj_mlp(torch.cat((H.reshape([H.shape[0],H.shape[1],H.shape[2]*H.shape[3]]),Hs2), dim=-1)).reshape((H.shape[0],H.shape[1],self.hidden_dim,3))
            T = self.linear_proj_mlp(torch.cat((T.reshape([T.shape[0],T.shape[1],T.shape[2]*T.shape[3]]),Ts2), dim=-1)).reshape((T.shape[0],T.shape[1],self.hidden_dim,3))

            head_x = H[:,:,:,0]
            head_y = H[:,:,:,1]
            head_z = H[:,:,:,2]
            tail_x = T[:,:,:,0]
            tail_y = T[:,:,:,1]
            tail_z = T[:,:,:,2]

            _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z)

            if self.relation_embedding_has_mod:
                compute_tail_x = denominator * compute_tail_x
                compute_tail_y = denominator * compute_tail_y
                compute_tail_z = denominator * compute_tail_z
          

            param_s_a = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_s_a = param_s_a.permute([0,2,1]).reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[2]//self.num_heads, param_s_a.shape[1])).permute([0,2,1])
            else:
                param_s_a = param_s_a.reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[1]//self.num_heads))
            param_x_a = self.param_ent_x_a(self.batch_h)    
            if len(self.batch_h.shape) == 2:
                param_x_a = param_x_a.permute([0,2,1]).reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[2]//self.num_heads, param_x_a.shape[1])).permute([0,2,1])
            else:
                param_x_a = param_x_a.reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[1]//self.num_heads))
            norm_a = torch.sqrt(param_s_a**2 + param_x_a**2)
            param_s_a = param_s_a / norm_a
            param_x_a = param_x_a / norm_a
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)

            param_s_c = self.param_ent_s_a(self.batch_t)
            if len(self.batch_t.shape) == 2:
                param_s_c = param_s_c.permute([0,2,1]).reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[2]//self.num_heads, param_s_c.shape[1])).permute([0,2,1])
            else:
                param_s_c = param_s_c.reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[1]//self.num_heads))        
            param_x_c = self.param_ent_x_a(self.batch_t)    
            if len(self.batch_t.shape) == 2:
                param_x_c = param_x_c.permute([0,2,1]).reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[2]//self.num_heads, param_x_c.shape[1])).permute([0,2,1])
            else:
                param_x_c = param_x_c.reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[1]//self.num_heads))        
            norm_c = torch.sqrt(param_s_c**2 + param_x_c**2)
            param_s_c = param_s_c / norm_c
            param_x_c = param_x_c / norm_c
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)           
            if len(self.batch_t.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)

            # structural_score = self._calc_structural_score(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
            # if self.semantic_score_fn == 'pooled':
            #     semantic_score = self._calc_pooled_semantic_score_optim(semantics_a, semantics_b, Rs, H, T, set_sizes_h, set_sizes_t)
            # else:    
            #     semantic_score = self._calc_semantic_score_optim(semantics_a, semantics_b, Rs, H, T, set_sizes_h, set_sizes_t)
            if self.CUDA:
                H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1).cuda(),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
                T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1).cuda(),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
            else:
                H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
                T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
            # delta_w, delta_x, delta_y, delta_z = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            score1 = self._calc_semantic_score_optim_mlp(param_a, param_b, semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            
            delta_x = (compute_tail_x - tail_x)
            # delta_y = (compute_tail_y - tail_y)
            # delta_z = (compute_tail_z - tail_z)
            
            # score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
            # score1 = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
            # score1 = score1.norm(dim = 0)
            
            x = -x
            y = -y
            z = -z
            # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
            # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
            # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
            _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z)

            if self.relation_embedding_has_mod:
                compute_head_x = compute_head_x / denominator
                compute_head_y = compute_head_y / denominator
                compute_head_z = compute_head_z / denominator
            
            if self.CUDA:
                T = torch.cat((torch.zeros(compute_head_x.shape).cuda().unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
                H = torch.cat((torch.zeros(head_x.shape).cuda().unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
            else:
                T = torch.cat((torch.zeros(compute_head_x.shape).unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
                H = torch.cat((torch.zeros(head_x.shape).unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
            # delta_w2, delta_x2, delta_y2, delta_z2 = self._calc_semantic_score_optim(semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)
            score2 = self._calc_semantic_score_optim_after_map(param_a, param_b, semantics_a, semantics_b, H, T, set_sizes_h, set_sizes_t)            
        elif self.method_to_induce_semantics=='init':

            words_a, chars_a, mask_h, set_sizes_h = self.get_entity_properties_from_npy(self.batch_h, pos='head')
            words_b, chars_b, mask_t, set_sizes_t = self.get_entity_properties_from_npy(self.batch_t, pos='tail')
            
            if self.CUDA:
                words_a, chars_a, mask_h, set_sizes_h = torch.from_numpy(words_a).cuda(), torch.from_numpy(chars_a).cuda(), torch.from_numpy(mask_h).cuda(), torch.from_numpy(set_sizes_h).cuda()
                words_b, chars_b, mask_t, set_sizes_t = torch.from_numpy(words_b).cuda(), torch.from_numpy(chars_b).cuda(), torch.from_numpy(mask_t).cuda(), torch.from_numpy(set_sizes_t).cuda()
            else:
                words_a, chars_a, mask_h, set_sizes_h = torch.from_numpy(words_a), torch.from_numpy(chars_a), torch.from_numpy(mask_h), torch.from_numpy(set_sizes_h)
                words_b, chars_b, mask_t, set_sizes_t = torch.from_numpy(words_b), torch.from_numpy(chars_b), torch.from_numpy(mask_t), torch.from_numpy(set_sizes_t)
            semantics_a = self.semanticE(words_a, chars_a, mask_h)
            if len(semantics_a.shape)==2:        
                semantics_a = self.linear_proj_init(semantics_a).reshape((semantics_a.shape[0],self.hidden_dim,3))
                semantics_a = semantics_a.unsqueeze(dim=1)
            else:
                semantics_a = self.linear_proj_init(semantics_a).reshape((semantics_a.shape[0],semantics_a.shape[1],self.hidden_dim,3))
            
            semantics_b = self.semanticE(words_b, chars_b, mask_t)
            if len(semantics_b.shape)==2:
                semantics_b = self.linear_proj_init(semantics_b).reshape((semantics_b.shape[0],self.hidden_dim,3))
                semantics_b = semantics_b.unsqueeze(dim=1)
            else:
                semantics_b = self.linear_proj_init(semantics_b).reshape((semantics_b.shape[0],semantics_b.shape[1],self.hidden_dim,3))

            head_x = semantics_a[:,:,:,0]
            head_y = semantics_a[:,:,:,1]
            head_z = semantics_a[:,:,:,2]
            tail_x = semantics_b[:,:,:,0]
            tail_y = semantics_b[:,:,:,1]
            tail_z = semantics_b[:,:,:,2]
            _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z)
            if self.relation_embedding_has_mod:
                compute_tail_x = denominator * compute_tail_x
                compute_tail_y = denominator * compute_tail_y
                compute_tail_z = denominator * compute_tail_z

            delta_x = (compute_tail_x - tail_x)
            delta_y = (compute_tail_y - tail_y)
            delta_z = (compute_tail_z - tail_z)
            
            score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
            score1 = score1.norm(dim = 0)    

            x = -x
            y = -y
            z = -z
            # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
            # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
            # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
            _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z)

            if self.relation_embedding_has_mod:
                compute_head_x = compute_head_x / denominator
                compute_head_y = compute_head_y / denominator
                compute_head_z = compute_head_z / denominator      
    
            delta_x = (compute_head_x - head_x)
            delta_y = (compute_head_y - head_y)
            delta_z = (compute_head_z - head_z)
            
            score2 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
            score2 = score2.norm(dim = 0)

            score1 = score1.mean(dim=2)
            score2 = score2.mean(dim=2)      

        score = (score1 + score2) / 2
        
        score = self.gamma.item() - score
            
        return score, score1, score2, torch.abs(delta_x)

    @staticmethod
    def train_step(model, optimizer, train_iterator, step, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode=mode) # 全是负样本分数 shape: batch_size, neg_size
        
        if step % 500 == 0:
            print(negative_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, head_mod, tail_mod, rel_mod = model(positive_sample) # 正样本分数 shape: batch_size, 1     

        if step % 500 == 0:
            print(positive_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_x.weight.data.norm(p = 3)**3 + 
                model.entity_y.weight.data.norm(p = 3)**3 + 
                model.entity_z.weight.data.norm(p = 3)**3 
            ) / args.batch_size

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        if args.test_in_parts:
                            num_parts = math.ceil(negative_sample.shape[1]/args.sub_batch_size)
                            score, head_mod, tail_mod, rel_mod =  torch.empty((1,0)), torch.empty((1,0)), torch.empty((1,0)), torch.empty((1,0,args.hidden_dim))
                            if args.cuda:
                                score, head_mod, tail_mod, rel_mod = score.cuda(), head_mod.cuda(), tail_mod.cuda(), rel_mod.cuda()
                            for part in range(num_parts):
                                start_idx = args.sub_batch_size*part
                                end_idx =  min(args.sub_batch_size*(1+part),negative_sample.shape[1])
                                cur_score, cur_head_mod, cur_tail_mod, cur_rel_mod = model((positive_sample, negative_sample[:,start_idx:end_idx]), mode)
                                score, head_mod, rel_mod, tail_mod = torch.cat((score,cur_score), dim=1), torch.cat((head_mod,cur_head_mod), dim=1), torch.cat((rel_mod,cur_rel_mod), dim=1), torch.cat((tail_mod,cur_tail_mod), dim=1)

                        else:
                            score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
    def get_entity_properties_from_npy(self, batch_indices, pos='head'):
        # pass
        n = batch_indices.shape[0]
        if len(batch_indices.shape)==1:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_h[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_t[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
        else:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
    

class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds

class RnnE(nn.Module):
    def __init__(self, params):
        super(RnnE, self).__init__()
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.layers = params['layers']
        self.is_bidirectional = params['is_bidirectional']
        self.drop_rate = params['drop_out_rate']
        embeddings = params['embeddings']
        char_vocab = params['char_vocab']

        # self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, self.drop_rate)
        # self.word_embeddings = word_embeddings
        self.word_embeddings = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        # self.word_embeddings.weight.requires_grad = False
        self.word_embeddings.weight.requires_grad = True
        self.char_embeddings = CharEmbeddings(len(char_vocab), params['char_embed_dim'], self.drop_rate)
        self.lstm = nn.LSTM(embeddings.shape[1]+params['char_feature_size'], self.hidden_dim, self.layers, batch_first=True,
          bidirectional=bool(self.is_bidirectional))

        self.conv1d = nn.Conv1d(params['char_embed_dim'], params['char_feature_size'], params['conv_filter_size'],padding=0)
        self.max_pool = nn.MaxPool1d(params['max_word_len_entity'] + params['conv_filter_size'] - 1, params['max_word_len_entity'] + params['conv_filter_size'] - 1)

    def forward(self, words, chars):
        batch_size = words.shape[0]
        if len(words.shape)==3:
          # max_batch_len = words.shape[1]
          words = words.view(words.shape[0]*words.shape[1],words.shape[2])
          chars = chars.view(chars.shape[0]*chars.shape[1],chars.shape[2])

        src_word_embeds = self.word_embeddings(words)
        char_embeds = self.char_embeddings(chars)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((src_word_embeds, char_feature), -1)
        outputs, hc = self.lstm(words_input)

        # h_drop = self.dropout(hc[0])
        h_n = hc[0].view(self.layers, 2, words.shape[0], self.hidden_dim)
        h_n = h_n[-1,:,:,:].squeeze() # (num_dir,batch,hidden)
        h_n = h_n.permute((1,0,2)) # (batch,num_dir,hidden)
        h_n = h_n.contiguous().view(h_n.shape[0],h_n.shape[1]*h_n.shape[2]) # (batch,num_dir*hidden)
        #h_n_batch = h_n.view(batch_size,max_batch_len,h_n.shape[1])
        
        return h_n


class CbowE(nn.Module):
    def __init__(self, params):
        super(CbowE, self).__init__()
        embeddings = params['embeddings']
        char_vocab = params['char_vocab']

        self.word_embeddings = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embeddings.weight.requires_grad = True
        self.char_embeddings = CharEmbeddings(len(char_vocab), params['char_embed_dim'], params['drop_rate'])
        self.conv1d = nn.Conv1d(params['char_embed_dim'], params['char_feature_size'], params['conv_filter_size'], padding=0)
        self.max_pool = nn.MaxPool1d(params['max_word_len_entity'] + params['conv_filter_size'] - 1, params['max_word_len_entity'] + params['conv_filter_size'] - 1)

    def forward(self, words, chars, mask):
        batch_size = words.shape[0]

        single_mode = True
        if len(words.shape)==3:
          single_mode = False
          neg_samples = words.shape[1]
          words = words.view(words.shape[0]*words.shape[1],words.shape[2])
          chars = chars.view(chars.shape[0]*chars.shape[1],chars.shape[2])

        src_word_embeds = self.word_embeddings(words)

        # TODO: comment below block if char embeddings not desired and change the input dimensions of linear_proj to 100
        char_embeds = self.char_embeddings(chars)
        char_embeds = char_embeds.permute(0, 2, 1)
        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)
        words_input = torch.cat((src_word_embeds, char_feature), -1)
        # TODO: uncomment below line if char embeddings not desired
        # words_input = src_word_embeds
        
        if single_mode:
            m1 = mask.unsqueeze(-1).repeat([1,1,words_input.shape[-1]])
            m2 = torch.sum(mask, dim=1).unsqueeze(-1).repeat([1,words_input.shape[-1]])
            avg_emb = torch.div(torch.sum(words_input*m1, dim=1),m2)
        else:
            words_input = words_input.unsqueeze(dim=1).reshape([mask.shape[0],mask.shape[1],words.shape[-1],words_input.shape[-1]])
            m1 = mask.unsqueeze(-1).repeat([1,1,1,words_input.shape[-1]])
            m2 = torch.sum(mask, dim=2).unsqueeze(-1).repeat([1,1,words_input.shape[-1]])
            avg_emb = torch.div(torch.sum(words_input*m1, dim=2),m2)

        return avg_emb

class CbowCharE(nn.Module):
    def __init__(self, params):
        super(CbowCharE, self).__init__()
        embeddings = params['embeddings']
        char_vocab = params['char_vocab']

        self.word_embeddings = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embeddings.weight.requires_grad = True
        self.char_embeddings = CharEmbeddings(len(char_vocab), params['char_embed_dim'], params['drop_rate'])
        self.conv1d = nn.Conv1d(params['char_embed_dim'], params['char_feature_size'], params['conv_filter_size'], padding=0)
        self.max_pool = nn.MaxPool1d(params['max_word_len_entity'] + params['conv_filter_size'] - 1, params['max_word_len_entity'] + params['conv_filter_size'] - 1)

    def forward(self, words, chars, mask):
        batch_size = words.shape[0]
        if len(words.shape)==3:
          words = words.view(words.shape[0]*words.shape[1],words.shape[2])
          chars = chars.view(chars.shape[0]*chars.shape[1],chars.shape[2])

        src_word_embeds = self.word_embeddings(words)
        char_embeds = self.char_embeddings(chars)
        # char_embeds = char_embeds.permute(0, 2, 1)
        # char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        # char_feature = char_feature.permute(0, 2, 1)
        # words_input = torch.cat((src_word_embeds, char_feature), -1)
        char_feature = torch.mean(char_embeds, dim=2)
        words_input = torch.cat((src_word_embeds, char_feature), -1)

        m1 = mask.unsqueeze(-1).repeat([1,1,words_input.shape[-1]])
        m2 = torch.sum(mask, dim=1).unsqueeze(-1).repeat([1,words_input.shape[-1]])
        avg_emb = torch.div(torch.sum(words_input*m1, dim=1),m2)

        return avg_emb




class HopfParaEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 entity_embedding_has_mod=False, relation_embedding_has_mod=False, rotate_fn="R1", params=None):
        super(HopfParaEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 1.2
        self.rel_high_bound = 2.0
        
        self.use_abs_norm = True
        self.allow_minus_mod = True
        self.use_entity_phase = False
        self.use_real_part = False
        
        self.criterion = 'he'
        
        if self.criterion == 'glorot':
            mod_range = 1. / np.sqrt(2 * (self.hidden_dim + self.hidden_dim))
        elif self.criterion == 'he':
            mod_range = 1. / np.sqrt(2 * self.hidden_dim)
        
        if self.allow_minus_mod:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range * 2.]), 
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range]), 
                requires_grad=False
            )
            
        self.gamma1 = nn.Parameter(
            torch.Tensor([(self.rel_high_bound + self.epsilon) * mod_range * self.hidden_dim]), 
            requires_grad=False
        )
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.unit_mod = nn.Parameter(
            torch.Tensor([1.]), 
            requires_grad=False
        )
        
        self.zero_ent_phase = nn.Parameter(
            torch.Tensor([0.]), 
            requires_grad=False
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.entity_embedding_has_mod = entity_embedding_has_mod
        self.relation_embedding_has_mod = relation_embedding_has_mod
                   
        self.entity_x = nn.Embedding(self.nentity, self.hidden_dim)
        self.entity_y = nn.Embedding(self.nentity, self.hidden_dim)   
        self.entity_z = nn.Embedding(self.nentity, self.hidden_dim)
        
        self.relation_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_z = nn.Embedding(self.nrelation, self.hidden_dim)
        
        self.init_weights()
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['HopfParaE']:
            raise ValueError('model %s not supported' % model_name)
        if self.use_real_part:
            try:
                assert(self.use_abs_norm == True)
            except:
                raise ValueError('use_abs_norm should be true if you only use real part')
        if (not self.entity_embedding_has_mod) and self.relation_embedding_has_mod:
            raise ValueError('when relation has mod, entity must have mod')
        
        if params is None:
            params={    
                'embeddings_path': '../../data/FB15K237_context/limit1_v3/embeddings.npy',
                'char_vocab_path': '../../data/FB15K237_context/limit1_v3/char2idx.json',
                'char_feature_size': 50,
                'char_embed_dim': 50,
                'max_word_len_entity': 10,
                'conv_filter_size': 3,
                'drop_rate': 0.0,
                'max_sent_len': 16,
                'entity_indices_file': '../../data/FB15K237_context/limit1_v3/entity_context_indices.json',
                'word2idx_path': '../../data/FB15K237_context/limit1_v3/word2idx.json',
                'all_word_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/word_indices_h.npy',
                'all_char_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/char_indices_h.npy',
                'mask_file_h': '../../data/FB15K237_context/limit1_v3/mask_h.npy',
                'all_word_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/word_indices_t.npy',
                'all_char_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/char_indices_t.npy',
                'mask_file_t': '../../data/FB15K237_context/limit1_v3/mask_t.npy',
                'padding': 1,
                'checkpoint_json_path': './result_hopfe_rot_2/HopfE.json',
                'num_heads': 1,
                'wassertein_approx': False
            }
        self.embeddings = np.load(params['embeddings_path'])
        with open(params['char_vocab_path'], 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.all_word_token_ids_h = np.load(params['all_word_token_ids_file_h'])
        self.all_char_token_ids_h = np.load(params['all_char_token_ids_file_h'])
        self.all_word_token_ids_t = np.load(params['all_word_token_ids_file_t'])
        self.all_char_token_ids_t = np.load(params['all_char_token_ids_file_t'])
        self.PADDING = params['padding']
        self.max_sent_len = params['max_sent_len']
        self.conv_filter_size = params['conv_filter_size']
        self.max_char_len = params['max_word_len_entity']
        self.char_embed_dim = params['char_embed_dim']
        self.drop_rate = params['drop_rate']
        self.mask_h = np.load(params['mask_file_h'])
        self.mask_t = np.load(params['mask_file_t'])
        self.char_feature_size = params['char_feature_size']

        self.semanticE = CbowE({'embeddings': self.embeddings, 
            'char_vocab': self.char_vocab, 
            'char_feature_size': self.char_feature_size, 
            'char_embed_dim': self.char_embed_dim, 
            'max_word_len_entity': self.max_char_len, 
            'conv_filter_size': self.conv_filter_size, 
            'drop_rate': self.drop_rate})
        self.linear_proj = nn.Linear(50, 2*self.hidden_dim)


        self.num_heads = params['num_heads']
        self.param_ent_s_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)
        self.param_ent_x_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)
        self.CUDA = torch.cuda.is_available()

        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
        self.wassertein_approx = params.get('wassertein_approx', False)
        self.single_param = False

        self.rotate_fn = rotate_fn

    def init_weights(self):

        rel_w, rel_x, rel_y, rel_z = self.relation_init(self.nrelation, self.hidden_dim)
        rel_w, rel_x, rel_y, rel_z = torch.from_numpy(rel_w), torch.from_numpy(rel_x), torch.from_numpy(rel_y), torch.from_numpy(rel_z)
        self.relation_w.weight.data = rel_w.type_as(self.relation_w.weight.data)
        self.relation_x.weight.data = rel_x.type_as(self.relation_x.weight.data)
        self.relation_y.weight.data = rel_y.type_as(self.relation_y.weight.data)
        self.relation_z.weight.data = rel_z.type_as(self.relation_z.weight.data)
        
        e_x, e_y, e_z = self.entity_init(self.nentity, self.hidden_dim)
        e_x, e_y, e_z = torch.from_numpy(e_x), torch.from_numpy(e_y), torch.from_numpy(e_z)
        self.entity_x.weight.data = e_x.type_as(self.entity_x.weight.data)
        self.entity_y.weight.data = e_y.type_as(self.entity_y.weight.data)
        self.entity_z.weight.data = e_z.type_as(self.entity_z.weight.data)
        
    def relation_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features
        
        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init rel_mod is: ', s)

        kernel_shape = (n_entries, features)
            
        rel_mod = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rotate_phase = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=kernel_shape)
        theta = np.random.uniform(low=0, high=np.pi, size=kernel_shape)
        phi = np.random.uniform(low=0, high=2*np.pi, size=kernel_shape)
        
        rel_w = rel_mod * np.cos(rotate_phase/2)
        rel_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
        rel_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
        rel_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

        return rel_w, rel_x, rel_y, rel_z

    def entity_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init x, y, z is: ', s)

        # rng = RandomState(456)
        kernel_shape = (n_entries, features)
            
        x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return x, y, z
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # batch_size, negative_sample_size = sample.size(0), 1
            
            head_x = self.entity_x(sample[:, 0]).unsqueeze(1)
            head_y = self.entity_y(sample[:, 0]).unsqueeze(1)
            head_z = self.entity_z(sample[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(sample[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(sample[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(sample[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(sample[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(sample[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(sample[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(sample[:, 1]).unsqueeze(1)

            self.batch_h = sample[:, 0]
            self.batch_r = sample[:, 1]
            self.batch_t = sample[:, 2]
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_x = self.entity_x(head_part)
            head_y = self.entity_y(head_part)
            head_z = self.entity_z(head_part)
            
            tail_x = self.entity_x(tail_part[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(tail_part[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(tail_part[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(tail_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(tail_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(tail_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(tail_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part
            self.batch_r = tail_part[:, 1]
            self.batch_t = tail_part[:, 2]
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_x = self.entity_x(head_part[:, 0]).unsqueeze(1)
            head_y = self.entity_y(head_part[:, 0]).unsqueeze(1)
            head_z = self.entity_z(head_part[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(tail_part)
            tail_y = self.entity_y(tail_part)
            tail_z = self.entity_z(tail_part)
            
            rel_w = self.relation_w(head_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(head_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(head_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(head_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part[:, 0]
            self.batch_r = head_part[:, 1]
            self.batch_t = tail_part
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'HopfParaE': self.HopfParaE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_x, head_y, head_z, 
                                                rel_w, rel_x, rel_y, rel_z, 
                                                tail_x, tail_y, tail_z, 
                                                mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def _quat_mul(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        
        return (A, B, C, D)

    def rotate(self, x, y, z, rel_w, rel_x, rel_y, rel_z):
        A, B, C, D = self._quat_mul(rel_w, rel_x, rel_y, rel_z, 0, x, y, z)
        return self._quat_mul(A, B, C, D, rel_w, -1.0*rel_x, -1.0*rel_y, -1.0*rel_z)

    def _calc_semantic_score_optim_test(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t):
        
        Ah = H[:,:,:,0]
        Bh = H[:,:,:,1]
        Ch = H[:,:,:,2]
        Dh = H[:,:,:,3]

        At = T[:,:,:,0]
        Bt = T[:,:,:,1]
        Ct = T[:,:,:,2]
        Dt = T[:,:,:,3]

        if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
            Ah = Ah.repeat([1,At.shape[1],1])
            Bh = Bh.repeat([1,Bt.shape[1],1])
            Ch = Ch.repeat([1,Ct.shape[1],1])
            Dh = Dh.repeat([1,Dt.shape[1],1])
        elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
            At = At.repeat([1,Ah.shape[1],1])
            Bt = Bt.repeat([1,Bh.shape[1],1])
            Ct = Ct.repeat([1,Ch.shape[1],1])
            Dt = Dt.repeat([1,Dh.shape[1],1])

        # Reshape Ah, Bh, Ch, Dh
        Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2]).contiguous()
        Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2]).contiguous()
        Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2]).contiguous()
        Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2]).contiguous()

        Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2]).contiguous()
        Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2]).contiguous()
        Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2]).contiguous()
        Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2]).contiguous()
        
        # find the pairwise dist
        # dw = torch.cdist(Ahr,Atr).reshape(Ah.shape[0]//self.num_heads,Ah.shape[1],self.num_heads,self.num_heads)
        # dx = torch.cdist(Bhr,Btr).reshape(Bh.shape[0]//self.num_heads,Bh.shape[1],self.num_heads,self.num_heads)
        # dy = torch.cdist(Chr,Ctr).reshape(Ch.shape[0]//self.num_heads,Ch.shape[1],self.num_heads,self.num_heads)
        # dz = torch.cdist(Dhr,Dtr).reshape(Dh.shape[0]//self.num_heads,Dh.shape[1],self.num_heads,self.num_heads)
        d1 = Ahr-Atr
        d2 = Bhr-Btr
        d3 = Chr-Ctr
        d4 = Dhr-Dtr
        d5 = torch.stack([d1,d2,d3,d4],dim=0)
        d6 = d5.norm(dim=0)
        d7 = torch.mean(d6,dim=-1)
        return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

        # # take the sum
        # delta = dw + dx + dy + dz

        # # take the min, min or max, min
        # delta_min1 = torch.min(delta, dim=-1)[0]
        # delta_min2 = torch.min(delta_min1, dim=-1)[0]     


        # return delta_min2

    def _calc_semantic_score_optim(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t, rotate_fn="R1"):

        # single_mode = True
        # if len(Hs.shape)==4:
        #     single_mode = False
        #     neg_samples = Hs.shape[1]
        #     Hs = Hs.reshape((-1,Hs.shape[2],Hs.shape[3]))
        #     set_sizes_h = set_sizes_h * neg_samples
        # if len(Ts.shape)==4:
        #     single_mode = False
        #     neg_samples = Ts.shape[1]
        #     Ts = Ts.reshape((-1,Ts.shape[2],Ts.shape[3]))
        #     set_sizes_t = set_sizes_t * neg_samples

        # s_b = Rs[:,:,0]
        # x_b = Rs[:,:,1]
        # y_b = Rs[:,:,2]
        # z_b = Rs[:,:,3]
        # denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        # s_b = s_b / denominator_b
        # x_b = x_b / denominator_b
        # y_b = y_b / denominator_b
        # z_b = z_b / denominator_b

        H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        
        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2":
            denominator_a = torch.sqrt(s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
            s_a = s_a / denominator_a
            x_a = x_a / denominator_a
            y_a = y_a / denominator_a
            z_a = z_a / denominator_a

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2": 
            denominator_c = torch.sqrt(s_c ** 2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
            s_c = s_c / denominator_c
            x_c = x_c / denominator_c
            y_c = y_c / denominator_c
            z_c = z_c / denominator_c

        # Rs0 = torch.cat((s_b.unsqueeze(-1),x_b.unsqueeze(-1),y_b.unsqueeze(-1),z_b.unsqueeze(-1)), dim=-1)

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        # A, B, C, D = self._quat_mul(H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # A, B, C, D = self._quat_mul(H[:,:,0], 1+H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # num = torch.sqrt(H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        # den = torch.sqrt(1+2*H[:,:,1] + H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)

        assert s_a.eq(0).all().cpu().numpy(), "s_a must be equal to 0"
        assert (x_a**2 + y_a**2 + z_a**2).eq(1).all().cpu().numpy(), "(x_a**2 + y_a**2 + z_a**2) must be equal to 1"

        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_a/(1+x_a+epsilon_xa), y_a/(1+x_a+epsilon_xa), Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2) 
            den = np.sqrt(2.)      
        else:
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)         
        norm = torch.div(num,den)
        # Ah = A*norm
        # Bh = B*norm
        # Ch = C*norm
        # Dh = D*norm
        if rotate_fn=="R1" or rotate_fn=="R2":
            Ah = A*norm*denominator_a
            Bh = B*norm*denominator_a
            Ch = C*norm*denominator_a
            Dh = D*norm*denominator_a   
        else:
            Ah = A*norm
            Bh = B*norm
            Ch = C*norm
            Dh = D*norm              
        # Hs[:,:,0] = A
        # Hs[:,:,1] = B
        # Hs[:,:,2] = C
        # Hs[:,:,3] = D
        Hs0 = torch.cat((Ah.unsqueeze(-1),Bh.unsqueeze(-1),Ch.unsqueeze(-1),Dh.unsqueeze(-1)), dim=-1)

        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        # A, B, C, D = self._quat_mul(T[:,:,0], T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # A, B, C, D = self._quat_mul(T[:,:,0], 1+ T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # num = torch.sqrt(T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        # den = torch.sqrt(1+2*T[:,:,1] + T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_c/(1+x_c+epsilon_xa), y_c/(1+x_c+epsilon_xa), Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2) 
            den = np.sqrt(2.) 
        else:
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        norm = torch.div(num,den)
        # At = A*norm
        # Bt = B*norm
        # Ct = C*norm
        # Dt = D*norm 
        if rotate_fn=="R1" or rotate_fn=="R2":  
            At = A*norm*denominator_c
            Bt = B*norm*denominator_c
            Ct = C*norm*denominator_c
            Dt = D*norm*denominator_c      
        else:
            At = A*norm
            Bt = B*norm
            Ct = C*norm
            Dt = D*norm     
        # Ts[:,:,0] = A
        # Ts[:,:,1] = B
        # Ts[:,:,2] = C
        # Ts[:,:,3] = D
        Ts0 = torch.cat((At.unsqueeze(-1),Bt.unsqueeze(-1),Ct.unsqueeze(-1),Dt.unsqueeze(-1)), dim=-1)

        if self.wassertein_approx:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])

            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            Hs1 = torch.cat([Ahr,Bhr,Chr,Dhr], dim=-1)
            Ts1 = torch.cat([Atr,Btr,Ctr,Dtr], dim=-1)

            # find the wass dist
            dist, P, C = self.sinkhorn(Hs1, Ts1)

            # dist = dist.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])    
            delta_min1 = torch.min(P, dim=-1)
            dr3 = delta_min1[1].unsqueeze(-1).repeat([1,1,self.hidden_dim])

            # Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3)
            Btrm = torch.gather(Btr,dim=1,index=dr3)
            Ctrm = torch.gather(Ctr,dim=1,index=dr3)
            Dtrm = torch.gather(Dtr,dim=1,index=dr3)

            d1 = Ahr-Atrm
            d2 = Bhr-Btrm
            d3 = Chr-Ctrm
            d4 = Dhr-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            d7 = torch.mean(d7,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

            # return dist
        else:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])

            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            # find the pairwise dist
            # dw = torch.cdist(Ahr,Atr).reshape(Ah.shape[0]//self.num_heads,Ah.shape[1],self.num_heads,self.num_heads)
            # dx = torch.cdist(Bhr,Btr).reshape(Bh.shape[0]//self.num_heads,Bh.shape[1],self.num_heads,self.num_heads)
            # dy = torch.cdist(Chr,Ctr).reshape(Ch.shape[0]//self.num_heads,Ch.shape[1],self.num_heads,self.num_heads)
            # dz = torch.cdist(Dhr,Dtr).reshape(Dh.shape[0]//self.num_heads,Dh.shape[1],self.num_heads,self.num_heads)
            dw = torch.cdist(Ahr,Atr)
            dx = torch.cdist(Bhr,Btr)
            dy = torch.cdist(Chr,Ctr)
            dz = torch.cdist(Dhr,Dtr)
                       
            delta = dw + dx + dy + dz

            # take the min, min or max, min
            delta_min1 = torch.min(delta, dim=-1)
            delta_min2 = torch.min(delta_min1[0], dim=-1)
            dr1 = delta_min1[1]
            dr2 = delta_min2[1].unsqueeze(-1)
            # dr3 = torch.cat((dr2,dr1[dr2]), dim=-1)
            dr1g = torch.gather(dr1,dim=1,index=dr2)
            dr3 = torch.cat((dr2,dr1g), dim=-1)

            Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Btrm = torch.gather(Btr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Ctrm = torch.gather(Ctr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dtrm = torch.gather(Dtr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))

            d1 = Ahrm-Atrm
            d2 = Bhrm-Btrm
            d3 = Chrm-Ctrm
            d4 = Dhrm-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])
            # # take the sum
            # delta = dw + dx + dy + dz

            # # take the min, min or max, min
            # delta_min1 = torch.min(delta, dim=-1)[0]
            # delta_min2 = torch.min(delta_min1, dim=-1)[0]     


            # return delta_min2

        # Hs0 = Hs0.unsqueeze(dim=1).reshape(Hs0.shape[0]//self.PADDING,self.PADDING,Hs0.shape[1],Hs0.shape[2],Hs.shape[3])
        # Ts0 = Ts0.unsqueeze(dim=1).reshape(Ts0.shape[0]//self.PADDING,self.PADDING,Ts0.shape[1],Ts0.shape[2],Hs.shape[3])

        # Hs0 = torch.mean(Hs0, dim=1)
        # Ts0 = torch.mean(Ts0, dim=1)

        # delta_w = Hs0[:,:,:,0] - Ts0[:,:,:,0]
        # delta_x = Hs0[:,:,:,1] - Ts0[:,:,:,1]
        # delta_y = Hs0[:,:,:,2] - Ts0[:,:,:,2]
        # delta_z = Hs0[:,:,:,3] - Ts0[:,:,:,3]

        # if not single_mode:
        #     delta_w = delta_w.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_x = delta_x.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_y = delta_y.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_z = delta_z.unsqueeze(-1).reshape((-1,neg_samples))

        # return delta_w, delta_x, delta_y, delta_z
        # Rs = torch.repeat_interleave(Rs0, set_sizes_h, dim=0)
        # # Rs = Rs0
        # # Rotate in 4-D using the relation quaternion
        # A, B, C, D = self._quat_mul(Hs0[:,:,0], Hs0[:,:,1], Hs0[:,:,2], Hs0[:,:,3], Rs[:,:,0], Rs[:,:,1], Rs[:,:,2], Rs[:,:,3])
        # # Hs[:,:,0] = A
        # # Hs[:,:,1] = B
        # # Hs[:,:,2] = C
        # # Hs[:,:,3] = D
        # Hs1 = torch.cat((A.unsqueeze(-1),B.unsqueeze(-1),C.unsqueeze(-1),D.unsqueeze(-1)), dim=-1)

        # '''# Repeat elements along Hs t times
        # # Repeat set elements along Ts h times
        # set_sizes_h2 = torch.repeat_interleave(set_sizes_t, set_sizes_h)
        # set_sizes_t2 = torch.repeat_interleave(set_sizes_h, set_sizes_t)
        # Hs2 = torch.repeat_interleave(Hs1, set_sizes_h2, dim=0)
        # Ts2 = torch.repeat_interleave(Ts0, set_sizes_t2, dim=0)
        # gather_batch_indices = []
        # cur_batch_indices = []
        # j = 0
        # for i in range(Ts0.shape[0]):
        #     cur_batch_indices.append(i)
        #     if len(cur_batch_indices)==set_sizes_t[j]:
        #         cur_batch_indices = cur_batch_indices*set_sizes_h[j]
        #         gather_batch_indices.extend(cur_batch_indices)
        #         cur_batch_indices = []
        #         j += 1
        # if self.CUDA:
        #     gather_batch_indices = torch.tensor(gather_batch_indices).cuda()
        # else:
        #     gather_batch_indices = torch.tensor(gather_batch_indices)
        #     # gather_indices = torch.ones(Ts2.shape)*gather_batch_indices 
        # gather_indices = gather_batch_indices.unsqueeze(-1).unsqueeze(-1).repeat([1,Ts2.shape[1],Ts2.shape[2]])
        # Ts2 = torch.gather(Ts2, 0, gather_indices)
        # set_sizes_gather = set_sizes_h*set_sizes_t'''
        # Ts2 = Ts0
        # Hs2 = Hs1

        # score_r = (Hs2[:,:,0] * Ts2[:,:,0] + Hs2[:,:,1] * Ts2[:,:,1] + Hs2[:,:,2] * Ts2[:,:,2] + Hs2[:,:,3] * Ts2[:,:,3])
        # score_r = -torch.sum(score_r, -1)
        # score_r = score_r.view([score_r.shape[0]//self.PADDING,self.PADDING]).unsqueeze(1)
        # pooled_score = torch.nn.MaxPool1d(self.PADDING, stride=self.PADDING)(score_r).squeeze()
        # return pooled_score


    def _calc_semantic_score_optim_1head(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t, rotate_fn="R1"):

        import time
        _t1 = time.time()
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        _t2 = time.time()
        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        _t3 = time.time()

        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2":
            denominator_a = torch.sqrt(s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
            s_a = s_a / denominator_a
            x_a = x_a / denominator_a
            y_a = y_a / denominator_a
            z_a = z_a / denominator_a

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2": 
            denominator_c = torch.sqrt(s_c ** 2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
            s_c = s_c / denominator_c
            x_c = x_c / denominator_c
            y_c = y_c / denominator_c
            z_c = z_c / denominator_c

        # Rs0 = torch.cat((s_b.unsqueeze(-1),x_b.unsqueeze(-1),y_b.unsqueeze(-1),z_b.unsqueeze(-1)), dim=-1)

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        # A, B, C, D = self._quat_mul(H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # A, B, C, D = self._quat_mul(H[:,:,0], 1+H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # num = torch.sqrt(H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        # den = torch.sqrt(1+2*H[:,:,1] + H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)

        assert s_a.eq(0).all().cpu().numpy(), "s_a must be equal to 0"
        # assert (x_a**2 + y_a**2 + z_a**2).eq(1).all().cpu().numpy(), "(x_a**2 + y_a**2 + z_a**2) must be equal to 1"

        _t4 = time.time()

        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_a/(1+x_a+epsilon_xa), y_a/(1+x_a+epsilon_xa), Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2) 
            den = np.sqrt(2.)      
        else:
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)     

        _t5 = time.time()

        norm = torch.div(num,den)
        # Ah = A*norm
        # Bh = B*norm
        # Ch = C*norm
        # Dh = D*norm
        if rotate_fn=="R1" or rotate_fn=="R2":
            Ah = A*norm*denominator_a
            Bh = B*norm*denominator_a
            Ch = C*norm*denominator_a
            Dh = D*norm*denominator_a   
        else:
            Ah = A*norm
            Bh = B*norm
            Ch = C*norm
            Dh = D*norm              
        # Hs[:,:,0] = A
        # Hs[:,:,1] = B
        # Hs[:,:,2] = C
        # Hs[:,:,3] = D
        # Hs0 = torch.cat((Ah.unsqueeze(-1),Bh.unsqueeze(-1),Ch.unsqueeze(-1),Dh.unsqueeze(-1)), dim=-1)

        _t6 = time.time()

        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        # A, B, C, D = self._quat_mul(T[:,:,0], T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # A, B, C, D = self._quat_mul(T[:,:,0], 1+ T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # num = torch.sqrt(T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        # den = torch.sqrt(1+2*T[:,:,1] + T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_c/(1+x_c+epsilon_xa), y_c/(1+x_c+epsilon_xa), Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2) 
            den = np.sqrt(2.) 
        else:
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        _t7 = time.time()
        norm = torch.div(num,den)
        # At = A*norm
        # Bt = B*norm
        # Ct = C*norm
        # Dt = D*norm 
        if rotate_fn=="R1" or rotate_fn=="R2":  
            At = A*norm*denominator_c
            Bt = B*norm*denominator_c
            Ct = C*norm*denominator_c
            Dt = D*norm*denominator_c      
        else:
            At = A*norm
            Bt = B*norm
            Ct = C*norm
            Dt = D*norm     

        # Ts0 = torch.cat((At.unsqueeze(-1),Bt.unsqueeze(-1),Ct.unsqueeze(-1),Dt.unsqueeze(-1)), dim=-1)

        # if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
        #     Ah = Ah.repeat([1,At.shape[1],1])
        #     Bh = Bh.repeat([1,Bt.shape[1],1])
        #     Ch = Ch.repeat([1,Ct.shape[1],1])
        #     Dh = Dh.repeat([1,Dt.shape[1],1])
        # elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
        #     At = At.repeat([1,Ah.shape[1],1])
        #     Bt = Bt.repeat([1,Bh.shape[1],1])
        #     Ct = Ct.repeat([1,Ch.shape[1],1])
        #     Dt = Dt.repeat([1,Dh.shape[1],1])

        _t8 = time.time()

        # Reshape Ah, Bh, Ch, Dh
        # Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
        # Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
        # Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
        # Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

        # Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
        # Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
        # Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
        # Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
        
        _t9 = time.time()

        # # find the pairwise dist
        # dw = torch.cdist(Ahr,Atr)
        # dx = torch.cdist(Bhr,Btr)
        # dy = torch.cdist(Chr,Ctr)
        # dz = torch.cdist(Dhr,Dtr)
                   
        # delta = dw + dx + dy + dz
        _t10 = time.time()

        # # take the min, min or max, min
        # delta_min1 = torch.min(delta, dim=-1)
        # delta_min2 = torch.min(delta_min1[0], dim=-1)
        _t11 = time.time()

        # dr1 = delta_min1[1]
        # dr2 = delta_min2[1].unsqueeze(-1)
        # # dr3 = torch.cat((dr2,dr1[dr2]), dim=-1)
        # dr1g = torch.gather(dr1,dim=1,index=dr2)
        # dr3 = torch.cat((dr2,dr1g), dim=-1)

        _t12 = time.time()

        # Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Atrm = torch.gather(Atr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Btrm = torch.gather(Btr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Ctrm = torch.gather(Ctr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Dtrm = torch.gather(Dtr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))

        Ahrm = Ah
        Bhrm = Bh
        Chrm = Ch
        Dhrm = Dh
        Atrm = At
        Btrm = Bt
        Ctrm = Ct
        Dtrm = Dt

        _t13 = time.time()

        d1 = Ahrm-Atrm
        d2 = Bhrm-Btrm
        d3 = Chrm-Ctrm
        d4 = Dhrm-Dtrm
        d5 = torch.stack([d1,d2,d3,d4],dim=0)
        d6 = d5.norm(dim=0)
        d7 = torch.mean(d6,dim=-1)

        _t14 = time.time()

        # print('== time')
        # print(_t2-_t1,_t3-_t2,_t4-_t3,_t5-_t4,_t6-_t5,_t7-_t6,_t8-_t7,_t9-_t8,_t10-_t9,_t11-_t10,_t12-_t11,_t13-_t12,_t14-_t13)
        # return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])
        num_negative_samples = max(Ah.shape[1],At.shape[1])
        return d7.reshape(Ah.shape[0],num_negative_samples)

    def HopfParaE(self, head_x, head_y, head_z, 
                   rel_w, rel_x, rel_y, rel_z, 
                   tail_x, tail_y, tail_z, 
                   mode):
        pi = 3.14159265358979323846
        assert(self.use_entity_phase == False)
        assert(self.use_real_part == False)
        
        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        # with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
        
        # compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
        # compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
        # compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
        _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z)

        if self.relation_embedding_has_mod:
            compute_tail_x = denominator * compute_tail_x
            compute_tail_y = denominator * compute_tail_y
            compute_tail_z = denominator * compute_tail_z

        if not self.single_param:
            param_s_a = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_s_a = param_s_a.permute([0,2,1]).reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[2]//self.num_heads, param_s_a.shape[1])).permute([0,2,1])
            else:
                param_s_a = param_s_a.reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[1]//self.num_heads))
            param_x_a = self.param_ent_x_a(self.batch_h)    
            if len(self.batch_h.shape) == 2:
                param_x_a = param_x_a.permute([0,2,1]).reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[2]//self.num_heads, param_x_a.shape[1])).permute([0,2,1])
            else:
                param_x_a = param_x_a.reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[1]//self.num_heads))
            norm_a = torch.sqrt(param_s_a**2 + param_x_a**2)
            param_s_a = param_s_a / norm_a
            param_x_a = param_x_a / norm_a
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)

            param_s_c = self.param_ent_s_a(self.batch_t)
            if len(self.batch_t.shape) == 2:
                param_s_c = param_s_c.permute([0,2,1]).reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[2]//self.num_heads, param_s_c.shape[1])).permute([0,2,1])
            else:
                param_s_c = param_s_c.reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[1]//self.num_heads))        
            param_x_c = self.param_ent_x_a(self.batch_t)    
            if len(self.batch_t.shape) == 2:
                param_x_c = param_x_c.permute([0,2,1]).reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[2]//self.num_heads, param_x_c.shape[1])).permute([0,2,1])
            else:
                param_x_c = param_x_c.reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[1]//self.num_heads))        
            norm_c = torch.sqrt(param_s_c**2 + param_x_c**2)
            param_s_c = param_s_c / norm_c
            param_x_c = param_x_c / norm_c
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)           
            if len(self.batch_t.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)
        else:
            param_a_prim = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_a_prim = param_a_prim.permute([0,2,1]).reshape((param_a_prim.shape[0]*self.num_heads, param_a_prim.shape[2]//self.num_heads, param_a_prim.shape[1])).permute([0,2,1])
            else:
                param_a_prim = param_a_prim.reshape((param_a_prim.shape[0]*self.num_heads, param_a_prim.shape[1]//self.num_heads))
            
            param_s_a = torch.cos(param_a_prim)
            param_x_a = torch.sin(param_a_prim)
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)   

            param_c_prim = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_c_prim = param_c_prim.permute([0,2,1]).reshape((param_c_prim.shape[0]*self.num_heads, param_c_prim.shape[2]//self.num_heads, param_c_prim.shape[1])).permute([0,2,1])
            else:
                param_c_prim = param_c_prim.reshape((param_c_prim.shape[0]*self.num_heads, param_c_prim.shape[1]//self.num_heads))
            
            param_s_c = torch.cos(param_c_prim)
            param_x_c = torch.sin(param_c_prim)
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)  
            if len(self.batch_h.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)            

        # structural_score = self._calc_structural_score(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        # if self.semantic_score_fn == 'pooled':
        #     semantic_score = self._calc_pooled_semantic_score_optim(param_a, param_b, Rs, H, T, set_sizes_h, set_sizes_t)
        # else:    
        #     semantic_score = self._calc_semantic_score_optim(param_a, param_b, Rs, H, T, set_sizes_h, set_sizes_t)
        if self.CUDA:
            H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1).cuda(),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
            T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1).cuda(),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
        else:
            H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
            T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
        if self.num_heads>1:
            score1 = self._calc_semantic_score_optim(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)
        else:
            score1 = self._calc_semantic_score_optim_1head(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)

        
        # delta_x = (compute_tail_x - tail_x)
        # delta_y = (compute_tail_y - tail_y)
        # delta_z = (compute_tail_z - tail_z)
        
        # score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
        # score1 = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
        # score1 = score1.norm(dim = 0)
        
        x = -x
        y = -y
        z = -z
        # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
        # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
        # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
        _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z)

        if self.relation_embedding_has_mod:
            compute_head_x = compute_head_x / denominator
            compute_head_y = compute_head_y / denominator
            compute_head_z = compute_head_z / denominator
        
        if self.CUDA:
            T = torch.cat((torch.zeros(compute_head_x.shape).cuda().unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
            H = torch.cat((torch.zeros(head_x.shape).cuda().unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
        else:
            T = torch.cat((torch.zeros(compute_head_x.shape).unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
            H = torch.cat((torch.zeros(head_x.shape).unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
        if self.num_heads>1:
            score2 = self._calc_semantic_score_optim(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)
        else:
            score2 = self._calc_semantic_score_optim_1head(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)


        # delta_x2 = (compute_head_x - head_x)
        # delta_y2 = (compute_head_y - head_y)
        # delta_z2 = (compute_head_z - head_z)
        
        # score2 = torch.stack([delta_w2, delta_x2, delta_y2, delta_z2], dim = 0)
        # score2 = score2.norm(dim = 0)     
        
        # score1 = score1.mean(dim=2)
        # score2 = score2.mean(dim=2)

   
        score = (score1 + score2) / 2
        
        score = self.gamma.item() - score

        # print(prof.key_averages().table(sort_by="self_gpu_memory_usage", row_limit=100))

        return score, score1, score2

    @staticmethod
    def train_step(model, optimizer, train_iterator, step, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # with LineProfiler(model.HopfParaE, model._calc_semantic_score_optim_1head) as prof:
        negative_score, head_mod, tail_mod = model((positive_sample, negative_sample), mode=mode) # 全是负样本分数 shape: batch_size, neg_size
        # prof.display()

        if step % 500 == 0:
            print(negative_score.mean(), head_mod.mean(), tail_mod.mean())

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        # with LineProfiler(model.HopfParaE, model._calc_semantic_score_optim_1head) as prof:
        positive_score, head_mod, tail_mod = model(positive_sample) # 正样本分数 shape: batch_size, 1     
        # prof.display()

        if step % 500 == 0:
            print(positive_score.mean(), head_mod.mean(), tail_mod.mean())

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_x.weight.data.norm(p = 3)**3 + 
                model.entity_y.weight.data.norm(p = 3)**3 + 
                model.entity_z.weight.data.norm(p = 3)**3 
            ) / args.batch_size

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score, head_mod, tail_mod = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
    def get_entity_properties_from_npy(self, batch_indices, pos='head'):
        # pass
        n = batch_indices.shape[0]
        if len(batch_indices.shape)==1:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_h[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_t[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
        else:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)

class HopfParaEBiasModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 entity_embedding_has_mod=False, relation_embedding_has_mod=False, rotate_fn="R1", params=None):
        super(HopfParaEBiasModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 1.2
        self.rel_high_bound = 2.0
        
        self.use_abs_norm = True
        self.allow_minus_mod = True
        self.use_entity_phase = False
        self.use_real_part = False
        
        self.criterion = 'he'
        
        if self.criterion == 'glorot':
            mod_range = 1. / np.sqrt(2 * (self.hidden_dim + self.hidden_dim))
        elif self.criterion == 'he':
            mod_range = 1. / np.sqrt(2 * self.hidden_dim)
        
        if self.allow_minus_mod:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range * 2.]), 
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range]), 
                requires_grad=False
            )
            
        self.gamma1 = nn.Parameter(
            torch.Tensor([(self.rel_high_bound + self.epsilon) * mod_range * self.hidden_dim]), 
            requires_grad=False
        )
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.unit_mod = nn.Parameter(
            torch.Tensor([1.]), 
            requires_grad=False
        )
        
        self.zero_ent_phase = nn.Parameter(
            torch.Tensor([0.]), 
            requires_grad=False
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.entity_embedding_has_mod = entity_embedding_has_mod
        self.relation_embedding_has_mod = relation_embedding_has_mod
                   
        self.entity_x = nn.Embedding(self.nentity, self.hidden_dim)
        self.entity_y = nn.Embedding(self.nentity, self.hidden_dim)   
        self.entity_z = nn.Embedding(self.nentity, self.hidden_dim)
        
        self.relation_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_z = nn.Embedding(self.nrelation, self.hidden_dim)
        
        self.relation_b = nn.Embedding(self.nrelation, self.hidden_dim)

        self.init_weights()
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['HopfParaEBias']:
            raise ValueError('model %s not supported' % model_name)
        if self.use_real_part:
            try:
                assert(self.use_abs_norm == True)
            except:
                raise ValueError('use_abs_norm should be true if you only use real part')
        if (not self.entity_embedding_has_mod) and self.relation_embedding_has_mod:
            raise ValueError('when relation has mod, entity must have mod')
        
        if params is None:
            params={    
                'embeddings_path': '../../data/FB15K237_context/limit1_v3/embeddings.npy',
                'char_vocab_path': '../../data/FB15K237_context/limit1_v3/char2idx.json',
                'char_feature_size': 50,
                'char_embed_dim': 50,
                'max_word_len_entity': 10,
                'conv_filter_size': 3,
                'drop_rate': 0.0,
                'max_sent_len': 16,
                'entity_indices_file': '../../data/FB15K237_context/limit1_v3/entity_context_indices.json',
                'word2idx_path': '../../data/FB15K237_context/limit1_v3/word2idx.json',
                'all_word_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/word_indices_h.npy',
                'all_char_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/char_indices_h.npy',
                'mask_file_h': '../../data/FB15K237_context/limit1_v3/mask_h.npy',
                'all_word_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/word_indices_t.npy',
                'all_char_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/char_indices_t.npy',
                'mask_file_t': '../../data/FB15K237_context/limit1_v3/mask_t.npy',
                'padding': 1,
                'checkpoint_json_path': './result_hopfe_rot_2/HopfE.json',
                'num_heads': 1,
                'wassertein_approx': False
            }
        self.embeddings = np.load(params['embeddings_path'])
        with open(params['char_vocab_path'], 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.all_word_token_ids_h = np.load(params['all_word_token_ids_file_h'])
        self.all_char_token_ids_h = np.load(params['all_char_token_ids_file_h'])
        self.all_word_token_ids_t = np.load(params['all_word_token_ids_file_t'])
        self.all_char_token_ids_t = np.load(params['all_char_token_ids_file_t'])
        self.PADDING = params['padding']
        self.max_sent_len = params['max_sent_len']
        self.conv_filter_size = params['conv_filter_size']
        self.max_char_len = params['max_word_len_entity']
        self.char_embed_dim = params['char_embed_dim']
        self.drop_rate = params['drop_rate']
        self.mask_h = np.load(params['mask_file_h'])
        self.mask_t = np.load(params['mask_file_t'])
        self.char_feature_size = params['char_feature_size']

        self.semanticE = CbowE({'embeddings': self.embeddings, 
            'char_vocab': self.char_vocab, 
            'char_feature_size': self.char_feature_size, 
            'char_embed_dim': self.char_embed_dim, 
            'max_word_len_entity': self.max_char_len, 
            'conv_filter_size': self.conv_filter_size, 
            'drop_rate': self.drop_rate})
        self.linear_proj = nn.Linear(50, 2*self.hidden_dim)


        self.num_heads = params['num_heads']
        self.param_ent_s_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)
        self.param_ent_x_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)
        self.CUDA = torch.cuda.is_available()

        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
        self.wassertein_approx = params.get('wassertein_approx', False)
        self.single_param = False

        self.rotate_fn = rotate_fn

    def init_weights(self):

        rel_w, rel_x, rel_y, rel_z, rel_b = self.relation_init(self.nrelation, self.hidden_dim)
        rel_w, rel_x, rel_y, rel_z, rel_b = torch.from_numpy(rel_w), torch.from_numpy(rel_x), torch.from_numpy(rel_y), torch.from_numpy(rel_z), torch.from_numpy(rel_b)
        self.relation_w.weight.data = rel_w.type_as(self.relation_w.weight.data)
        self.relation_x.weight.data = rel_x.type_as(self.relation_x.weight.data)
        self.relation_y.weight.data = rel_y.type_as(self.relation_y.weight.data)
        self.relation_z.weight.data = rel_z.type_as(self.relation_z.weight.data)
        self.relation_b.weight.data = rel_b.type_as(self.relation_b.weight.data)
        
        e_x, e_y, e_z = self.entity_init(self.nentity, self.hidden_dim)
        e_x, e_y, e_z = torch.from_numpy(e_x), torch.from_numpy(e_y), torch.from_numpy(e_z)
        self.entity_x.weight.data = e_x.type_as(self.entity_x.weight.data)
        self.entity_y.weight.data = e_y.type_as(self.entity_y.weight.data)
        self.entity_z.weight.data = e_z.type_as(self.entity_z.weight.data)
        
    def relation_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features
        
        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init rel_mod is: ', s)

        kernel_shape = (n_entries, features)
            
        rel_mod = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rotate_phase = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=kernel_shape)
        theta = np.random.uniform(low=0, high=np.pi, size=kernel_shape)
        phi = np.random.uniform(low=0, high=2*np.pi, size=kernel_shape)
        
        rel_w = rel_mod * np.cos(rotate_phase/2)
        rel_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
        rel_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
        rel_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

        rel_b = np.ones(kernel_shape)

        return rel_w, rel_x, rel_y, rel_z, rel_b

    def entity_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init x, y, z is: ', s)

        # rng = RandomState(456)
        kernel_shape = (n_entries, features)
            
        x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return x, y, z
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # batch_size, negative_sample_size = sample.size(0), 1
            
            head_x = self.entity_x(sample[:, 0]).unsqueeze(1)
            head_y = self.entity_y(sample[:, 0]).unsqueeze(1)
            head_z = self.entity_z(sample[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(sample[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(sample[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(sample[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(sample[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(sample[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(sample[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(sample[:, 1]).unsqueeze(1)

            rel_bias = self.relation_b(sample[:, 1]).unsqueeze(1)

            self.batch_h = sample[:, 0]
            self.batch_r = sample[:, 1]
            self.batch_t = sample[:, 2]
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_x = self.entity_x(head_part)
            head_y = self.entity_y(head_part)
            head_z = self.entity_z(head_part)
            
            tail_x = self.entity_x(tail_part[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(tail_part[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(tail_part[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(tail_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(tail_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(tail_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(tail_part[:, 1]).unsqueeze(1)

            rel_bias = self.relation_b(tail_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part
            self.batch_r = tail_part[:, 1]
            self.batch_t = tail_part[:, 2]
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_x = self.entity_x(head_part[:, 0]).unsqueeze(1)
            head_y = self.entity_y(head_part[:, 0]).unsqueeze(1)
            head_z = self.entity_z(head_part[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(tail_part)
            tail_y = self.entity_y(tail_part)
            tail_z = self.entity_z(tail_part)
            
            rel_w = self.relation_w(head_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(head_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(head_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(head_part[:, 1]).unsqueeze(1)

            rel_bias = self.relation_b(head_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part[:, 0]
            self.batch_r = head_part[:, 1]
            self.batch_t = tail_part
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'HopfParaEBias': self.HopfParaEBias
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_x, head_y, head_z, 
                                                rel_w, rel_x, rel_y, rel_z, 
                                                tail_x, tail_y, tail_z, rel_bias,
                                                mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def _quat_mul(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        
        return (A, B, C, D)

    def rotate(self, x, y, z, rel_w, rel_x, rel_y, rel_z, b):
        A, B, C, D = self._quat_mul(rel_w, rel_x, rel_y, rel_z, 0, x, y, z)
        A, B, C, D = self._quat_mul(A, B, C, D, rel_w, -1.0*rel_x, -1.0*rel_y, -1.0*rel_z)
        return A*b, B*b, C*b, D*b


    def _calc_semantic_score_optim_test(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t):
        
        Ah = H[:,:,:,0]
        Bh = H[:,:,:,1]
        Ch = H[:,:,:,2]
        Dh = H[:,:,:,3]

        At = T[:,:,:,0]
        Bt = T[:,:,:,1]
        Ct = T[:,:,:,2]
        Dt = T[:,:,:,3]

        if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
            Ah = Ah.repeat([1,At.shape[1],1])
            Bh = Bh.repeat([1,Bt.shape[1],1])
            Ch = Ch.repeat([1,Ct.shape[1],1])
            Dh = Dh.repeat([1,Dt.shape[1],1])
        elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
            At = At.repeat([1,Ah.shape[1],1])
            Bt = Bt.repeat([1,Bh.shape[1],1])
            Ct = Ct.repeat([1,Ch.shape[1],1])
            Dt = Dt.repeat([1,Dh.shape[1],1])

        # Reshape Ah, Bh, Ch, Dh
        Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2]).contiguous()
        Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2]).contiguous()
        Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2]).contiguous()
        Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2]).contiguous()

        Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2]).contiguous()
        Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2]).contiguous()
        Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2]).contiguous()
        Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2]).contiguous()
        
        # find the pairwise dist
        # dw = torch.cdist(Ahr,Atr).reshape(Ah.shape[0]//self.num_heads,Ah.shape[1],self.num_heads,self.num_heads)
        # dx = torch.cdist(Bhr,Btr).reshape(Bh.shape[0]//self.num_heads,Bh.shape[1],self.num_heads,self.num_heads)
        # dy = torch.cdist(Chr,Ctr).reshape(Ch.shape[0]//self.num_heads,Ch.shape[1],self.num_heads,self.num_heads)
        # dz = torch.cdist(Dhr,Dtr).reshape(Dh.shape[0]//self.num_heads,Dh.shape[1],self.num_heads,self.num_heads)
        d1 = Ahr-Atr
        d2 = Bhr-Btr
        d3 = Chr-Ctr
        d4 = Dhr-Dtr
        d5 = torch.stack([d1,d2,d3,d4],dim=0)
        d6 = d5.norm(dim=0)
        d7 = torch.mean(d6,dim=-1)
        return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

        # # take the sum
        # delta = dw + dx + dy + dz

        # # take the min, min or max, min
        # delta_min1 = torch.min(delta, dim=-1)[0]
        # delta_min2 = torch.min(delta_min1, dim=-1)[0]     


        # return delta_min2

    def _calc_semantic_score_optim(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t, rotate_fn="R1"):

        # single_mode = True
        # if len(Hs.shape)==4:
        #     single_mode = False
        #     neg_samples = Hs.shape[1]
        #     Hs = Hs.reshape((-1,Hs.shape[2],Hs.shape[3]))
        #     set_sizes_h = set_sizes_h * neg_samples
        # if len(Ts.shape)==4:
        #     single_mode = False
        #     neg_samples = Ts.shape[1]
        #     Ts = Ts.reshape((-1,Ts.shape[2],Ts.shape[3]))
        #     set_sizes_t = set_sizes_t * neg_samples

        # s_b = Rs[:,:,0]
        # x_b = Rs[:,:,1]
        # y_b = Rs[:,:,2]
        # z_b = Rs[:,:,3]
        # denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        # s_b = s_b / denominator_b
        # x_b = x_b / denominator_b
        # y_b = y_b / denominator_b
        # z_b = z_b / denominator_b

        H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        
        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2":
            denominator_a = torch.sqrt(s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
            s_a = s_a / denominator_a
            x_a = x_a / denominator_a
            y_a = y_a / denominator_a
            z_a = z_a / denominator_a

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2": 
            denominator_c = torch.sqrt(s_c ** 2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
            s_c = s_c / denominator_c
            x_c = x_c / denominator_c
            y_c = y_c / denominator_c
            z_c = z_c / denominator_c

        # Rs0 = torch.cat((s_b.unsqueeze(-1),x_b.unsqueeze(-1),y_b.unsqueeze(-1),z_b.unsqueeze(-1)), dim=-1)

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        # A, B, C, D = self._quat_mul(H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # A, B, C, D = self._quat_mul(H[:,:,0], 1+H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # num = torch.sqrt(H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        # den = torch.sqrt(1+2*H[:,:,1] + H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)

        assert s_a.eq(0).all().cpu().numpy(), "s_a must be equal to 0"
        assert (x_a**2 + y_a**2 + z_a**2).eq(1).all().cpu().numpy(), "(x_a**2 + y_a**2 + z_a**2) must be equal to 1"

        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_a/(1+x_a+epsilon_xa), y_a/(1+x_a+epsilon_xa), Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2) 
            den = np.sqrt(2.)      
        else:
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)         
        norm = torch.div(num,den)
        # Ah = A*norm
        # Bh = B*norm
        # Ch = C*norm
        # Dh = D*norm
        if rotate_fn=="R1" or rotate_fn=="R2":
            Ah = A*norm*denominator_a
            Bh = B*norm*denominator_a
            Ch = C*norm*denominator_a
            Dh = D*norm*denominator_a   
        else:
            Ah = A*norm
            Bh = B*norm
            Ch = C*norm
            Dh = D*norm              
        # Hs[:,:,0] = A
        # Hs[:,:,1] = B
        # Hs[:,:,2] = C
        # Hs[:,:,3] = D
        Hs0 = torch.cat((Ah.unsqueeze(-1),Bh.unsqueeze(-1),Ch.unsqueeze(-1),Dh.unsqueeze(-1)), dim=-1)

        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        # A, B, C, D = self._quat_mul(T[:,:,0], T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # A, B, C, D = self._quat_mul(T[:,:,0], 1+ T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # num = torch.sqrt(T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        # den = torch.sqrt(1+2*T[:,:,1] + T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_c/(1+x_c+epsilon_xa), y_c/(1+x_c+epsilon_xa), Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2) 
            den = np.sqrt(2.) 
        else:
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        norm = torch.div(num,den)
        # At = A*norm
        # Bt = B*norm
        # Ct = C*norm
        # Dt = D*norm 
        if rotate_fn=="R1" or rotate_fn=="R2":  
            At = A*norm*denominator_c
            Bt = B*norm*denominator_c
            Ct = C*norm*denominator_c
            Dt = D*norm*denominator_c      
        else:
            At = A*norm
            Bt = B*norm
            Ct = C*norm
            Dt = D*norm     
        # Ts[:,:,0] = A
        # Ts[:,:,1] = B
        # Ts[:,:,2] = C
        # Ts[:,:,3] = D
        Ts0 = torch.cat((At.unsqueeze(-1),Bt.unsqueeze(-1),Ct.unsqueeze(-1),Dt.unsqueeze(-1)), dim=-1)

        if self.wassertein_approx:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])

            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            Hs1 = torch.cat([Ahr,Bhr,Chr,Dhr], dim=-1)
            Ts1 = torch.cat([Atr,Btr,Ctr,Dtr], dim=-1)

            # find the wass dist
            dist, P, C = self.sinkhorn(Hs1, Ts1)

            # dist = dist.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])    
            delta_min1 = torch.min(P, dim=-1)
            dr3 = delta_min1[1].unsqueeze(-1).repeat([1,1,self.hidden_dim])

            # Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3)
            Btrm = torch.gather(Btr,dim=1,index=dr3)
            Ctrm = torch.gather(Ctr,dim=1,index=dr3)
            Dtrm = torch.gather(Dtr,dim=1,index=dr3)

            d1 = Ahr-Atrm
            d2 = Bhr-Btrm
            d3 = Chr-Ctrm
            d4 = Dhr-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            d7 = torch.mean(d7,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

            # return dist
        else:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])

            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            # find the pairwise dist
            # dw = torch.cdist(Ahr,Atr).reshape(Ah.shape[0]//self.num_heads,Ah.shape[1],self.num_heads,self.num_heads)
            # dx = torch.cdist(Bhr,Btr).reshape(Bh.shape[0]//self.num_heads,Bh.shape[1],self.num_heads,self.num_heads)
            # dy = torch.cdist(Chr,Ctr).reshape(Ch.shape[0]//self.num_heads,Ch.shape[1],self.num_heads,self.num_heads)
            # dz = torch.cdist(Dhr,Dtr).reshape(Dh.shape[0]//self.num_heads,Dh.shape[1],self.num_heads,self.num_heads)
            dw = torch.cdist(Ahr,Atr)
            dx = torch.cdist(Bhr,Btr)
            dy = torch.cdist(Chr,Ctr)
            dz = torch.cdist(Dhr,Dtr)
                       
            delta = dw + dx + dy + dz

            # take the min, min or max, min
            delta_min1 = torch.min(delta, dim=-1)
            delta_min2 = torch.min(delta_min1[0], dim=-1)
            dr1 = delta_min1[1]
            dr2 = delta_min2[1].unsqueeze(-1)
            # dr3 = torch.cat((dr2,dr1[dr2]), dim=-1)
            dr1g = torch.gather(dr1,dim=1,index=dr2)
            dr3 = torch.cat((dr2,dr1g), dim=-1)

            Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Btrm = torch.gather(Btr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Ctrm = torch.gather(Ctr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dtrm = torch.gather(Dtr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))

            d1 = Ahrm-Atrm
            d2 = Bhrm-Btrm
            d3 = Chrm-Ctrm
            d4 = Dhrm-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])
            # # take the sum
            # delta = dw + dx + dy + dz

            # # take the min, min or max, min
            # delta_min1 = torch.min(delta, dim=-1)[0]
            # delta_min2 = torch.min(delta_min1, dim=-1)[0]     


            # return delta_min2

        # Hs0 = Hs0.unsqueeze(dim=1).reshape(Hs0.shape[0]//self.PADDING,self.PADDING,Hs0.shape[1],Hs0.shape[2],Hs.shape[3])
        # Ts0 = Ts0.unsqueeze(dim=1).reshape(Ts0.shape[0]//self.PADDING,self.PADDING,Ts0.shape[1],Ts0.shape[2],Hs.shape[3])

        # Hs0 = torch.mean(Hs0, dim=1)
        # Ts0 = torch.mean(Ts0, dim=1)

        # delta_w = Hs0[:,:,:,0] - Ts0[:,:,:,0]
        # delta_x = Hs0[:,:,:,1] - Ts0[:,:,:,1]
        # delta_y = Hs0[:,:,:,2] - Ts0[:,:,:,2]
        # delta_z = Hs0[:,:,:,3] - Ts0[:,:,:,3]

        # if not single_mode:
        #     delta_w = delta_w.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_x = delta_x.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_y = delta_y.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_z = delta_z.unsqueeze(-1).reshape((-1,neg_samples))

        # return delta_w, delta_x, delta_y, delta_z
        # Rs = torch.repeat_interleave(Rs0, set_sizes_h, dim=0)
        # # Rs = Rs0
        # # Rotate in 4-D using the relation quaternion
        # A, B, C, D = self._quat_mul(Hs0[:,:,0], Hs0[:,:,1], Hs0[:,:,2], Hs0[:,:,3], Rs[:,:,0], Rs[:,:,1], Rs[:,:,2], Rs[:,:,3])
        # # Hs[:,:,0] = A
        # # Hs[:,:,1] = B
        # # Hs[:,:,2] = C
        # # Hs[:,:,3] = D
        # Hs1 = torch.cat((A.unsqueeze(-1),B.unsqueeze(-1),C.unsqueeze(-1),D.unsqueeze(-1)), dim=-1)

        # '''# Repeat elements along Hs t times
        # # Repeat set elements along Ts h times
        # set_sizes_h2 = torch.repeat_interleave(set_sizes_t, set_sizes_h)
        # set_sizes_t2 = torch.repeat_interleave(set_sizes_h, set_sizes_t)
        # Hs2 = torch.repeat_interleave(Hs1, set_sizes_h2, dim=0)
        # Ts2 = torch.repeat_interleave(Ts0, set_sizes_t2, dim=0)
        # gather_batch_indices = []
        # cur_batch_indices = []
        # j = 0
        # for i in range(Ts0.shape[0]):
        #     cur_batch_indices.append(i)
        #     if len(cur_batch_indices)==set_sizes_t[j]:
        #         cur_batch_indices = cur_batch_indices*set_sizes_h[j]
        #         gather_batch_indices.extend(cur_batch_indices)
        #         cur_batch_indices = []
        #         j += 1
        # if self.CUDA:
        #     gather_batch_indices = torch.tensor(gather_batch_indices).cuda()
        # else:
        #     gather_batch_indices = torch.tensor(gather_batch_indices)
        #     # gather_indices = torch.ones(Ts2.shape)*gather_batch_indices 
        # gather_indices = gather_batch_indices.unsqueeze(-1).unsqueeze(-1).repeat([1,Ts2.shape[1],Ts2.shape[2]])
        # Ts2 = torch.gather(Ts2, 0, gather_indices)
        # set_sizes_gather = set_sizes_h*set_sizes_t'''
        # Ts2 = Ts0
        # Hs2 = Hs1

        # score_r = (Hs2[:,:,0] * Ts2[:,:,0] + Hs2[:,:,1] * Ts2[:,:,1] + Hs2[:,:,2] * Ts2[:,:,2] + Hs2[:,:,3] * Ts2[:,:,3])
        # score_r = -torch.sum(score_r, -1)
        # score_r = score_r.view([score_r.shape[0]//self.PADDING,self.PADDING]).unsqueeze(1)
        # pooled_score = torch.nn.MaxPool1d(self.PADDING, stride=self.PADDING)(score_r).squeeze()
        # return pooled_score


    def _calc_semantic_score_optim_1head(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t, rotate_fn="R1"):

        import time
        _t1 = time.time()
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        _t2 = time.time()
        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        _t3 = time.time()

        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2":
            denominator_a = torch.sqrt(s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
            s_a = s_a / denominator_a
            x_a = x_a / denominator_a
            y_a = y_a / denominator_a
            z_a = z_a / denominator_a

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2": 
            denominator_c = torch.sqrt(s_c ** 2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
            s_c = s_c / denominator_c
            x_c = x_c / denominator_c
            y_c = y_c / denominator_c
            z_c = z_c / denominator_c

        # Rs0 = torch.cat((s_b.unsqueeze(-1),x_b.unsqueeze(-1),y_b.unsqueeze(-1),z_b.unsqueeze(-1)), dim=-1)

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        # A, B, C, D = self._quat_mul(H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # A, B, C, D = self._quat_mul(H[:,:,0], 1+H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # num = torch.sqrt(H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        # den = torch.sqrt(1+2*H[:,:,1] + H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)

        assert s_a.eq(0).all().cpu().numpy(), "s_a must be equal to 0"
        # assert (x_a**2 + y_a**2 + z_a**2).eq(1).all().cpu().numpy(), "(x_a**2 + y_a**2 + z_a**2) must be equal to 1"

        _t4 = time.time()

        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_a/(1+x_a+epsilon_xa), y_a/(1+x_a+epsilon_xa), Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2) 
            den = np.sqrt(2.)      
        else:
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)     

        _t5 = time.time()

        norm = torch.div(num,den)
        # Ah = A*norm
        # Bh = B*norm
        # Ch = C*norm
        # Dh = D*norm
        if rotate_fn=="R1" or rotate_fn=="R2":
            Ah = A*norm*denominator_a
            Bh = B*norm*denominator_a
            Ch = C*norm*denominator_a
            Dh = D*norm*denominator_a   
        else:
            Ah = A*norm
            Bh = B*norm
            Ch = C*norm
            Dh = D*norm              
        # Hs[:,:,0] = A
        # Hs[:,:,1] = B
        # Hs[:,:,2] = C
        # Hs[:,:,3] = D
        # Hs0 = torch.cat((Ah.unsqueeze(-1),Bh.unsqueeze(-1),Ch.unsqueeze(-1),Dh.unsqueeze(-1)), dim=-1)

        _t6 = time.time()

        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        # A, B, C, D = self._quat_mul(T[:,:,0], T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # A, B, C, D = self._quat_mul(T[:,:,0], 1+ T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # num = torch.sqrt(T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        # den = torch.sqrt(1+2*T[:,:,1] + T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_c/(1+x_c+epsilon_xa), y_c/(1+x_c+epsilon_xa), Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2) 
            den = np.sqrt(2.) 
        else:
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        _t7 = time.time()
        norm = torch.div(num,den)
        # At = A*norm
        # Bt = B*norm
        # Ct = C*norm
        # Dt = D*norm 
        if rotate_fn=="R1" or rotate_fn=="R2":  
            At = A*norm*denominator_c
            Bt = B*norm*denominator_c
            Ct = C*norm*denominator_c
            Dt = D*norm*denominator_c      
        else:
            At = A*norm
            Bt = B*norm
            Ct = C*norm
            Dt = D*norm     

        # Ts0 = torch.cat((At.unsqueeze(-1),Bt.unsqueeze(-1),Ct.unsqueeze(-1),Dt.unsqueeze(-1)), dim=-1)

        # if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
        #     Ah = Ah.repeat([1,At.shape[1],1])
        #     Bh = Bh.repeat([1,Bt.shape[1],1])
        #     Ch = Ch.repeat([1,Ct.shape[1],1])
        #     Dh = Dh.repeat([1,Dt.shape[1],1])
        # elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
        #     At = At.repeat([1,Ah.shape[1],1])
        #     Bt = Bt.repeat([1,Bh.shape[1],1])
        #     Ct = Ct.repeat([1,Ch.shape[1],1])
        #     Dt = Dt.repeat([1,Dh.shape[1],1])

        _t8 = time.time()

        # Reshape Ah, Bh, Ch, Dh
        # Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
        # Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
        # Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
        # Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

        # Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
        # Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
        # Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
        # Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
        
        _t9 = time.time()

        # # find the pairwise dist
        # dw = torch.cdist(Ahr,Atr)
        # dx = torch.cdist(Bhr,Btr)
        # dy = torch.cdist(Chr,Ctr)
        # dz = torch.cdist(Dhr,Dtr)
                   
        # delta = dw + dx + dy + dz
        _t10 = time.time()

        # # take the min, min or max, min
        # delta_min1 = torch.min(delta, dim=-1)
        # delta_min2 = torch.min(delta_min1[0], dim=-1)
        _t11 = time.time()

        # dr1 = delta_min1[1]
        # dr2 = delta_min2[1].unsqueeze(-1)
        # # dr3 = torch.cat((dr2,dr1[dr2]), dim=-1)
        # dr1g = torch.gather(dr1,dim=1,index=dr2)
        # dr3 = torch.cat((dr2,dr1g), dim=-1)

        _t12 = time.time()

        # Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Atrm = torch.gather(Atr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Btrm = torch.gather(Btr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Ctrm = torch.gather(Ctr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
        # Dtrm = torch.gather(Dtr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))

        Ahrm = Ah
        Bhrm = Bh
        Chrm = Ch
        Dhrm = Dh
        Atrm = At
        Btrm = Bt
        Ctrm = Ct
        Dtrm = Dt

        _t13 = time.time()

        d1 = Ahrm-Atrm
        d2 = Bhrm-Btrm
        d3 = Chrm-Ctrm
        d4 = Dhrm-Dtrm
        d5 = torch.stack([d1,d2,d3,d4],dim=0)
        d6 = d5.norm(dim=0)
        d7 = torch.mean(d6,dim=-1)

        _t14 = time.time()

        # print('== time')
        # print(_t2-_t1,_t3-_t2,_t4-_t3,_t5-_t4,_t6-_t5,_t7-_t6,_t8-_t7,_t9-_t8,_t10-_t9,_t11-_t10,_t12-_t11,_t13-_t12,_t14-_t13)
        # return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])
        num_negative_samples = max(Ah.shape[1],At.shape[1])
        return d7.reshape(Ah.shape[0],num_negative_samples)

    def HopfParaEBias(self, head_x, head_y, head_z, 
                   rel_w, rel_x, rel_y, rel_z, 
                   tail_x, tail_y, tail_z, rel_bias, 
                   mode):
        pi = 3.14159265358979323846
        assert(self.use_entity_phase == False)
        assert(self.use_real_part == False)
        
        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        # with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
        
        # compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
        # compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
        # compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
        _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z, rel_bias)

        if self.relation_embedding_has_mod:
            compute_tail_x = denominator * compute_tail_x
            compute_tail_y = denominator * compute_tail_y
            compute_tail_z = denominator * compute_tail_z

        if not self.single_param:
            param_s_a = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_s_a = param_s_a.permute([0,2,1]).reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[2]//self.num_heads, param_s_a.shape[1])).permute([0,2,1])
            else:
                param_s_a = param_s_a.reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[1]//self.num_heads))
            param_x_a = self.param_ent_x_a(self.batch_h)    
            if len(self.batch_h.shape) == 2:
                param_x_a = param_x_a.permute([0,2,1]).reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[2]//self.num_heads, param_x_a.shape[1])).permute([0,2,1])
            else:
                param_x_a = param_x_a.reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[1]//self.num_heads))
            norm_a = torch.sqrt(param_s_a**2 + param_x_a**2)
            param_s_a = param_s_a / norm_a
            param_x_a = param_x_a / norm_a
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)

            param_s_c = self.param_ent_s_a(self.batch_t)
            if len(self.batch_t.shape) == 2:
                param_s_c = param_s_c.permute([0,2,1]).reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[2]//self.num_heads, param_s_c.shape[1])).permute([0,2,1])
            else:
                param_s_c = param_s_c.reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[1]//self.num_heads))        
            param_x_c = self.param_ent_x_a(self.batch_t)    
            if len(self.batch_t.shape) == 2:
                param_x_c = param_x_c.permute([0,2,1]).reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[2]//self.num_heads, param_x_c.shape[1])).permute([0,2,1])
            else:
                param_x_c = param_x_c.reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[1]//self.num_heads))        
            norm_c = torch.sqrt(param_s_c**2 + param_x_c**2)
            param_s_c = param_s_c / norm_c
            param_x_c = param_x_c / norm_c
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)           
            if len(self.batch_t.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)
        else:
            param_a_prim = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_a_prim = param_a_prim.permute([0,2,1]).reshape((param_a_prim.shape[0]*self.num_heads, param_a_prim.shape[2]//self.num_heads, param_a_prim.shape[1])).permute([0,2,1])
            else:
                param_a_prim = param_a_prim.reshape((param_a_prim.shape[0]*self.num_heads, param_a_prim.shape[1]//self.num_heads))
            
            param_s_a = torch.cos(param_a_prim)
            param_x_a = torch.sin(param_a_prim)
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)   

            param_c_prim = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_c_prim = param_c_prim.permute([0,2,1]).reshape((param_c_prim.shape[0]*self.num_heads, param_c_prim.shape[2]//self.num_heads, param_c_prim.shape[1])).permute([0,2,1])
            else:
                param_c_prim = param_c_prim.reshape((param_c_prim.shape[0]*self.num_heads, param_c_prim.shape[1]//self.num_heads))
            
            param_s_c = torch.cos(param_c_prim)
            param_x_c = torch.sin(param_c_prim)
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)  
            if len(self.batch_h.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)            

        # structural_score = self._calc_structural_score(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        # if self.semantic_score_fn == 'pooled':
        #     semantic_score = self._calc_pooled_semantic_score_optim(param_a, param_b, Rs, H, T, set_sizes_h, set_sizes_t)
        # else:    
        #     semantic_score = self._calc_semantic_score_optim(param_a, param_b, Rs, H, T, set_sizes_h, set_sizes_t)
        if self.CUDA:
            H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1).cuda(),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
            T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1).cuda(),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
        else:
            H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
            T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
        if self.num_heads>1:
            score1 = self._calc_semantic_score_optim(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)
        else:
            score1 = self._calc_semantic_score_optim_1head(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)

        
        # delta_x = (compute_tail_x - tail_x)
        # delta_y = (compute_tail_y - tail_y)
        # delta_z = (compute_tail_z - tail_z)
        
        # score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
        # score1 = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
        # score1 = score1.norm(dim = 0)
        
        x = -x
        y = -y
        z = -z
        # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
        # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
        # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
        _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z, rel_bias)

        if self.relation_embedding_has_mod:
            compute_head_x = compute_head_x / denominator
            compute_head_y = compute_head_y / denominator
            compute_head_z = compute_head_z / denominator
        
        if self.CUDA:
            T = torch.cat((torch.zeros(compute_head_x.shape).cuda().unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
            H = torch.cat((torch.zeros(head_x.shape).cuda().unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
        else:
            T = torch.cat((torch.zeros(compute_head_x.shape).unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
            H = torch.cat((torch.zeros(head_x.shape).unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
        if self.num_heads>1:
            score2 = self._calc_semantic_score_optim(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)
        else:
            score2 = self._calc_semantic_score_optim_1head(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rotate_fn=self.rotate_fn)


        # delta_x2 = (compute_head_x - head_x)
        # delta_y2 = (compute_head_y - head_y)
        # delta_z2 = (compute_head_z - head_z)
        
        # score2 = torch.stack([delta_w2, delta_x2, delta_y2, delta_z2], dim = 0)
        # score2 = score2.norm(dim = 0)     
        
        # score1 = score1.mean(dim=2)
        # score2 = score2.mean(dim=2)

   
        score = (score1 + score2) / 2
        
        score = self.gamma.item() - score

        # print(prof.key_averages().table(sort_by="self_gpu_memory_usage", row_limit=100))

        return score, score1, score2

    @staticmethod
    def train_step(model, optimizer, train_iterator, step, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # with LineProfiler(model.HopfParaE, model._calc_semantic_score_optim_1head) as prof:
        negative_score, head_mod, tail_mod = model((positive_sample, negative_sample), mode=mode) # 全是负样本分数 shape: batch_size, neg_size
        # prof.display()

        if step % 500 == 0:
            print(negative_score.mean(), head_mod.mean(), tail_mod.mean())

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            # negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
            #                   * F.logsigmoid(-negative_score)).sum(dim = 1)
            neg_score_wo_gamma = negative_score-args.gamma
            negative_score = (F.softmax(neg_score_wo_gamma * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        # with LineProfiler(model.HopfParaE, model._calc_semantic_score_optim_1head) as prof:
        positive_score, head_mod, tail_mod = model(positive_sample) # 正样本分数 shape: batch_size, 1     
        # prof.display()

        if step % 500 == 0:
            print(positive_score.mean(), head_mod.mean(), tail_mod.mean())

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            # regularization = args.regularization * (
            #     model.entity_x.weight.data.norm(p = 3)**3 + 
            #     model.entity_y.weight.data.norm(p = 3)**3 + 
            #     model.entity_z.weight.data.norm(p = 3)**3 
            # ) / args.batch_size
            regularization = args.regularization * (
                model.entity_x.weight.data.norm(p = 2)**2 + 
                model.entity_y.weight.data.norm(p = 2)**2 + 
                model.entity_z.weight.data.norm(p = 2)**2 
            ) / args.batch_size

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score, head_mod, tail_mod = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
    def get_entity_properties_from_npy(self, batch_indices, pos='head'):
        # pass
        n = batch_indices.shape[0]
        if len(batch_indices.shape)==1:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_h[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_t[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
        else:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)


class HopfParaE2Model(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 entity_embedding_has_mod=False, relation_embedding_has_mod=False, rotate_fn="R1", params=None):
        super(HopfParaE2Model, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 1.2
        self.rel_high_bound = 2.0
        
        self.use_abs_norm = True
        self.allow_minus_mod = True
        self.use_entity_phase = False
        self.use_real_part = False
        
        self.criterion = 'he'
        
        if self.criterion == 'glorot':
            mod_range = 1. / np.sqrt(2 * (self.hidden_dim + self.hidden_dim))
        elif self.criterion == 'he':
            mod_range = 1. / np.sqrt(2 * self.hidden_dim)
        
        if self.allow_minus_mod:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range * 2.]), 
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range]), 
                requires_grad=False
            )
            
        self.gamma1 = nn.Parameter(
            torch.Tensor([(self.rel_high_bound + self.epsilon) * mod_range * self.hidden_dim]), 
            requires_grad=False
        )
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.unit_mod = nn.Parameter(
            torch.Tensor([1.]), 
            requires_grad=False
        )
        
        self.zero_ent_phase = nn.Parameter(
            torch.Tensor([0.]), 
            requires_grad=False
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.entity_embedding_has_mod = entity_embedding_has_mod
        self.relation_embedding_has_mod = relation_embedding_has_mod
                   
        self.entity_x = nn.Embedding(self.nentity, self.hidden_dim)
        self.entity_y = nn.Embedding(self.nentity, self.hidden_dim)   
        self.entity_z = nn.Embedding(self.nentity, self.hidden_dim)
        
        self.relation_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_z = nn.Embedding(self.nrelation, self.hidden_dim)

        self.relation_s3_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_s3_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_s3_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_s3_z = nn.Embedding(self.nrelation, self.hidden_dim)
        
        self.init_weights()
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['HopfParaE2']:
            raise ValueError('model %s not supported' % model_name)
        if self.use_real_part:
            try:
                assert(self.use_abs_norm == True)
            except:
                raise ValueError('use_abs_norm should be true if you only use real part')
        if (not self.entity_embedding_has_mod) and self.relation_embedding_has_mod:
            raise ValueError('when relation has mod, entity must have mod')
        
        if params is None:
            params={    
                'embeddings_path': '../../data/FB15K237_context/limit1_v3/embeddings.npy',
                'char_vocab_path': '../../data/FB15K237_context/limit1_v3/char2idx.json',
                'char_feature_size': 50,
                'char_embed_dim': 50,
                'max_word_len_entity': 10,
                'conv_filter_size': 3,
                'drop_rate': 0.0,
                'max_sent_len': 16,
                'entity_indices_file': '../../data/FB15K237_context/limit1_v3/entity_context_indices.json',
                'word2idx_path': '../../data/FB15K237_context/limit1_v3/word2idx.json',
                'all_word_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/word_indices_h.npy',
                'all_char_token_ids_file_h': '../../data/FB15K237_context/limit1_v3/char_indices_h.npy',
                'mask_file_h': '../../data/FB15K237_context/limit1_v3/mask_h.npy',
                'all_word_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/word_indices_t.npy',
                'all_char_token_ids_file_t': '../../data/FB15K237_context/limit1_v3/char_indices_t.npy',
                'mask_file_t': '../../data/FB15K237_context/limit1_v3/mask_t.npy',
                'padding': 1,
                'checkpoint_json_path': './result_hopfe_rot_2/HopfE.json',
                'num_heads': 1,
                'wassertein_approx': False
            }
        self.embeddings = np.load(params['embeddings_path'])
        with open(params['char_vocab_path'], 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.all_word_token_ids_h = np.load(params['all_word_token_ids_file_h'])
        self.all_char_token_ids_h = np.load(params['all_char_token_ids_file_h'])
        self.all_word_token_ids_t = np.load(params['all_word_token_ids_file_t'])
        self.all_char_token_ids_t = np.load(params['all_char_token_ids_file_t'])
        self.PADDING = params['padding']
        self.max_sent_len = params['max_sent_len']
        self.conv_filter_size = params['conv_filter_size']
        self.max_char_len = params['max_word_len_entity']
        self.char_embed_dim = params['char_embed_dim']
        self.drop_rate = params['drop_rate']
        self.mask_h = np.load(params['mask_file_h'])
        self.mask_t = np.load(params['mask_file_t'])
        self.char_feature_size = params['char_feature_size']

        self.semanticE = CbowE({'embeddings': self.embeddings, 
            'char_vocab': self.char_vocab, 
            'char_feature_size': self.char_feature_size, 
            'char_embed_dim': self.char_embed_dim, 
            'max_word_len_entity': self.max_char_len, 
            'conv_filter_size': self.conv_filter_size, 
            'drop_rate': self.drop_rate})
        self.linear_proj = nn.Linear(50, 2*self.hidden_dim)


        self.num_heads = params['num_heads']
        self.param_ent_s_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)
        self.param_ent_x_a = nn.Embedding(self.nentity, self.hidden_dim*self.num_heads)
        self.CUDA = torch.cuda.is_available()

        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
        self.wassertein_approx = params.get('wassertein_approx', False)
        self.single_param = False

        self.rotate_fn = rotate_fn

    def init_weights(self):

        rel_w, rel_x, rel_y, rel_z, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z = self.relation_init(self.nrelation, self.hidden_dim)
        rel_w, rel_x, rel_y, rel_z, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z = torch.from_numpy(rel_w), torch.from_numpy(rel_x), torch.from_numpy(rel_y), torch.from_numpy(rel_z), torch.from_numpy(rel_s3_w), torch.from_numpy(rel_s3_x), torch.from_numpy(rel_s3_y), torch.from_numpy(rel_s3_z)
        self.relation_w.weight.data = rel_w.type_as(self.relation_w.weight.data)
        self.relation_x.weight.data = rel_x.type_as(self.relation_x.weight.data)
        self.relation_y.weight.data = rel_y.type_as(self.relation_y.weight.data)
        self.relation_z.weight.data = rel_z.type_as(self.relation_z.weight.data)
        self.relation_s3_w.weight.data = rel_s3_w.type_as(self.relation_s3_w.weight.data)
        self.relation_s3_x.weight.data = rel_s3_x.type_as(self.relation_s3_x.weight.data)
        self.relation_s3_y.weight.data = rel_s3_y.type_as(self.relation_s3_y.weight.data)
        self.relation_s3_z.weight.data = rel_s3_z.type_as(self.relation_s3_z.weight.data)
        
        e_x, e_y, e_z = self.entity_init(self.nentity, self.hidden_dim)
        e_x, e_y, e_z = torch.from_numpy(e_x), torch.from_numpy(e_y), torch.from_numpy(e_z)
        self.entity_x.weight.data = e_x.type_as(self.entity_x.weight.data)
        self.entity_y.weight.data = e_y.type_as(self.entity_y.weight.data)
        self.entity_z.weight.data = e_z.type_as(self.entity_z.weight.data)
        
    def relation_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features
        
        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init rel_mod is: ', s)

        kernel_shape = (n_entries, features)
            
        rel_mod = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rotate_phase = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=kernel_shape)
        theta = np.random.uniform(low=0, high=np.pi, size=kernel_shape)
        phi = np.random.uniform(low=0, high=2*np.pi, size=kernel_shape)
        
        rel_w = rel_mod * np.cos(rotate_phase/2)
        rel_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
        rel_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
        rel_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

        rel_s3_w = rel_mod * np.cos(rotate_phase/2)
        rel_s3_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
        rel_s3_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
        rel_s3_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

        return rel_w, rel_x, rel_y, rel_z, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z

    def entity_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init x, y, z is: ', s)

        # rng = RandomState(456)
        kernel_shape = (n_entries, features)
            
        x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return x, y, z
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # batch_size, negative_sample_size = sample.size(0), 1
            
            head_x = self.entity_x(sample[:, 0]).unsqueeze(1)
            head_y = self.entity_y(sample[:, 0]).unsqueeze(1)
            head_z = self.entity_z(sample[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(sample[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(sample[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(sample[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(sample[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(sample[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(sample[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(sample[:, 1]).unsqueeze(1)

            rel_s3_w = self.relation_s3_w(sample[:, 1]).unsqueeze(1)
            rel_s3_x = self.relation_s3_x(sample[:, 1]).unsqueeze(1)
            rel_s3_y = self.relation_s3_y(sample[:, 1]).unsqueeze(1)
            rel_s3_z = self.relation_s3_z(sample[:, 1]).unsqueeze(1)

            self.batch_h = sample[:, 0]
            self.batch_r = sample[:, 1]
            self.batch_t = sample[:, 2]
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_x = self.entity_x(head_part)
            head_y = self.entity_y(head_part)
            head_z = self.entity_z(head_part)
            
            tail_x = self.entity_x(tail_part[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(tail_part[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(tail_part[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(tail_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(tail_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(tail_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(tail_part[:, 1]).unsqueeze(1)

            rel_s3_w = self.relation_s3_w(tail_part[:, 1]).unsqueeze(1)
            rel_s3_x = self.relation_s3_x(tail_part[:, 1]).unsqueeze(1)
            rel_s3_y = self.relation_s3_y(tail_part[:, 1]).unsqueeze(1)
            rel_s3_z = self.relation_s3_z(tail_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part
            self.batch_r = tail_part[:, 1]
            self.batch_t = tail_part[:, 2]
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_x = self.entity_x(head_part[:, 0]).unsqueeze(1)
            head_y = self.entity_y(head_part[:, 0]).unsqueeze(1)
            head_z = self.entity_z(head_part[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(tail_part)
            tail_y = self.entity_y(tail_part)
            tail_z = self.entity_z(tail_part)
            
            rel_w = self.relation_w(head_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(head_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(head_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(head_part[:, 1]).unsqueeze(1)

            rel_s3_w = self.relation_s3_w(head_part[:, 1]).unsqueeze(1)
            rel_s3_x = self.relation_s3_x(head_part[:, 1]).unsqueeze(1)
            rel_s3_y = self.relation_s3_y(head_part[:, 1]).unsqueeze(1)
            rel_s3_z = self.relation_s3_z(head_part[:, 1]).unsqueeze(1)

            self.batch_h = head_part[:, 0]
            self.batch_r = head_part[:, 1]
            self.batch_t = tail_part
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'HopfParaE2': self.HopfParaE2
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_x, head_y, head_z, 
                                                rel_w, rel_x, rel_y, rel_z, 
                                                tail_x, tail_y, tail_z, 
                                                rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z,
                                                mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def _quat_mul(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        
        return (A, B, C, D)

    def rotate(self, x, y, z, rel_w, rel_x, rel_y, rel_z):
        A, B, C, D = self._quat_mul(rel_w, rel_x, rel_y, rel_z, 0, x, y, z)
        return self._quat_mul(A, B, C, D, rel_w, -1.0*rel_x, -1.0*rel_y, -1.0*rel_z)

    def _calc_semantic_score_optim_test(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t):
        
        Ah = H[:,:,:,0]
        Bh = H[:,:,:,1]
        Ch = H[:,:,:,2]
        Dh = H[:,:,:,3]

        At = T[:,:,:,0]
        Bt = T[:,:,:,1]
        Ct = T[:,:,:,2]
        Dt = T[:,:,:,3]

        if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
            Ah = Ah.repeat([1,At.shape[1],1])
            Bh = Bh.repeat([1,Bt.shape[1],1])
            Ch = Ch.repeat([1,Ct.shape[1],1])
            Dh = Dh.repeat([1,Dt.shape[1],1])
        elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
            At = At.repeat([1,Ah.shape[1],1])
            Bt = Bt.repeat([1,Bh.shape[1],1])
            Ct = Ct.repeat([1,Ch.shape[1],1])
            Dt = Dt.repeat([1,Dh.shape[1],1])

        # Reshape Ah, Bh, Ch, Dh
        Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2]).contiguous()
        Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2]).contiguous()
        Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2]).contiguous()
        Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2]).contiguous()

        Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2]).contiguous()
        Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2]).contiguous()
        Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2]).contiguous()
        Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2]).contiguous()
        
        # find the pairwise dist
        # dw = torch.cdist(Ahr,Atr).reshape(Ah.shape[0]//self.num_heads,Ah.shape[1],self.num_heads,self.num_heads)
        # dx = torch.cdist(Bhr,Btr).reshape(Bh.shape[0]//self.num_heads,Bh.shape[1],self.num_heads,self.num_heads)
        # dy = torch.cdist(Chr,Ctr).reshape(Ch.shape[0]//self.num_heads,Ch.shape[1],self.num_heads,self.num_heads)
        # dz = torch.cdist(Dhr,Dtr).reshape(Dh.shape[0]//self.num_heads,Dh.shape[1],self.num_heads,self.num_heads)
        d1 = Ahr-Atr
        d2 = Bhr-Btr
        d3 = Chr-Ctr
        d4 = Dhr-Dtr
        d5 = torch.stack([d1,d2,d3,d4],dim=0)
        d6 = d5.norm(dim=0)
        d7 = torch.mean(d6,dim=-1)
        return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

        # # take the sum
        # delta = dw + dx + dy + dz

        # # take the min, min or max, min
        # delta_min1 = torch.min(delta, dim=-1)[0]
        # delta_min2 = torch.min(delta_min1, dim=-1)[0]     


        # return delta_min2

    def _calc_semantic_score_optim(self, Hs, Ts, H, T, set_sizes_h, set_sizes_t, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z, rotate_fn="R1"):

        # single_mode = True
        # if len(Hs.shape)==4:
        #     single_mode = False
        #     neg_samples = Hs.shape[1]
        #     Hs = Hs.reshape((-1,Hs.shape[2],Hs.shape[3]))
        #     set_sizes_h = set_sizes_h * neg_samples
        # if len(Ts.shape)==4:
        #     single_mode = False
        #     neg_samples = Ts.shape[1]
        #     Ts = Ts.reshape((-1,Ts.shape[2],Ts.shape[3]))
        #     set_sizes_t = set_sizes_t * neg_samples

        # s_b = Rs[:,:,0]
        # x_b = Rs[:,:,1]
        # y_b = Rs[:,:,2]
        # z_b = Rs[:,:,3]
        # denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        # s_b = s_b / denominator_b
        # x_b = x_b / denominator_b
        # y_b = y_b / denominator_b
        # z_b = z_b / denominator_b

        H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        
        s_a = H[:,:,:,0]
        x_a = H[:,:,:,1]
        y_a = H[:,:,:,2]
        z_a = H[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2":
            denominator_a = torch.sqrt(s_a ** 2 + x_a ** 2 + y_a ** 2 + z_a ** 2)
            s_a = s_a / denominator_a
            x_a = x_a / denominator_a
            y_a = y_a / denominator_a
            z_a = z_a / denominator_a

        s_c = T[:,:,:,0]
        x_c = T[:,:,:,1]
        y_c = T[:,:,:,2]
        z_c = T[:,:,:,3]
        if rotate_fn=="R1" or rotate_fn=="R2": 
            denominator_c = torch.sqrt(s_c ** 2 + x_c ** 2 + y_c ** 2 + z_c ** 2)
            s_c = s_c / denominator_c
            x_c = x_c / denominator_c
            y_c = y_c / denominator_c
            z_c = z_c / denominator_c

        # Rs0 = torch.cat((s_b.unsqueeze(-1),x_b.unsqueeze(-1),y_b.unsqueeze(-1),z_b.unsqueeze(-1)), dim=-1)

        if self.CUDA:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],Hs.shape[2],2) ).cuda() ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],Ts.shape[2],2) ).cuda() ), dim=-1)
        else:
            Hs = torch.cat(( Hs, torch.zeros( (Hs.shape[0],Hs.shape[1],2) ) ), dim=-1)
            Ts = torch.cat(( Ts, torch.zeros( (Ts.shape[0],Ts.shape[1],2) ) ), dim=-1)
        # Convert from structural 3D space to semantic 4D space using the reverse Hopf map
        # H = torch.repeat_interleave(H, set_sizes_h, dim=0)
        # A, B, C, D = self._quat_mul(H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # A, B, C, D = self._quat_mul(H[:,:,0], 1+H[:,:,1], H[:,:,2], H[:,:,3], Hs[:,:,0], Hs[:,:,1], Hs[:,:,2], Hs[:,:,3])
        # num = torch.sqrt(H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)
        # den = torch.sqrt(1+2*H[:,:,1] + H[:,:,1]**2 + H[:,:,2]**2 + H[:,:,3]**2)

        assert s_a.eq(0).all().cpu().numpy(), "s_a must be equal to 0"
        # assert (x_a**2 + y_a**2 + z_a**2).eq(1).all().cpu().numpy(), "(x_a**2 + y_a**2 + z_a**2) must be equal to 1"

        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_a/(1+x_a+epsilon_xa), y_a/(1+x_a+epsilon_xa), Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2) 
            den = np.sqrt(2.)      
        else:
            A, B, C, D = self._quat_mul(s_a, 1+x_a, y_a, z_a, Hs[:,:,:,0], Hs[:,:,:,1], Hs[:,:,:,2], Hs[:,:,:,3])
            num = torch.sqrt(x_a**2 + y_a**2 + z_a**2)
            den = torch.sqrt(1+2*x_a + x_a**2 + y_a**2 + z_a**2)         
        norm = torch.div(num,den)
        # Ah = A*norm
        # Bh = B*norm
        # Ch = C*norm
        # Dh = D*norm
        if rotate_fn=="R1" or rotate_fn=="R2":
            Ah = A*norm*denominator_a
            Bh = B*norm*denominator_a
            Ch = C*norm*denominator_a
            Dh = D*norm*denominator_a   
        else:
            Ah = A*norm
            Bh = B*norm
            Ch = C*norm
            Dh = D*norm              
        # Hs[:,:,0] = A
        # Hs[:,:,1] = B
        # Hs[:,:,2] = C
        # Hs[:,:,3] = D
        Hs0 = torch.cat((Ah.unsqueeze(-1),Bh.unsqueeze(-1),Ch.unsqueeze(-1),Dh.unsqueeze(-1)), dim=-1)

        # T = torch.repeat_interleave(T, set_sizes_t, dim=0)
        # A, B, C, D = self._quat_mul(T[:,:,0], T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # A, B, C, D = self._quat_mul(T[:,:,0], 1+ T[:,:,1], T[:,:,2], T[:,:,3], Ts[:,:,0], Ts[:,:,1], Ts[:,:,2], Ts[:,:,3])
        # num = torch.sqrt(T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        # den = torch.sqrt(1+2*T[:,:,1] + T[:,:,1]**2 + T[:,:,2]**2 + T[:,:,3]**2)
        if rotate_fn=="R1":
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        elif rotate_fn=="R2":
            epsilon_xa = 1e-8*np.random.random()
            A, B, C, D = self._quat_mul(1., 0., -z_c/(1+x_c+epsilon_xa), y_c/(1+x_c+epsilon_xa), Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2) 
            den = np.sqrt(2.) 
        else:
            A, B, C, D = self._quat_mul(s_c, 1+ x_c, y_c, z_c, Ts[:,:,:,0], Ts[:,:,:,1], Ts[:,:,:,2], Ts[:,:,:,3])
            num = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
            den = torch.sqrt(1+2*x_c + x_c**2 + y_c**2 + z_c**2)
        norm = torch.div(num,den)
        # At = A*norm
        # Bt = B*norm
        # Ct = C*norm
        # Dt = D*norm 
        if rotate_fn=="R1" or rotate_fn=="R2":  
            At = A*norm*denominator_c
            Bt = B*norm*denominator_c
            Ct = C*norm*denominator_c
            Dt = D*norm*denominator_c      
        else:
            At = A*norm
            Bt = B*norm
            Ct = C*norm
            Dt = D*norm     
        # Ts[:,:,0] = A
        # Ts[:,:,1] = B
        # Ts[:,:,2] = C
        # Ts[:,:,3] = D
        Ts0 = torch.cat((At.unsqueeze(-1),Bt.unsqueeze(-1),Ct.unsqueeze(-1),Dt.unsqueeze(-1)), dim=-1)

        Ah, Bh, Ch, Dh = self._quat_mul(Ah, Bh, Ch, Dh, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z)
        # At, Bt, ct, Dt = self._quat_mul(At, Bt, Ct, Dt, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z)

        if self.wassertein_approx:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])

            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            Hs1 = torch.cat([Ahr,Bhr,Chr,Dhr], dim=-1)
            Ts1 = torch.cat([Atr,Btr,Ctr,Dtr], dim=-1)

            # find the wass dist
            dist, P, C = self.sinkhorn(Hs1, Ts1)

            # dist = dist.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])    
            delta_min1 = torch.min(P, dim=-1)
            dr3 = delta_min1[1].unsqueeze(-1).repeat([1,1,self.hidden_dim])

            # Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            # Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3)
            Btrm = torch.gather(Btr,dim=1,index=dr3)
            Ctrm = torch.gather(Ctr,dim=1,index=dr3)
            Dtrm = torch.gather(Dtr,dim=1,index=dr3)

            d1 = Ahr-Atrm
            d2 = Bhr-Btrm
            d3 = Chr-Ctrm
            d4 = Dhr-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            d7 = torch.mean(d7,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])

            # return dist
        else:
            if Ah.shape[1]==1 and Ah.shape[1]<At.shape[1]:
                Ah = Ah.repeat([1,At.shape[1],1])
                Bh = Bh.repeat([1,Bt.shape[1],1])
                Ch = Ch.repeat([1,Ct.shape[1],1])
                Dh = Dh.repeat([1,Dt.shape[1],1])
            elif At.shape[1]==1 and At.shape[1]<Ah.shape[1]:
                At = At.repeat([1,Ah.shape[1],1])
                Bt = Bt.repeat([1,Bh.shape[1],1])
                Ct = Ct.repeat([1,Ch.shape[1],1])
                Dt = Dt.repeat([1,Dh.shape[1],1])

            # Reshape Ah, Bh, Ch, Dh
            Ahr = Ah.unsqueeze(dim=1).reshape(Ah.shape[0]//self.num_heads,self.num_heads,Ah.shape[1],Ah.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ah.shape[2])
            Bhr = Bh.unsqueeze(dim=1).reshape(Bh.shape[0]//self.num_heads,self.num_heads,Bh.shape[1],Bh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bh.shape[2])
            Chr = Ch.unsqueeze(dim=1).reshape(Ch.shape[0]//self.num_heads,self.num_heads,Ch.shape[1],Ch.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ch.shape[2])
            Dhr = Dh.unsqueeze(dim=1).reshape(Dh.shape[0]//self.num_heads,self.num_heads,Dh.shape[1],Dh.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dh.shape[2])

            Atr = At.unsqueeze(dim=1).reshape(At.shape[0]//self.num_heads,self.num_heads,At.shape[1],At.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,At.shape[2])
            Btr = Bt.unsqueeze(dim=1).reshape(Bt.shape[0]//self.num_heads,self.num_heads,Bt.shape[1],Bt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Bt.shape[2])
            Ctr = Ct.unsqueeze(dim=1).reshape(Ct.shape[0]//self.num_heads,self.num_heads,Ct.shape[1],Ct.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Ct.shape[2])
            Dtr = Dt.unsqueeze(dim=1).reshape(Dt.shape[0]//self.num_heads,self.num_heads,Dt.shape[1],Dt.shape[2]).permute([0,2,1,3]).reshape(-1,self.num_heads,Dt.shape[2])
            
            # find the pairwise dist
            # dw = torch.cdist(Ahr,Atr).reshape(Ah.shape[0]//self.num_heads,Ah.shape[1],self.num_heads,self.num_heads)
            # dx = torch.cdist(Bhr,Btr).reshape(Bh.shape[0]//self.num_heads,Bh.shape[1],self.num_heads,self.num_heads)
            # dy = torch.cdist(Chr,Ctr).reshape(Ch.shape[0]//self.num_heads,Ch.shape[1],self.num_heads,self.num_heads)
            # dz = torch.cdist(Dhr,Dtr).reshape(Dh.shape[0]//self.num_heads,Dh.shape[1],self.num_heads,self.num_heads)
            dw = torch.cdist(Ahr,Atr)
            dx = torch.cdist(Bhr,Btr)
            dy = torch.cdist(Chr,Ctr)
            dz = torch.cdist(Dhr,Dtr)
                       
            delta = dw + dx + dy + dz

            # take the min, min or max, min
            delta_min1 = torch.min(delta, dim=-1)
            delta_min2 = torch.min(delta_min1[0], dim=-1)
            dr1 = delta_min1[1]
            dr2 = delta_min2[1].unsqueeze(-1)
            # dr3 = torch.cat((dr2,dr1[dr2]), dim=-1)
            dr1g = torch.gather(dr1,dim=1,index=dr2)
            dr3 = torch.cat((dr2,dr1g), dim=-1)

            Ahrm = torch.gather(Ahr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Bhrm = torch.gather(Bhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Chrm = torch.gather(Chr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dhrm = torch.gather(Dhr,dim=1,index=dr3[:,0].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Atrm = torch.gather(Atr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Btrm = torch.gather(Btr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Ctrm = torch.gather(Ctr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))
            Dtrm = torch.gather(Dtr,dim=1,index=dr3[:,1].unsqueeze(-1).unsqueeze(-1).repeat([1,1,self.hidden_dim]))

            d1 = Ahrm-Atrm
            d2 = Bhrm-Btrm
            d3 = Chrm-Ctrm
            d4 = Dhrm-Dtrm
            d5 = torch.stack([d1,d2,d3,d4],dim=0)
            d6 = d5.norm(dim=0)
            d7 = torch.mean(d6,dim=-1)
            return d7.reshape(Ah.shape[0]//self.num_heads,Ah.shape[1])
            # # take the sum
            # delta = dw + dx + dy + dz

            # # take the min, min or max, min
            # delta_min1 = torch.min(delta, dim=-1)[0]
            # delta_min2 = torch.min(delta_min1, dim=-1)[0]     


            # return delta_min2

        # Hs0 = Hs0.unsqueeze(dim=1).reshape(Hs0.shape[0]//self.PADDING,self.PADDING,Hs0.shape[1],Hs0.shape[2],Hs.shape[3])
        # Ts0 = Ts0.unsqueeze(dim=1).reshape(Ts0.shape[0]//self.PADDING,self.PADDING,Ts0.shape[1],Ts0.shape[2],Hs.shape[3])

        # Hs0 = torch.mean(Hs0, dim=1)
        # Ts0 = torch.mean(Ts0, dim=1)

        # delta_w = Hs0[:,:,:,0] - Ts0[:,:,:,0]
        # delta_x = Hs0[:,:,:,1] - Ts0[:,:,:,1]
        # delta_y = Hs0[:,:,:,2] - Ts0[:,:,:,2]
        # delta_z = Hs0[:,:,:,3] - Ts0[:,:,:,3]

        # if not single_mode:
        #     delta_w = delta_w.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_x = delta_x.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_y = delta_y.unsqueeze(-1).reshape((-1,neg_samples))
        #     delta_z = delta_z.unsqueeze(-1).reshape((-1,neg_samples))

        # return delta_w, delta_x, delta_y, delta_z
        # Rs = torch.repeat_interleave(Rs0, set_sizes_h, dim=0)
        # # Rs = Rs0
        # # Rotate in 4-D using the relation quaternion
        # A, B, C, D = self._quat_mul(Hs0[:,:,0], Hs0[:,:,1], Hs0[:,:,2], Hs0[:,:,3], Rs[:,:,0], Rs[:,:,1], Rs[:,:,2], Rs[:,:,3])
        # # Hs[:,:,0] = A
        # # Hs[:,:,1] = B
        # # Hs[:,:,2] = C
        # # Hs[:,:,3] = D
        # Hs1 = torch.cat((A.unsqueeze(-1),B.unsqueeze(-1),C.unsqueeze(-1),D.unsqueeze(-1)), dim=-1)

        # '''# Repeat elements along Hs t times
        # # Repeat set elements along Ts h times
        # set_sizes_h2 = torch.repeat_interleave(set_sizes_t, set_sizes_h)
        # set_sizes_t2 = torch.repeat_interleave(set_sizes_h, set_sizes_t)
        # Hs2 = torch.repeat_interleave(Hs1, set_sizes_h2, dim=0)
        # Ts2 = torch.repeat_interleave(Ts0, set_sizes_t2, dim=0)
        # gather_batch_indices = []
        # cur_batch_indices = []
        # j = 0
        # for i in range(Ts0.shape[0]):
        #     cur_batch_indices.append(i)
        #     if len(cur_batch_indices)==set_sizes_t[j]:
        #         cur_batch_indices = cur_batch_indices*set_sizes_h[j]
        #         gather_batch_indices.extend(cur_batch_indices)
        #         cur_batch_indices = []
        #         j += 1
        # if self.CUDA:
        #     gather_batch_indices = torch.tensor(gather_batch_indices).cuda()
        # else:
        #     gather_batch_indices = torch.tensor(gather_batch_indices)
        #     # gather_indices = torch.ones(Ts2.shape)*gather_batch_indices 
        # gather_indices = gather_batch_indices.unsqueeze(-1).unsqueeze(-1).repeat([1,Ts2.shape[1],Ts2.shape[2]])
        # Ts2 = torch.gather(Ts2, 0, gather_indices)
        # set_sizes_gather = set_sizes_h*set_sizes_t'''
        # Ts2 = Ts0
        # Hs2 = Hs1

        # score_r = (Hs2[:,:,0] * Ts2[:,:,0] + Hs2[:,:,1] * Ts2[:,:,1] + Hs2[:,:,2] * Ts2[:,:,2] + Hs2[:,:,3] * Ts2[:,:,3])
        # score_r = -torch.sum(score_r, -1)
        # score_r = score_r.view([score_r.shape[0]//self.PADDING,self.PADDING]).unsqueeze(1)
        # pooled_score = torch.nn.MaxPool1d(self.PADDING, stride=self.PADDING)(score_r).squeeze()
        # return pooled_score

    def HopfParaE2(self, head_x, head_y, head_z, 
                   rel_w, rel_x, rel_y, rel_z, 
                   tail_x, tail_y, tail_z, 
                   rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z,
                   mode):
        pi = 3.14159265358979323846
        assert(self.use_entity_phase == False)
        assert(self.use_real_part == False)
        
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
        
        # compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
        # compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
        # compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
        _, compute_tail_x, compute_tail_y, compute_tail_z = self.rotate(head_x, head_y, head_z, w, x, y, z)

        if self.relation_embedding_has_mod:
            compute_tail_x = denominator * compute_tail_x
            compute_tail_y = denominator * compute_tail_y
            compute_tail_z = denominator * compute_tail_z

        if not self.single_param:
            param_s_a = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_s_a = param_s_a.permute([0,2,1]).reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[2]//self.num_heads, param_s_a.shape[1])).permute([0,2,1])
            else:
                param_s_a = param_s_a.reshape((param_s_a.shape[0]*self.num_heads, param_s_a.shape[1]//self.num_heads))
            param_x_a = self.param_ent_x_a(self.batch_h)    
            if len(self.batch_h.shape) == 2:
                param_x_a = param_x_a.permute([0,2,1]).reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[2]//self.num_heads, param_x_a.shape[1])).permute([0,2,1])
            else:
                param_x_a = param_x_a.reshape((param_x_a.shape[0]*self.num_heads, param_x_a.shape[1]//self.num_heads))
            norm_a = torch.sqrt(param_s_a**2 + param_x_a**2)
            param_s_a = param_s_a / norm_a
            param_x_a = param_x_a / norm_a
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)

            param_s_c = self.param_ent_s_a(self.batch_t)
            if len(self.batch_t.shape) == 2:
                param_s_c = param_s_c.permute([0,2,1]).reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[2]//self.num_heads, param_s_c.shape[1])).permute([0,2,1])
            else:
                param_s_c = param_s_c.reshape((param_s_c.shape[0]*self.num_heads, param_s_c.shape[1]//self.num_heads))        
            param_x_c = self.param_ent_x_a(self.batch_t)    
            if len(self.batch_t.shape) == 2:
                param_x_c = param_x_c.permute([0,2,1]).reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[2]//self.num_heads, param_x_c.shape[1])).permute([0,2,1])
            else:
                param_x_c = param_x_c.reshape((param_x_c.shape[0]*self.num_heads, param_x_c.shape[1]//self.num_heads))        
            norm_c = torch.sqrt(param_s_c**2 + param_x_c**2)
            param_s_c = param_s_c / norm_c
            param_x_c = param_x_c / norm_c
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)           
            if len(self.batch_t.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)
        else:
            param_a_prim = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_a_prim = param_a_prim.permute([0,2,1]).reshape((param_a_prim.shape[0]*self.num_heads, param_a_prim.shape[2]//self.num_heads, param_a_prim.shape[1])).permute([0,2,1])
            else:
                param_a_prim = param_a_prim.reshape((param_a_prim.shape[0]*self.num_heads, param_a_prim.shape[1]//self.num_heads))
            
            param_s_a = torch.cos(param_a_prim)
            param_x_a = torch.sin(param_a_prim)
            param_a = torch.cat((param_s_a.unsqueeze(-1), param_x_a.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_h = self.num_heads*torch.ones((self.batch_h.shape[0]), dtype=np.long)
            if len(self.batch_h.shape)==1:
                param_s_a = param_s_a.unsqueeze(dim=1)
                param_x_a = param_x_a.unsqueeze(dim=1)
                param_a = param_a.unsqueeze(dim=1)   

            param_c_prim = self.param_ent_s_a(self.batch_h)
            if len(self.batch_h.shape) == 2:
                param_c_prim = param_c_prim.permute([0,2,1]).reshape((param_c_prim.shape[0]*self.num_heads, param_c_prim.shape[2]//self.num_heads, param_c_prim.shape[1])).permute([0,2,1])
            else:
                param_c_prim = param_c_prim.reshape((param_c_prim.shape[0]*self.num_heads, param_c_prim.shape[1]//self.num_heads))
            
            param_s_c = torch.cos(param_c_prim)
            param_x_c = torch.sin(param_c_prim)
            param_b = torch.cat((param_s_c.unsqueeze(-1), param_x_c.unsqueeze(-1)), dim=-1)
            if self.CUDA:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long).cuda()
            else:
                set_sizes_t = self.num_heads*torch.ones((self.batch_t.shape[0]), dtype=np.long)  
            if len(self.batch_h.shape)==1:
                param_s_c = param_s_c.unsqueeze(dim=1)
                param_x_c = param_x_c.unsqueeze(dim=1)
                param_b = param_b.unsqueeze(dim=1)            

        # structural_score = self._calc_structural_score(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        # if self.semantic_score_fn == 'pooled':
        #     semantic_score = self._calc_pooled_semantic_score_optim(param_a, param_b, Rs, H, T, set_sizes_h, set_sizes_t)
        # else:    
        #     semantic_score = self._calc_semantic_score_optim(param_a, param_b, Rs, H, T, set_sizes_h, set_sizes_t)
        if self.CUDA:
            H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1).cuda(),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
            T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1).cuda(),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
        else:
            H = torch.cat((torch.zeros(compute_tail_x.shape).unsqueeze(-1),compute_tail_x.unsqueeze(-1),compute_tail_y.unsqueeze(-1),compute_tail_z.unsqueeze(-1)), dim=-1)
            T = torch.cat((torch.zeros(tail_x.shape).unsqueeze(-1),tail_x.unsqueeze(-1),tail_y.unsqueeze(-1),tail_z.unsqueeze(-1)), dim=-1)                
        score1 = self._calc_semantic_score_optim(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z, rotate_fn=self.rotate_fn)

        
        # delta_x = (compute_tail_x - tail_x)
        # delta_y = (compute_tail_y - tail_y)
        # delta_z = (compute_tail_z - tail_z)
        
        # score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
        # score1 = torch.stack([delta_w, delta_x, delta_y, delta_z], dim = 0)
        # score1 = score1.norm(dim = 0)
        
        x = -x
        y = -y
        z = -z
        # compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
        # compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
        # compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
        _, compute_head_x, compute_head_y, compute_head_z = self.rotate(tail_x, tail_y, tail_z, w, x, y, z)

        if self.relation_embedding_has_mod:
            compute_head_x = compute_head_x / denominator
            compute_head_y = compute_head_y / denominator
            compute_head_z = compute_head_z / denominator
        
        if self.CUDA:
            T = torch.cat((torch.zeros(compute_head_x.shape).cuda().unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
            H = torch.cat((torch.zeros(head_x.shape).cuda().unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
        else:
            T = torch.cat((torch.zeros(compute_head_x.shape).unsqueeze(-1),compute_head_x.unsqueeze(-1),compute_head_y.unsqueeze(-1),compute_head_z.unsqueeze(-1)), dim=-1)
            H = torch.cat((torch.zeros(head_x.shape).unsqueeze(-1),head_x.unsqueeze(-1),head_y.unsqueeze(-1),head_z.unsqueeze(-1)), dim=-1)                
        score2 = self._calc_semantic_score_optim(param_a, param_b, H, T, set_sizes_h, set_sizes_t, rel_s3_w, rel_s3_x, rel_s3_y, rel_s3_z, rotate_fn=self.rotate_fn)

        # delta_x2 = (compute_head_x - head_x)
        # delta_y2 = (compute_head_y - head_y)
        # delta_z2 = (compute_head_z - head_z)
        
        # score2 = torch.stack([delta_w2, delta_x2, delta_y2, delta_z2], dim = 0)
        # score2 = score2.norm(dim = 0)     
        
        # score1 = score1.mean(dim=2)
        # score2 = score2.mean(dim=2)

    
        score = (score1 + score2) / 2
        
        score = self.gamma.item() - score
            
        return score, score1, score2

    @staticmethod
    def train_step(model, optimizer, train_iterator, step, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score, head_mod, tail_mod = model((positive_sample, negative_sample), mode=mode) # 全是负样本分数 shape: batch_size, neg_size
        
        if step % 500 == 0:
            print(negative_score.mean(), head_mod.mean(), tail_mod.mean())

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, head_mod, tail_mod = model(positive_sample) # 正样本分数 shape: batch_size, 1     

        if step % 500 == 0:
            print(positive_score.mean(), head_mod.mean(), tail_mod.mean())

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_x.weight.data.norm(p = 3)**3 + 
                model.entity_y.weight.data.norm(p = 3)**3 + 
                model.entity_z.weight.data.norm(p = 3)**3 
            ) / args.batch_size

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score, head_mod, tail_mod = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
    def get_entity_properties_from_npy(self, batch_indices, pos='head'):
        # pass
        n = batch_indices.shape[0]
        if len(batch_indices.shape)==1:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_h[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n,self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].reshape((self.PADDING*n, self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1)   )).astype(np.long), \
                self.mask_t[batch_indices.cpu()].reshape((self.PADDING*n, self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
        else:
            if pos=='head':
                return self.all_word_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_h[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)
            elif pos=='tail':
                return self.all_word_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.max_sent_len)).astype(np.long), \
                self.all_char_token_ids_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n,batch_indices.shape[1],self.conv_filter_size-1 + self.max_sent_len*(self.max_char_len+self.conv_filter_size-1))).astype(np.long), \
                self.mask_t[batch_indices.cpu()].transpose([0,2,1,3]).reshape((self.PADDING*n, batch_indices.shape[1], self.max_sent_len)).astype(np.long), \
                np.array([self.PADDING]*batch_indices.shape[0]).astype(np.long)



# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        if torch.cuda.is_available():
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()
        else:
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / x_points).squeeze()
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / y_points).squeeze()

        if torch.cuda.is_available():
            u = torch.zeros_like(mu).cuda()
            v = torch.zeros_like(nu).cuda()
        else:
            u = torch.zeros_like(mu)
            v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1