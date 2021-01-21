import torch
import copy
import torch.nn as nn
from torch import Tensor
from apex import amp
import logging
from CBCL.utils import move_to_device
from CBCL.utils import concat_pad, get_lr, cal_acc, epoch_time, get_segment_ids_vaild_len, gen_attention_mask, do_well_train

logger = logging.getLogger(__name__)


class EWC(nn.Module):
    def __init__(self, args, opt, model):
        super(EWC, self).__init__()
        self.model = model
        self.args = args
        self.opt = opt

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        for step, batch in enumerate(self.opt['arper_loader']):

            self.model.zero_grad()

            input_sentence = batch.source
            decoder_sentence = batch.target
            target_sentence = copy.deepcopy(decoder_sentence)
            target_sentence = concat_pad(target_sentence, torch.tensor([self.opt['dataloader'].pad_token_idx]).to(self.args.device))

            segment_ids, valid_len = get_segment_ids_vaild_len(input_sentence, self.opt['dataloader'].pad_token_idx,
                                                               self.args)
            attention_mask = gen_attention_mask(input_sentence, valid_len)
            bert_opt = {'segment_ids': segment_ids,
                        'inputs': input_sentence,
                        'attention_mask': attention_mask}

            outputs = self.model(bert_opt, decoder_sentence, self.opt)
            loss = self.opt['criterion'](outputs, target_sentence.view(-1))

            if self.args.fp16 == 'True':
                with amp.scale_loss(loss, self.opt['optimizer']) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            for n, p in self.model.named_parameters():
                try:
                    precision_matrices[n].data += p.grad.data ** 2 / self.opt['arper_data_len']
                except AttributeError:
                    continue
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss


class ARPER:
    def __init__(self, args, opt, model, M=500):
        self.args = args
        self.opt = opt
        self.model = model

        self.M = M
        self.priority_list = []
        self.priority_scores = []
        self.dataInexemplars = {}
        self.total_number_of_data = []

        # self.vocab_size = 0

    def reducing_exemplars(self):
        logger.info("Reduce previous exemplars data to current data")
        noe = self.args.number_of_exemplars
        exemplars = []

        if noe == 1:
            exemplars.append(self.args.exemplars)
        else:
            splited_data = self.args.exemplars.split('-')
            assert len(splited_data) == noe
            exemplars = [d_name for d_name in splited_data]

        for d_name in exemplars:self.dataInexemplars[d_name] = []

        totalN = 0
        for idx in range(len(exemplars)):
            path_to_data = self.args.path_to_data+'/'+exemplars[idx]
            with open(path_to_data, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
                for line in lines:
                    self.dataInexemplars[exemplars[idx]] += [line]
            self.total_number_of_data.append(len(self.dataInexemplars[exemplars[idx]]))
            totalN += len(self.dataInexemplars[exemplars[idx]])

        EiList = {}
        for idx in range(len(exemplars)):
            EiList[exemplars[idx]] = int(self.M * self.total_number_of_data[idx]/totalN)
            # slice data num
            self.dataInexemplars[exemplars[idx]] = self.dataInexemplars[exemplars[idx]][:EiList[exemplars[idx]]]

    def concat_data(self):
        path_to_current_data = self.args.path_to_data + '/' + self.args.train_data
        # D[t] U E[1:t-1]
        path_to_store_data = self.args.path_to_data + '/' + 'concated_exemplar_and_cur_data.tsv'
        # E[1:t-1]
        path_to_store_exemdata = self.args.path_to_data + '/' + 'concated_exemplar_data.tsv'

        current_data = []
        for key, val in self.dataInexemplars.items():
            current_data += val

        with open(path_to_store_exemdata, 'w', encoding='utf-8-sig') as w:
            for data in current_data:
                w.write(data)

        with open(path_to_current_data, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                current_data += line

        with open(path_to_store_data, 'w', encoding='utf-8-sig') as w:
            for data in current_data:
                w.write(data)

    def exemplar(self):
        logger.info("Get exemplar data on current task")

        with torch.no_grad():
            for step, batch in enumerate(self.opt['train_loader']):

                input_sentence = batch.source
                decoder_sentence = batch.target
                target_sentence = copy.deepcopy(decoder_sentence)
                target_sentence = concat_pad(target_sentence,torch.tensor([self.opt['dataloader'].pad_token_idx]).to(self.args.device))

                segment_ids, valid_len = get_segment_ids_vaild_len(input_sentence, self.opt['dataloader'].pad_token_idx, self.args)
                attention_mask = gen_attention_mask(input_sentence, valid_len)
                bert_opt = {'segment_ids': segment_ids,
                            'inputs': input_sentence,
                            'attention_mask': attention_mask}

                outputs = self.model(bert_opt, decoder_sentence, self.opt)

                loss = self.opt['criterion'](outputs, target_sentence.view(-1))
                self.priority_scores.append([loss])

        #self.vocab_size = len(self.opt['src_vocab'].vocab.stoi)
        self.priority_list = self.priority_scores

    def get_priority_data(self):
        train_data_path = self.args.path_to_data+'/'+self.args.train_data
        with open(train_data_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for step, line in enumerate(lines):
                self.priority_list[step].append(line)

        self.priority_list.sort(key=lambda x: x[0])

    def store_priority_data(self):
        exemplar_path = self.args.path_to_data+'/'+self.args.train_data.split('.')[0]+'_exemplar'
        exemplar_config_path = self.args.path_to_data + '/' + self.args.train_data.split('.')[0] + '_exemplar_config'
        with open(exemplar_path, 'w', encoding='utf-8-sig') as w:
            for idx in range(len(self.priority_list)):
                # except tensor -> ex) [tensor[8.972], '느낌이 왔어\t사랑의 느낌이길 바라요.\n']
                # -> ['느낌이 왔어\t사랑의 느낌이길 바라요.\n']
                w.write(self.priority_list[idx][1])

        #with open(exemplar_config_path, 'w', encoding='utf-8-sig') as w:
        #    sentence = self.args.train_data.split('.')[0]+"_exemplar's vocab size \t" + str(self.vocab_size)
        #    w.write(sentence)