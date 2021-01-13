import torch
import torch.nn as nn


class Beam:

    def __init__(self, beam_size=5, opt=None,
                 start_token_id=2, end_token_id=3):
        self.opt = opt
        self.beam_size = beam_size

        self.end_token_id = end_token_id
        self.start_token_id = start_token_id

        self.top_sentence_ended = False
        self.isfirst_step = True

        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id).to(opt['args'].device)]
        self.current_scores = torch.FloatTensor(beam_size).zero_()

        self.finished_beam = [False] * self.beam_size
        self.finished_beam_score = [float(0)] * beam_size

        self.softmax = nn.Softmax(dim=-1)

    def advance(self, next_log_probs, cur_idx=0):
        next_log_probs = self.softmax(next_log_probs)

        if self.isfirst_step:
            # first step
            beam_scores = next_log_probs[0]
            top_scores, top_score_ids = beam_scores.topk(k=self.beam_size, dim=0)

            self.current_scores = top_scores

            self.isfirst_step = False
            self.next_ys.append(top_score_ids)

            new_inputs = self.next_ys[0]
            # same target data length, don't need to cal length penalty

            new_inputs = self.continual_concat(self.next_ys)
        else:
            temp_current_score = [1] * self.beam_size
            required_idx = []
            next_pred = []

            for idx in range(len(next_log_probs)):
                if idx % (cur_idx + 1) == cur_idx: required_idx.append((idx))

            for idx in range(len(required_idx)):
                if self.finished_beam[idx] == True:
                    score, ids = 1.0, torch.tensor([self.end_token_id]).to(self.opt['args'].device)
                else:
                    score, ids = next_log_probs[required_idx[idx]].topk(k=1, dim=0)
                    temp_current_score[idx] = score[0]

                next_pred.append(ids)

            next_pred_ = self.continual_concat(next_pred)
            self.next_ys.append(next_pred_)

            new_inputs = self.continual_concat(self.next_ys)

            length_penalty_list = self.cal_length_penalty(new_inputs)
            for i in range(len(self.current_scores)):
                if self.finished_beam[i] == True:
                    continue
                else:
                    # logP/lp + cp
                    self.current_scores[i] = temp_current_score[i] / length_penalty_list[i] + self.current_scores[i]

        for idx, token_idx in enumerate(new_inputs):

            if new_inputs[idx][-1] == self.end_token_id and self.finished_beam[idx] != True:
                self.finished_beam[idx] = True
                self.finished_beam_score[idx] = (self.current_scores[idx])

            # finish to search
            if False not in self.finished_beam:
                self.top_sentence_ended = True
                self.next_ys = self.continual_concat(self.next_ys)
                break

        return new_inputs

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def cal_length_penalty(self, cur_pred_token):
        lp_list = []
        count_elements = [0] * self.beam_size
        for i in range(len(cur_pred_token)):
            for j in range(len(cur_pred_token[i])):
                if cur_pred_token[i][j] != self.end_token_id and cur_pred_token[i][j] != self.start_token_id:
                    count_elements[i] += 1

        for i in range(self.beam_size):
            lp_list.append(self.get_length_penalty(count_elements[i]))

        return lp_list

    def get_length_penalty(self, length, min_length=0, alpha=1.2):
        "Get length penalty because shorter sentences usually have bigger probability."
        return ((min_length + length) / (min_length + 1)) ** alpha

    def continual_concat(self, input):
        new_inputs = input[0]
        for i in range(len(input)):
            if i+1 == len(input):break
            if i>=1:
                new_inputs = torch.cat([new_inputs, input[i + 1].view(-1, 1)], dim=1)
            else:
                new_inputs = torch.cat([input[i].view(-1, 1), input[i+1].view(-1, 1)], dim=1)
        return new_inputs