import torch
import logging
from konlpy.tag import Mecab
from CBCL.beam import Beam
from CBCL.utils import get_segment_ids_vaild_len, gen_attention_mask, move_to_device
import nltk.translate.bleu_score as bleu
from chatspace import ChatSpace
from tqdm import tqdm

# The skt-kobert tokenizer is used to evaluate korean model's performance
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(self, args, opt):
        self.spacer = ChatSpace()
        self.args = args
        self.opt = opt

        self.bert_tokenizer = opt['tokenizer']

        self.sos_token = opt['dataloader'].init_token
        self.eos_token = opt['dataloader'].eos_token
        self.pad_token = opt['dataloader'].pad_token

        self.sos_token_idx = opt['dataloader'].init_token_idx
        self.eos_token_idx = opt['dataloader'].eos_token_idx
        self.pad_token_idx = opt['dataloader'].pad_token_idx

    def get_data(self):
        test_data_path = self.args.path_to_data+'/'+self.args.test_data
        sources = []
        references = []

        with open(test_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split('\t')
                que, ans = data[0].strip(), data[1].strip()
                sources.append(que)
                references.append(ans)

        gold_data = {'src': sources, 'tgt': references}
        return gold_data

    def greedy_search(self, model):
        logger.info("Evaluating greedy")

        gold_data = self.get_data()
        greedy_hypothesis = []

        for idx in tqdm(range(len(gold_data['src']))):

            src = gold_data['src'][idx]
            tgt = []
            src = self.tokenize(src, type='source')
            tgt = self.tokenize(tgt, type='target')

            for i in range(self.args.max_len):
                pred = []
                y_pred, _ = model(src, tgt)
                y_pred_ids = y_pred.max(dim=-1)[1]
                next_word = y_pred_ids.data[i]
                next_symbol = next_word.item()

                if next_symbol == self.eos_token_idx:
                    y_pred_ids = y_pred_ids.squeeze(0).cpu()
                    for idx in range(len(y_pred_ids)):
                        if y_pred_ids[idx] == self.eos_token_idx:
                            pred = list([pred[x].numpy().tolist() for x in range(len(pred))])
                            pred = self.bert_tokenizer.convert_ids_to_tokens(pred)
                            pred_sentence = "".join(pred).replace('_', ' ')
                            greedy_hypothesis.append(self.spacer.space(pred_sentence))
                            break
                        else:pred.append(y_pred_ids[idx])

                    break
                else:
                    try:tgt[0][i + 1] = y_pred_ids[i]
                    except IndexError:
                        if i + 1 == self.args.max_len:
                            greedy_hypothesis.append('-1')

        assert len(gold_data['src']) == len(greedy_hypothesis)
        self.predicted_data_writing(gold_data, greedy_hypothesis, name='greedy')

    def beam_search(self, model):
        logger.info("Evaluating beam")

        gold_data = self.get_data()
        beam_hypothesis = []

        beam_size = self.args.beam_size

        for idx in tqdm(range(len(gold_data['src']))):
            beam = Beam(beam_size=beam_size,
                        opt=self.opt,
                        start_token_id=self.sos_token_idx,
                        end_token_id=self.eos_token_idx, )

            src = gold_data['src'][idx]
            src = self.tokenize(src, type='source')

            for i in range(beam_size):
                src = torch.cat([src, src], dim=0)
            src = src[:beam_size, :]

            for i in range(self.args.max_len):

                # finish to search
                if beam.top_sentence_ended:
                    max_score_idx = beam.finished_beam_score.index(max(beam.finished_beam_score))
                    result = beam.next_ys[max_score_idx]
                    result_sen = self.bert_tokenizer.convert_ids_to_tokens(result.data.tolist())
                    beam_hypothesis.append(self.spacer.space(''.join(result_sen[1:]).replace(self.eos_token, '')))
                    break

                # search
                if i == 0:
                    new_inputs = beam.get_current_state().unsqueeze(1)

                decoder_outputs, _ = model(src, new_inputs.to(self.args.device))
                new_inputs = move_to_device(beam.advance(decoder_outputs.squeeze(1), cur_idx=i),
                                            self.args.device)

                # ex) beam.finished_beam = [T, F, T, T, T]
                if i+1 == self.args.max_len and beam.top_sentence_ended == False:
                    if beam.finished_beam.count(True) >= 1:
                        where_False = [ step for step, val in enumerate(beam.finished_beam) if val == False]
                        beam.finished_beam_score = [torch.tensor(0).to(self.args.device) if step in where_False else score for step, score in enumerate(beam.finished_beam_score)]

                        max_score_idx = beam.finished_beam_score.index(max(beam.finished_beam_score))
                        result = beam.next_ys[max_score_idx]
                        result_sen = self.bert_tokenizer.convert_ids_to_tokens(result.squeeze(0).data.tolist())
                        beam_hypothesis.append(self.spacer.space(''.join(result_sen[1:]).replace(self.eos_token, '')))
                        break
                    else:
                        beam_hypothesis.append('-1')

        assert len(gold_data['src']) == len(beam_hypothesis)
        self.predicted_data_writing(gold_data, beam_hypothesis, name='beam')

    def calc_bleu_score(self):
        references = self.get_data()
        references = references['tgt']

        references_list = []
        greedy_hypothesis_list = []
        beam_hypothesis_list = []

        greedy_path = self.args.path_to_sorted+'/'+'greedy.txt'
        beam_path = self.args.path_to_sorted+'/'+'beam.txt'

        with open(greedy_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split('\t')
                token = ' '.join(self.bert_tokenizer(data[1].strip()))
                greedy_hypothesis_list.append(token)

        with open(beam_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split('\t')
                token = ' '.join(self.bert_tokenizer(data[1].strip()))
                beam_hypothesis_list.append(token)

        for idx in range(len(references)):
            token = ' '.join(self.bert_tokenizer(references[idx].strip()))
            references_list.append(token)

        # Calculate BLEU Score
        for i in range(2):
            bleu_score = 0
            if i == 0:
                for idx in range(len(references_list)):
                    candidate = greedy_hypothesis_list[idx].split(' ')
                    refer = references_list[idx].split(' ')
                    bleu_score += bleu.sentence_bleu([refer], candidate, weights=(1, 0, 0, 0))
                print("avg greedy bleu score >", bleu_score / len(references_list))
            else:
                for idx in range(len(references_list)):
                    candidate = beam_hypothesis_list[idx].split(' ')
                    refer = references_list[idx].split(' ')
                    bleu_score += bleu.sentence_bleu([refer], candidate, weights=(1, 0, 0, 0))
                print("avg beam bleu score >", bleu_score / len(references_list))

    def tokenize(self, text, type):

        if type == 'source':
            tokens = self.bert_tokenizer.tokenize(text)
            tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        else:
            tokens = [self.sos_token_idx]

        for idx in range(self.args.max_len - len(tokens)):
            tokens += [self.pad_token_idx]

        return torch.tensor([tokens]).to(self.args.device)

    def predicted_data_writing(self, gold, p_data, name='noname'):
        logger.info("Writing result file to server")
        path = self.args.path_to_sorted+'/'+name+'.txt'

        with open(path, 'w', encoding='utf-8') as w:
            for i in tqdm(range(len(gold['src']))):
                sentence = gold['src'][i] + '\t' + p_data[i]+'\n'
                w.write(sentence)

