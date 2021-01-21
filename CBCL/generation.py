import torch
import logging
from CBCL.beam import Beam
from CBCL.utils import get_segment_ids_vaild_len, gen_attention_mask, move_to_device
#from chatspace import ChatSpace


#spacer = ChatSpace()
logger = logging.getLogger(__name__)


def inference(args, model, opt):
    pad_token = '<pad>'
    pad_token_idx = opt['tgt_vocab'].vocab.stoi[pad_token]

    user_input = input("\n입력하세요 (exit:-1) : ")
    if user_input == '-1': exit()

    tokens = opt['dataloader'].tokenize(user_input)
    tokens = torch.tensor([opt['tgt_vocab'].vocab.stoi[token] for token in tokens])

    for i in range(args.max_len - len(tokens)):
        tokens = torch.cat([tokens, torch.tensor([pad_token_idx])], dim=-1)
    tokens = tokens.unsqueeze(0)
    target = torch.tensor([opt['tgt_vocab'].vocab.stoi['<sos>']])

    for i in range(args.max_len - len(target)):
        target = torch.cat([target, torch.tensor([pad_token_idx])], dim=-1)
    target = target.unsqueeze(0)

    tokens = move_to_device(tokens, args.device)
    target = move_to_device(target, args.device)

    greedy_search(model, opt, tokens, target)
    beam_search(model, opt, tokens)


def greedy_search(model, opt, tokens, target):
    pred = []
    eos_token_idx = opt['dataloader'].eos_token_idx

    for i in range(opt['args'].max_len):

        y_pred = model(tokens, target)
        y_pred_ids = y_pred.max(dim=-1)[1]
        next_word = y_pred_ids.data[i]
        next_symbol = next_word.item()

        if next_symbol == eos_token_idx:
            y_pred_ids = y_pred_ids.squeeze(0).cpu()
            for idx in range(len(y_pred_ids)):
                if y_pred_ids[idx] == eos_token_idx:
                    pred = list([pred[x].numpy().tolist() for x in range(len(pred))])
                    pred = opt['tokenizer'].convert_ids_to_tokens(pred)
                    #pred = [opt['tgt_vocab'].vocab.itos[token] for token in pred]
                    pred_sentence = "".join(pred).replace('_', ' ')
                    print("Greedy Result >> ", pred_sentence)#spacer.space(pred_sentence))
                    break
                else:pred.append(y_pred_ids[idx])
            break
        else:
            try:target[0][i + 1] = y_pred_ids[i]
            except IndexError:break


def beam_search(model, opt, tokens):
    beam_size = opt['args'].beam_size

    beam = Beam(beam_size=beam_size,
                opt=opt,
                start_token_id=opt['dataloader'].init_token_idx,
                end_token_id=opt['tgt_vocab'].eos_token_idx,)

    for i in range(beam_size):
        tokens = torch.cat([tokens, tokens], dim=0)
    tokens = tokens[:beam_size, :]

    for i in range(opt['args'].max_len):

        # finish to search
        if beam.top_sentence_ended:
            max_score_idx = beam.finished_beam_score.index(max(beam.finished_beam_score))
            result = beam.next_ys[max_score_idx]
            #result_sen = [opt['tgt_vocab'].vocab.itos[token] for token in result.data.tolist()]
            result_sen = opt['tokenizer'].convert_ids_to_tokens(result.data.tolist())
            #print(f"Beam Result >> {spacer.space(''.join(result_sen[1:]).replace(opt['dataloader'].eos_token, ''))}")
            print(f"Beam Result >> {''.join(result_sen[1:]).replace(opt['dataloader'].eos_token, '')}")
            break

        # search
        if i == 0:
            new_inputs = beam.get_current_state().unsqueeze(1)

        decoder_outputs = model(tokens, new_inputs.to(opt['args'].device))
        new_inputs = move_to_device(beam.advance(decoder_outputs.squeeze(1), cur_idx=i),
                                    opt['args'].device)
