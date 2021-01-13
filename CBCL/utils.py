import torch
import logging
from torch import Tensor
from chatspace import ChatSpace

spacer = ChatSpace()
logger = logging.getLogger(__name__)


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def cal_acc(yhat: 'model_output', y: 'label', padding_idx) -> Tensor:
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1]  # [0]: max value, [1]: index of max value
        acc = (yhat == y).float().mean()
    return acc


def epoch_time(start_time, end_time) -> (int, int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def do_well_train(inputs, target, outputs, obj):
    pad_token = obj['dataloader'].pad_token
    eos_token = obj['dataloader'].eos_token

    outputs = outputs.max(dim=-1)[1]
    outputs = outputs[0:obj['args'].max_len].squeeze().tolist()

    inputs_sentence_list = obj['tokenizer'].convert_ids_to_tokens(inputs[0])
    target_sentence_list = obj['tokenizer'].convert_ids_to_tokens(target[0])
    output_sentence_list = obj['tokenizer'].convert_ids_to_tokens(outputs)

    input = ''.join(inputs_sentence_list).replace(pad_token, '').replace('_', ' ')
    target = ''.join(target_sentence_list).replace(pad_token, '').replace(eos_token, '').replace('_', ' ')
    output = ''.join(output_sentence_list).replace(pad_token, '').replace('_', ' ')

    output_idx = outputs.find(eos_token)
    output = output[:output_idx]

    print("input> ", input, "\n")
    print("refer> ", target, "\n")
    print(f"outputs> ", output)
    print("----------------------------------------------------")
    exit()
    print("input> ", spacer.space(input), "\n")
    print("refer> ", spacer.space(target), "\n")
    print(f"outputs> ", spacer.space(output))
    print("----------------------------------------------------")


def get_segment_ids_vaild_len(inputs, pad_token_idx, args) -> (Tensor, int):
    v_len_list = [0] * len(inputs)

    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j] == pad_token_idx : break
            else : v_len_list[i] += 1

    segment_ids = torch.zeros_like(inputs).long().to(args.device)
    valid_length = torch.tensor(v_len_list, dtype=torch.int32)
    return segment_ids, valid_length


def gen_attention_mask(token_ids, valid_length) -> Tensor:
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length) : attention_mask[i][:v] = 1
    return attention_mask.float()


def concat_pad(dec_inputs, pad_token_idx):
    for idx in range(len(dec_inputs)):
        temp = dec_inputs[idx][1:]
        temp = torch.cat([temp, pad_token_idx], dim=-1)
        dec_inputs[idx] = temp

    return dec_inputs