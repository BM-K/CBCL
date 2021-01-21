import time
import math
import copy
import torch
from apex import amp
import torch.nn as nn
from torch import Tensor
from CBCL.generation import inference
from CBCL.evaluation import Evaluation
from CBCL.arper import ARPER, EWC
from data.dataloader import DataLoader
from tensorboardX import SummaryWriter
from CBCL.model.encoder_decoder import Transformer
from CBCL.setting import set_args, set_logger, set_seed, print_args
from CBCL.model.functions import get_loss_func, get_optim, get_scheduler
from CBCL.utils import concat_pad, get_lr, cal_acc, epoch_time, get_segment_ids_vaild_len, gen_attention_mask, do_well_train

iteration = 0
summary = SummaryWriter()


def system_setting():
    args = set_args()
    print_args(args)
    set_seed(args)
    return args


def train(model: nn.Module, objectives, args, lambda_base=100000) -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    train_ppl = 0

    if objectives['args'].arper_train == 'True':
        logger.info("Training EWC")

        adaptive_regularization = lambda_base# * math.sqrt(
            #args.exemplars_vocab_size * args.number_of_exemplars / objectives['model'].bert.model.config.vocab_size)

        objectives['arper_loader'] = objectives['dataloader'].train_loader_arper
        objectives['arper_data_len'] = len(objectives['dataloader'].train_data_arper)
        ewc = EWC(args, objectives, model)

    logger.info("Training main")
    for step, batch in enumerate(objectives['train_loader']):

        objectives['optimizer'].zero_grad()

        input_sentence = batch.source
        decoder_sentence = batch.target
        target_sentence = copy.deepcopy(decoder_sentence)

        # remove [CLS] token
        target_sentence = concat_pad(target_sentence, torch.tensor([objectives['dataloader'].pad_token_idx]).to(args.device))

        segment_ids, valid_len = get_segment_ids_vaild_len(input_sentence, objectives['dataloader'].pad_token_idx, args)
        attention_mask = gen_attention_mask(input_sentence, valid_len)
        bert_opt = {'segment_ids': segment_ids,
                    'inputs': input_sentence,
                    'attention_mask': attention_mask}

        outputs = model(bert_opt, decoder_sentence, objectives)

        # outputs : [batch x max_len , vocab_size]
        # target_sentence.view(-1) : [batch x max_len]

        loss = objectives['criterion'](outputs, target_sentence.view(-1))
        if objectives['args'].arper_train == 'True':
            ewc_loss = adaptive_regularization * ewc.penalty(model)
            loss += ewc_loss

        if args.fp16 == 'True':
            with amp.scale_loss(loss, objectives['optimizer']) as scaled_loss : scaled_loss.backward()
        else: loss.backward()

        objectives['optimizer'].step()
        # objectives['scheduler'].step()

        total_loss += loss
        iter_num += 1

        train_ppl += math.exp(loss)

    return total_loss.data.cpu().numpy() / iter_num, train_ppl / iter_num


def valid(model: nn.Module, objectives, args) -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    test_ppl = 0
    global iteration

    with torch.no_grad():
        for step, batch in enumerate(objectives['valid_loader']):

            input_sentence = batch.source
            decoder_sentence = batch.target
            target_sentence = copy.deepcopy(decoder_sentence)

            # remove [CLS] token
            target_sentence = concat_pad(target_sentence, torch.tensor([objectives['dataloader'].pad_token_idx]).to(args.device))

            segment_ids, valid_len = get_segment_ids_vaild_len(input_sentence, objectives['dataloader'].pad_token_idx,                                               args)
            attention_mask = gen_attention_mask(input_sentence, valid_len)
            bert_opt = {'segment_ids': segment_ids,
                        'inputs': input_sentence,
                        'attention_mask': attention_mask}

            outputs = model(bert_opt, decoder_sentence, objectives)

            loss = objectives['criterion'](outputs, target_sentence.view(-1))
            do_well_train(input_sentence, decoder_sentence, outputs, objectives)

            total_loss += loss
            iter_num += 1

            test_ppl += math.exp(loss)

            if iteration % 10 == 0:
                summary.add_scalar('loss/val_loss', loss.item()/iter_num, iteration)
            else : iteration += 1

    return total_loss.data.cpu().numpy() / iter_num, test_ppl / iter_num


def test(model: nn.Module, objectives, args) -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    test_ppl = 0

    with torch.no_grad():
        for step, batch in enumerate(objectives['test_loader']):

            input_sentence = batch.source
            decoder_sentence = batch.target
            target_sentence = copy.deepcopy(decoder_sentence)

            # remove [CLS] token
            target_sentence = concat_pad(target_sentence, torch.tensor([objectives['dataloader'].pad_token_idx]).to(args.device))

            segment_ids, valid_len = get_segment_ids_vaild_len(input_sentence, objectives['dataloader'].pad_token_idx,                                               args)
            attention_mask = gen_attention_mask(input_sentence, valid_len)
            bert_opt = {'segment_ids': segment_ids,
                        'inputs': input_sentence,
                        'attention_mask': attention_mask}

            outputs = model(bert_opt, decoder_sentence, objectives)

            loss = objectives['criterion'](outputs, target_sentence.view(-1))
            do_well_train(input_sentence, decoder_sentence, outputs, objectives)

            total_loss += loss

            iter_num += 1

            test_ppl += math.exp(loss)

    return total_loss.data.cpu().numpy() / iter_num, test_ppl / iter_num


def main() -> None:
    args = system_setting()

    dataloader = DataLoader(args)

    model = Transformer(args, dataloader)
    criterion = get_loss_func(dataloader.pad_token_idx)
    optimizer = get_optim(args, model)
    #scheduler = get_scheduler(optimizer, args, dataloader.train_loader)

    model.to(args.device)
    criterion.to(args.device)

    if args.fp16 == 'True':
        logger.info('Use Automatic Mixed Precision (AMP)')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    early_stop_check = 0
    best_valid_loss = float('inf')
    sorted_path = args.path_to_sorted+'/baseChar_CL.pt'
    privious_model_path = args.path_to_sorted + '/' + args.previous_model

    objectives = {'train_loader': dataloader.train_loader,
                  'valid_loader': dataloader.valid_loader,
                  'test_loader': dataloader.test_loader,
                  'dataloader': dataloader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  #'scheduler': scheduler,
                  'tokenizer': dataloader.bert_tokenizer,
                  'args': args,
                  'model': model}

    if objectives['args'].arper_train == 'True':
        logger.info("Weight init on previous model")
        model.load_state_dict(torch.load(privious_model_path))

    if args.train_ == 'True':
        logger.info('Start Training')
        for epoch in range(args.epochs):
            start_time = time.time()

            model.train()
            train_loss, train_ppl = train(model, objectives, args)

            model.eval()
            valid_loss, valid_ppl = valid(model, objectives, args)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if early_stop_check == args.patience:
                logger.info("Early stopping")
                break

            if valid_loss < best_valid_loss:
                early_stop_check = 0
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), sorted_path)
                print(f'\n\t## SAVE valid_loss: {valid_loss:.3f} | valid_ppl: {valid_ppl:.3f} ##')
            else : early_stop_check += 1

            print(f'\t==Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s==')
            print(f'\t==Train Loss: {train_loss:.3f} | Train ppl: {train_ppl:.3f}==')
            print(f'\t==Valid Loss: {valid_loss:.3f} | Valid ppl: {valid_ppl:.3f}==')
            print(f'\t==Epoch latest LR: {get_lr(objectives["optimizer"]):.9f}==\n')

    if args.test_ == 'True':
        logger.info("Start Test")

        model = Transformer(args, dataloader)
        optimizer = get_optim(args, model)
        model.to(args.device)

        if args.fp16 == 'True':
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        model.load_state_dict(torch.load(sorted_path))
        model.eval()

        test_loss, test_ppl = test(model, objectives, args)
        print(f'\n\t==Test loss: {test_loss:.3f} | Test ppl: {test_ppl:.3f}==\n')

    # Get exemplar data on current data set
    if args.ARPER == 'True' \
            and args.data_shuffle == 'False' \
            and args.batch_size == 1 \
            and args.epochs == 1:
        logger.info("Start EXEMPLAR")
        # set model.eval() and start train() f

        arper = ARPER(args, objectives, model)
        arper.exemplar()
        arper.get_priority_data()
        arper.store_priority_data()

    # Make new data including previous exemplar data and current data
    if args.ARPER == 'True' and args.exemplars is not None:
        arper = ARPER(args, objectives, model)
        arper.reducing_exemplars()
        arper.concat_data()

    if args.inference == 'True':
        logger.info("Start Inference")

        model = Transformer(args, dataloader)
        optimizer = get_optim(args, model)
        model.to(args.device)

        if args.fp16 == 'True':
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        model.load_state_dict(torch.load(sorted_path))

        model.eval()
        while(1):
            inference(args, model, objectives)

    if args.eval == 'True':

        model = Transformer(args, dataloader)
        optimizer = get_optim(args, model)
        model.to(args.device)

        if args.fp16 == 'True':
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        model.load_state_dict(torch.load(sorted_path))
        model.eval()

        evaluation = Evaluation(args, objectives)
        evaluation.greedy_search(model)
        evaluation.beam_search(model)
        evaluation.calc_bleu_score()


if __name__ == '__main__':
    logger = set_logger()
    main()