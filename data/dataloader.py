from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from ETRI_tok.tokenization_etri_eojeol import BertTokenizer

class DataLoader():

    def __init__(self, args):
        super(DataLoader, self).__init__()

        self.bert_tokenizer = \
            BertTokenizer.from_pretrained('./ETRI_KoBERT/003_bert_eojeol_pytorch/vocab.txt', do_lower_case=False)

        sepcial_tokens_dict = {'eos_token': '[EOS]'}
        self.bert_tokenizer.add_special_tokens(sepcial_tokens_dict)

        self.init_token = self.bert_tokenizer.cls_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.eos_token = self.bert_tokenizer.eos_token

        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.eos_token)

        self.train_data = args.train_data
        self.test_data = args.test_data
        self.valid_data = args.val_data

        self.src = data.Field(use_vocab=False,
                              tokenize=self.tokenize,
                              fix_length=args.max_len,
                              init_token=self.init_token_idx,
                              pad_token=self.pad_token_idx,
                              batch_first=True, )

        self.tgt = data.Field(use_vocab=False,
                              tokenize=self.tokenize,
                              init_token=self.init_token_idx,
                              pad_token=self.pad_token_idx,
                              unk_token=self.unk_token_idx,
                              eos_token=self.eos_token_idx,
                              fix_length=args.max_len,
                              batch_first=True, )

        self.train_data, self.test_data, self.valid_data = TabularDataset.splits(
            path=args.path_to_data,
            train=self.train_data,
            validation=self.valid_data,
            test=self.test_data,
            format='tsv',
            fields=[('source', self.src), ('target', self.tgt)],
            skip_header=False)

        self.train_loader, self.valid_loader, self.test_loader = BucketIterator.splits(
            (self.train_data, self.test_data, self.valid_data),
            batch_size=args.batch_size,
            device=args.device,
            shuffle=args.data_shuffle,
            sort=False, )

        # ARPER Data loader
        if args.arper_train == 'True':

            self.src_arper = data.Field(use_vocab=True,
                                        tokenize=self.tokenize,
                                        fix_length=args.max_len,
                                        pad_token=self.pad_token,
                                        batch_first=True, )

            self.tgt_arper = data.Field(use_vocab=True,
                                        tokenize=self.tokenize,
                                        init_token=self.init_token,
                                        pad_token=self.pad_token,
                                        unk_token=self.unk_token,
                                        eos_token=self.eos_token,
                                        fix_length=args.max_len,
                                        batch_first=True, )

            self.train_data_arper = TabularDataset(
                path=args.path_to_data+'/'+args.only_exemplars_data,
                format='tsv',
                fields=[('source', self.src_arper), ('target', self.tgt_arper)],
                skip_header=False)

            # GloVe, word2vec
            if args.iswordemb == 'True':
                vectors = Vectors(name="kr-projected.txt")
                self.src_arper.build_vocab(self.train_data_arper, vectors=vectors, max_size=50000, min_freq=1)
                self.tgt_arper.build_vocab(self.train_data_arper, vectors=vectors, max_size=50000, min_freq=1)
            else:
                self.src_arper.build_vocab(self.train_data_arper, max_size=50000, min_freq=1)
                self.tgt_arper.build_vocab(self.train_data_arper, max_size=50000, min_freq=1)

            self.train_loader_arper = BucketIterator(
                self.train_data_arper,
                batch_size=args.batch_size,
                device=args.device,
                shuffle=args.data_shuffle,
                sort=False, )

    def tokenize(self, text):
        tokens = self.bert_tokenizer.tokenize(text)
        tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        return tokens


if __name__ == "__main__":
    print("__main__ data_loader")