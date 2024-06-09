'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import argparse
import random
from itertools import cycle
from types import SimpleNamespace

import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from bert import BertModel, BertLoraSelfAttention
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)
from evaluation import *
from optimizer import AdamW
from tokenizer import BertTokenizer

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config, enableLora=False):
        super(MultitaskBERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        # raise NotImplementedError
        self.dtype = torch.long
        ### Sentiment Classifier
        self.hidden_size = config.hidden_size
        self.classifier_layer_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_layer_2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_layer_3 = torch.nn.Linear(self.hidden_size, len(config.num_labels))
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        ### Paraphrase detector
        self.classifier_layer_para_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_layer_para_2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_layer_para_3 = torch.nn.Linear(self.hidden_size, 1)
        self.dropout_para_1 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout_para_2 = torch.nn.Dropout(config.hidden_dropout_prob)

        ### Semantic Textual Analysis
        self.classifier_layer_sem_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_layer_sem_2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_layer_sem_3 = torch.nn.Linear(self.hidden_size, 1)
        self.dropout_sem_1 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout_sem_2 = torch.nn.Dropout(config.hidden_dropout_prob)

        ### LoRA Freeze weights
        self.freeze_model()
        # self.replace_multihead_attention_recursion(self.bert)

        ### LoRA parameters
        # enableLora = True
        # self.enableLora = enableLora
        # if(self.enableLora):
        #     self.rank = 1
        #     self.freeze_model(self.bert)
        #     self.add_lora_to_model()



    # def add_lora_to_model(self):
    #     modules = self.bert.named_modules()
    #     for idx, (name, module) in enumerate(modules):
    #         # print("======layer details======", name, idx)
    #         if isinstance(module, nn.Linear) and "lora" not in name:
    #             # print("======layer details======", name, idx, module.in_features, module.out_features)
    #             input_dim = module.in_features
    #             output_dim = module.out_features
    #             lora_layer = LoRA(input_dim, output_dim, self.rank)
    #             module.lora_layer = lora_layer


    def freeze_model(self):
        """Freezes all layers except the LoRa modules and classifier."""
        for name, param in self.bert.named_parameters():
            if "lora" not in name and "classifier" not in name and "dropout" not in name:
                param.requires_grad = False

    def replace_multihead_attention_recursion(self, model):
        """
        Replaces RobertaSelfAttention with LoraRobertaSelfAttention in the model.
        This method applies the replacement recursively to all sub-components.

        Parameters
        ----------
        model : nn.Module
            The PyTorch module or model to be modified.
        """
        for name, module in model.named_children():
            if isinstance(module, BertLoraSelfAttention):
                # Replace RobertaSelfAttention with LoraRobertaSelfAttention
                new_layer = BertLoraSelfAttention(self.bert.config)
                new_layer.load_state_dict(module.state_dict(), strict=False)
                setattr(model, name, new_layer)
            else:
                # Recursive call for child modules
                self.replace_multihead_attention_recursion(module)


    def forward(self, input_ids, attention_mask):
        """
        Takes a batch of sentences and produces embeddings for them.
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        """
        ### TODO
        # outputs = input_ids
        # # extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
        # if(self.enableLora):
        #
        #     for idx, (name, module) in enumerate(self.bert.named_modules()):
        #         if isinstance(module, nn.Linear) and hasattr(module, 'lora_layer'):
        #             # original_output = module.forward(outputs, extended_attention_mask)
        #             print("====idx, name====", idx, name)
        #             original_output = module(input_ids=outputs, attention_mask=attention_mask)['pooler_output'][:, idx]
        #             print("====orig output shape====", original_output.size())
        #             print("====input shape====", outputs.size())
        #             lora_output = module.lora_layer(outputs)
        #             outputs = original_output + lora_output
        #     outputs = outputs['pooler_output']
        # else:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        return outputs


    def predict_sentiment(self, input_ids, attention_mask):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """
        ### TODO

        pooler_output = self.forward(input_ids, attention_mask)
        dropped_output = self.dropout(pooler_output)
        logits = F.relu(self.classifier_layer_1(dropped_output))
        logits = F.relu(self.classifier_layer_2(logits))
        logits = F.relu(self.classifier_layer_3(logits))
        return F.log_softmax(logits, dim=1)


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        """Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        """
        ### TODO
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)

        input_ids = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2,
                                    torch.ones_like(batch_sep_token_id)), dim=1)

        pooler_output = self.forward(input_ids, attention_mask)
        # pooler_output_2 = self.forward(input_ids_2, attention_mask_2)
        pooler_output = self.dropout_para_1(pooler_output)
        # pooler_output_2 = self.dropout_para_2(pooler_output_2)
        # dropped_output_all = torch.cat((pooler_output_1, pooler_output_2), dim=1)
        logits = F.relu(self.classifier_layer_1(pooler_output))
        logits = F.relu(self.classifier_layer_2(logits))
        logits = F.relu(self.classifier_layer_para_3(logits))
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO

        pooler_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooler_output_2 = self.forward(input_ids_2, attention_mask_2)
        # pooler_output_1 = self.dropout_para_1(pooler_output_1)
        # pooler_output_2 = self.dropout_para_2(pooler_output_2)
        logits1 = F.relu(self.classifier_layer_1(pooler_output_1))
        logits1 = F.relu(self.classifier_layer_2(logits1))
        logits2 = F.relu(self.classifier_layer_1(pooler_output_2))
        logits2 = F.relu(self.classifier_layer_2(logits2))
        logits = torch.cosine_similarity(logits1, logits2)

        return logits



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def save_model_loss(model, loss_sst, loss_sts, loss_para, loss, filepath):
    save_info = {
        'model': model.state_dict(),
        'loss_sst': loss_sst,
        'loss_sts': loss_sts,
        'loss_para': loss_para,
        'loss': loss
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


# gradient surgery util function
def project_gradients(grads):
    """
    Project conflicting gradients to make them orthogonal if their dot product is negative.
    Args:
        grads: List of gradients for each task.
    Returns:
        List of projected gradients.
    """
    projected_grads = [g.clone() if g is not None else None for g in grads]
    # print("size proj grads", len(projected_grads))
    # sizes = [l.size() if l is not None else 0 for l in projected_grads]
    # print(sizes)
    for i in range(0, len(projected_grads), 2):
        # for j in range(i + 1, len(projected_grads)):
        j = i+1
        # print("i,j", i, j)
        g_i = projected_grads[i]
        g_j = projected_grads[j]

        if g_i is not None and g_j is not None and g_i.size() == g_j.size():
            # print("g_i", g_i.size())
            # print("g_j", g_j.size())
            dot_product = torch.dot(g_i.view(-1), g_j.view(-1))
            if dot_product < 0:
                # Project g_i to be orthogonal to g_j
                g_i -= dot_product * g_j / torch.norm(g_j)**2
                projected_grads[i] = g_i
    return projected_grads


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    ### STS
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    ### PARA
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    ### STS
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    # config = SimpleNamespace(**config)
    # model = MultitaskBERT(config)
    # model = model.to(device)

    saved = torch.load(args.filepath)
    config = saved['model_config']

    model = MultitaskBERT(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    sst_train = False
    para_train = False
    sts_train = False
    continue_training = True
    mixed_train = True

    # SST Run for the specified number of epochs.
    if (sst_train):
        train_sst(args, config, device, model, optimizer, sst_train_dataloader)

    # torch.cuda.empty_cache() import gc; gc.collect()
    if (para_train):
        train_para(args, config, device, model, optimizer, para_train_dataloader)

    # torch.cuda.empty_cache()
    if (sts_train):
        train_sts(args, best_dev_acc, config, device, model, optimizer, sts_dev_dataloader, sts_train_dataloader)

    # train mixed data
    if(mixed_train):
        train_mixed_model(args, config, device, model, optimizer,
                    sts_dev_dataloader, sts_train_dataloader,
                    para_dev_dataloader, para_train_dataloader,
                    sst_dev_dataloader, sst_train_dataloader)


# weight sharing + stepping after every training
def train_mixed_weight_sharing(args, config, device, model, optimizer,
                    sts_dev_dataloader, sts_train_dataloader,
                    para_dev_dataloader, para_train_dataloader,
                    sst_dev_dataloader, sst_train_dataloader):
    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_sts, batch_sst, batch_para in zip(
                cycle(tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)),
                cycle(tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)),
                tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
        ):
            # STS
            token_ids_sts = batch_sts['token_ids_1'].to(device)
            attention_mask_sts = batch_sts['attention_mask_1'].to(device)
            token_ids2_sts = batch_sts['token_ids_2'].to(device)
            attention_mask2_sts = batch_sts['attention_mask_2'].to(device)
            labels_sts = batch_sts['labels'].to(device)

            # SST
            token_ids_sst, attention_mask_sst, labels_sst = (batch_sst['token_ids'].to(device),
                                                             batch_sst['attention_mask'].to(device),
                                                             batch_sst['labels'].to(device))

            ### Para
            token_ids_para = batch_para['token_ids_1'].to(device)
            attention_mask_para = batch_para['attention_mask_1'].to(device)
            token_ids2_para = batch_para['token_ids_2'].to(device)
            attention_mask2_para = batch_para['attention_mask_2'].to(device)
            labels_para = batch_para['labels'].to(device)


            logits_sts = model.predict_similarity(token_ids_sts, attention_mask_sts, token_ids2_sts, attention_mask2_sts)
            logits_sts = logits_sts.flatten()
            logits_sts = (logits_sts.sigmoid() * 5).to(device)

            logits_para = model.predict_paraphrase(token_ids_para, attention_mask_para, token_ids2_para, attention_mask2_para)
            logits_para = logits_para.sigmoid().round().flatten().to(device)

            logits_sst = model.predict_sentiment(token_ids_sst, attention_mask_sst)

            optimizer.zero_grad()
            loss_sts = F.cross_entropy(logits_sts, labels_sts.view(-1).float(), reduction='sum') / args.batch_size
            loss_sts.backward()
            optimizer.step()

            optimizer.zero_grad()
            loss_para = F.cross_entropy(logits_para, labels_para.view(-1).float(), reduction='sum') / args.batch_size
            loss_para.backward()
            optimizer.step()

            optimizer.zero_grad()
            loss_sst = F.cross_entropy(logits_sst, labels_sst.view(-1), reduction='sum') / args.batch_size
            loss_sst.backward()
            optimizer.step()


            # loss.backward()
            # optimizer.step()

            train_loss += loss_sts.item()  + loss_sst.item() + loss_para.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        # train_acc_sts, train_f1_sts, *_ = model_eval_sts(sts_train_dataloader, model, device)
        # dev_acc_sts, dev_f1_sts, *_ = model_eval_sts(sts_dev_dataloader, model, device)
        #
        # train_acc_para, train_f1_para, *_ = model_eval_para(para_train_dataloader, model, device)
        # dev_acc_para, dev_f1_para, *_ = model_eval_para(para_dev_dataloader, model, device)
        #
        # train_acc_sst, train_f1_sst, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc_sst, dev_f1_sst, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        #
        # dev_acc = dev_acc_sst + dev_acc_para + dev_acc_sts
        # train_acc = train_acc_sst + train_acc_para + train_acc_sts
        #
        # if dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc
        save_model(model, optimizer, args, config, str(epoch)+args.filepath)

        print(
            f"STS Epoch {epoch}: train loss :: {train_loss :.3f}, ") #train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")



def train_mixed_model(args, config, device, model, optimizer,
                    sts_dev_dataloader, sts_train_dataloader,
                    para_dev_dataloader, para_train_dataloader,
                    sst_dev_dataloader, sst_train_dataloader):
    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_sts, batch_sst, batch_para in zip(
                cycle(tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)),
                cycle(tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)),
                tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
        ):
            # STS
            token_ids_sts = batch_sts['token_ids_1'].to(device)
            attention_mask_sts = batch_sts['attention_mask_1'].to(device)
            token_ids2_sts = batch_sts['token_ids_2'].to(device)
            attention_mask2_sts = batch_sts['attention_mask_2'].to(device)
            labels_sts = batch_sts['labels'].to(device)

            # SST
            token_ids_sst, attention_mask_sst, labels_sst = (batch_sst['token_ids'].to(device),
                                                             batch_sst['attention_mask'].to(device),
                                                             batch_sst['labels'].to(device))

            ### Para
            token_ids_para = batch_para['token_ids_1'].to(device)
            attention_mask_para = batch_para['attention_mask_1'].to(device)
            token_ids2_para = batch_para['token_ids_2'].to(device)
            attention_mask2_para = batch_para['attention_mask_2'].to(device)
            labels_para = batch_para['labels'].to(device)


            logits_sts = model.predict_similarity(token_ids_sts, attention_mask_sts, token_ids2_sts, attention_mask2_sts)
            logits_sts = logits_sts.flatten()
            logits_sts = (logits_sts.sigmoid() * 5).to(device)

            logits_para = model.predict_paraphrase(token_ids_para, attention_mask_para, token_ids2_para, attention_mask2_para)
            logits_para = logits_para.sigmoid().round().flatten().to(device)

            logits_sst = model.predict_sentiment(token_ids_sst, attention_mask_sst)

            optimizer.zero_grad()
            loss_sts = F.cross_entropy(logits_sts, labels_sts.view(-1).float(), reduction='sum') / args.batch_size
            loss_sts.backward()
            optimizer.step()

            optimizer.zero_grad()
            loss_para = F.cross_entropy(logits_para, labels_para.view(-1).float(), reduction='sum') / args.batch_size
            loss_para.backward()
            optimizer.step()

            optimizer.zero_grad()
            loss_sst = F.cross_entropy(logits_sst, labels_sst.view(-1), reduction='sum') / args.batch_size
            loss_sst.backward()
            optimizer.step()


            # loss.backward()
            # optimizer.step()

            train_loss += loss_sts.item()  + loss_sst.item() + loss_para.item()
            num_batches += 1
            filepath = str(num_batches) + "-model-loss"
            save_model_loss(model, loss_sst.cpu().detach().numpy(), loss_sts.cpu().detach().numpy(),
                            loss_para.cpu().detach().numpy(), train_loss, filepath)
            if(num_batches > 50):
                return

        train_loss = train_loss / (num_batches)


        # train_acc_sts, train_f1_sts, *_ = model_eval_sts(sts_train_dataloader, model, device)
        # dev_acc_sts, dev_f1_sts, *_ = model_eval_sts(sts_dev_dataloader, model, device)
        #
        # train_acc_para, train_f1_para, *_ = model_eval_para(para_train_dataloader, model, device)
        # dev_acc_para, dev_f1_para, *_ = model_eval_para(para_dev_dataloader, model, device)
        #
        # train_acc_sst, train_f1_sst, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc_sst, dev_f1_sst, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        #
        # dev_acc = dev_acc_sst + dev_acc_para + dev_acc_sts
        # train_acc = train_acc_sst + train_acc_para + train_acc_sts
        #
        # if dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc
        save_model(model, optimizer, args, config, str(epoch)+args.filepath)

        print(
            f"STS Epoch {epoch}: train loss :: {train_loss :.3f}, ") #train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

def pcgrad(grads1, grads2):
    """
    Project conflicting gradients to make them orthogonal if their dot product is negative.
    Args:
        grads: List of gradients for each task.
    Returns:
        List of projected gradients.
    """
    # projected_grads = [g.clone() if g is not None else None for g in grads]
    # print("size proj grads", len(projected_grads))
    # sizes = [l.size() if l is not None else 0 for l in projected_grads]
    # print(sizes)
    out1 = []
    out2 = []
    count = 0
    for g_i, g_j in zip(grads1, grads2):
        count += 1
        # print(count)
        g_ii = g_i.clone() if g_i is not None else None
        g_jj = g_j.clone() if g_j is not None else None
        if (g_i is not None) and (g_j is not None) and (g_i.size() == g_j.size()):
            dot_product = torch.dot(g_i.view(-1), g_j.view(-1))
            if dot_product < 0:
                # Project g_i to be orthogonal to g_j

                g_ii -= dot_product * g_j / torch.norm(g_j)**2
                g_jj -= dot_product * g_i / torch.norm(g_i)**2
        out1.append(g_ii)
        out2.append(g_jj)
    return (out1, out2)


# Training with gradient surgery for conflicting gradients
def train_mixed_gradient_surgery(args, config, device, model, optimizer,
                    sts_dev_dataloader, sts_train_dataloader,
                    para_dev_dataloader, para_train_dataloader,
                    sst_dev_dataloader, sst_train_dataloader):
    best_dev_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_sts, batch_sst, batch_para in zip(
                cycle(tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE))
                ,cycle(tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE))
                ,tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)

        ):
            # STS
            token_ids_sts = batch_sts['token_ids_1'].to(device)
            attention_mask_sts = batch_sts['attention_mask_1'].to(device)
            token_ids2_sts = batch_sts['token_ids_2'].to(device)
            attention_mask2_sts = batch_sts['attention_mask_2'].to(device)
            labels_sts = batch_sts['labels'].to(device)

            # SST
            token_ids_sst, attention_mask_sst, labels_sst = (batch_sst['token_ids'].to(device),
                                                             batch_sst['attention_mask'].to(device),
                                                             batch_sst['labels'].to(device))

            # Para
            token_ids_para = batch_para['token_ids_1'].to(device)
            attention_mask_para = batch_para['attention_mask_1'].to(device)
            token_ids2_para = batch_para['token_ids_2'].to(device)
            attention_mask2_para = batch_para['attention_mask_2'].to(device)
            labels_para = batch_para['labels'].to(device)


            logits_sts = model.predict_similarity(token_ids_sts, attention_mask_sts, token_ids2_sts, attention_mask2_sts)
            logits_sts = logits_sts.flatten()
            logits_sts = (logits_sts.sigmoid() * 5).to(device)

            logits_para = model.predict_paraphrase(token_ids_para, attention_mask_para, token_ids2_para, attention_mask2_para)
            logits_para = logits_para.sigmoid().round().flatten().to(device)

            logits_sst = model.predict_sentiment(token_ids_sst, attention_mask_sst)

            optimizer.zero_grad()
            loss_sts = F.cross_entropy(logits_sts, labels_sts.view(-1).float(), reduction='sum') / args.batch_size
            loss_sts.backward()
            grads_sts = [param.grad.clone() if param.grad is not None else None for param in model.parameters()]

            optimizer.zero_grad()
            loss_para = F.cross_entropy(logits_para, labels_para.view(-1).float(), reduction='sum') / args.batch_size
            loss_para.backward()
            grads_para = [param.grad.clone() if param.grad is not None else None for param in model.parameters()]

            optimizer.zero_grad()
            loss_sst = F.cross_entropy(logits_sst, labels_sst.view(-1), reduction='sum') / args.batch_size
            loss_sst.backward()
            grads_sst = [param.grad.clone() if param.grad is not None else None for param in model.parameters()]

            # Apply gradient projection
            g_sts, g_sst = pcgrad(grads_sts + grads_para, grads_sst)
            for param, proj_grad in zip(model.parameters(), g_sts[:-14]):
                if proj_grad is not None:
                    param.grad = proj_grad
            optimizer.step()
            g_sts, g_sst = pcgrad(grads_sts, grads_sst + grads_para)
            for param, proj_grad in zip(model.parameters(), g_sst[:14]):
                if proj_grad is not None:
                    param.grad = proj_grad
            optimizer.step()

            # g_sts, g_para = pcgrad(grads_sts, grads_para)
            # for param, proj_grad in zip(model.parameters(), g_sts[:-14]):
            #     if proj_grad is not None:
            #         param.grad = proj_grad
            # optimizer.step()
            # for param, proj_grad in zip(model.parameters(), g_para[:14]):
            #     if proj_grad is not None:
            #         param.grad = proj_grad
            # optimizer.step()

            g_sst, g_para = pcgrad(grads_sst+grads_sts, grads_para)
            # for param, proj_grad in zip(model.parameters(), g_sst[:-14]):
            #     if proj_grad is not None:
            #         param.grad = proj_grad
            # optimizer.step()
            for param, proj_grad in zip(model.parameters(), g_para[:14]):
                if proj_grad is not None:
                    param.grad = proj_grad
            optimizer.step()


            # loss.backward()
            # optimizer.step()

            train_loss += loss_sts.item() + loss_sst.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        # train_acc_sts, train_f1_sts, *_ = model_eval_sts(sts_train_dataloader, model, device)
        # dev_acc_sts, dev_f1_sts, *_ = model_eval_sts(sts_dev_dataloader, model, device)
        #
        # train_acc_para, train_f1_para, *_ = model_eval_para(para_train_dataloader, model, device)
        # dev_acc_para, dev_f1_para, *_ = model_eval_para(para_dev_dataloader, model, device)
        #
        # train_acc_sst, train_f1_sst, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc_sst, dev_f1_sst, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        #
        # dev_acc = dev_acc_sst + dev_acc_para + dev_acc_sts
        # train_acc = train_acc_sst + train_acc_para + train_acc_sts
        #
        # if dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc
        save_model(model, optimizer, args, config, str(epoch)+args.filepath)

        print(
            f"STS Epoch {epoch}: train loss :: {train_loss :.3f}, ") #train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

def train_sts(args, best_dev_acc, config, device, model, optimizer, sts_dev_dataloader, sts_train_dataloader):
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            token_ids = batch['token_ids_1'].to(device)
            attention_mask = batch['attention_mask_1'].to(device)
            token_ids2 = batch['token_ids_2'].to(device)
            attention_mask2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(token_ids, attention_mask, token_ids2, attention_mask2)
            logits = logits.flatten().to(device)
            loss = F.cross_entropy(logits, labels.view(-1).float(), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sts(sts_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sts(sts_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"STS Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        # print(
        #     f"STS Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


def train_para(args, config, device, model, optimizer, para_train_dataloader):
    for epoch in range(1):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            token_ids = batch['token_ids_1'].to(device)
            attention_mask = batch['attention_mask_1'].to(device)
            token_ids2 = batch['token_ids_2'].to(device)
            attention_mask2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(token_ids, attention_mask, token_ids2, attention_mask2)
            logits = logits.sigmoid().round().flatten().to(device)
            loss = F.cross_entropy(logits, labels.view(-1).float(), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        print("Saving model.")
        save_model(model, optimizer, args, config, args.filepath)
        print("Model saved.")
        train_loss = train_loss / (num_batches)

        # train_acc, train_f1, *_ = model_eval_para(para_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_para(para_dev_dataloader, model, device)

        # if dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc

        # print(f"Para Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        # print(
        #     f"Para Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


def train_sst(args, config, device, model, optimizer, sst_train_dataloader):
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        # if dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc
        save_model(model, optimizer, args, config, args.filepath)

        # print(f"SST Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        # print(
        #     f"SST Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


def train_mixed_lora(args, config, device, model, optimizer,
                    sts_dev_dataloader, sts_train_dataloader,
                    para_dev_dataloader, para_train_dataloader,
                    sst_dev_dataloader, sst_train_dataloader):
    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_sts, batch_sst, batch_para in zip(
                cycle(tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)),
                cycle(tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)),
                tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            # STS
            token_ids_sts = batch_sts['token_ids_1'].to(device)
            attention_mask_sts = batch_sts['attention_mask_1'].to(device)
            token_ids2_sts = batch_sts['token_ids_2'].to(device)
            attention_mask2_sts = batch_sts['attention_mask_2'].to(device)
            labels_sts = batch_sts['labels'].to(device)

            # SST
            token_ids_sst, attention_mask_sst, labels_sst = (batch_sst['token_ids'].to(device),
                                                             batch_sst['attention_mask'].to(device),
                                                             batch_sst['labels'].to(device))

            # Para
            token_ids_para = batch_para['token_ids_1'].to(device)
            attention_mask_para = batch_para['attention_mask_1'].to(device)
            token_ids2_para = batch_para['token_ids_2'].to(device)
            attention_mask2_para = batch_para['attention_mask_2'].to(device)
            labels_para = batch_para['labels'].to(device)

            logits_sts = model.predict_similarity(token_ids_sts, attention_mask_sts, token_ids2_sts, attention_mask2_sts)
            logits_sts = logits_sts.flatten()
            logits_sts = (logits_sts.sigmoid() * 5).to(device)

            logits_para = model.predict_paraphrase(token_ids_para, attention_mask_para, token_ids2_para, attention_mask2_para)
            logits_para = logits_para.sigmoid().round().flatten().to(device)

            logits_sst = model.predict_sentiment(token_ids_sst, attention_mask_sst)

            optimizer.zero_grad()
            loss_sts = F.cross_entropy(logits_sts, labels_sts.view(-1).float(), reduction='sum') / args.batch_size
            loss_sts.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_para = F.cross_entropy(logits_para, labels_para.view(-1).float(), reduction='sum') / args.batch_size
            loss_para.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sst = F.cross_entropy(logits_sst, labels_sst.view(-1), reduction='sum') / args.batch_size
            loss_sst.backward()
            optimizer.step()


            # loss.backward()
            # optimizer.step()

            train_loss += loss_sts.item() + loss_para.item() + loss_sst.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        # train_acc_sts, train_f1_sts, *_ = model_eval_sts(sts_train_dataloader, model, device)
        # dev_acc_sts, dev_f1_sts, *_ = model_eval_sts(sts_dev_dataloader, model, device)
        #
        # train_acc_para, train_f1_para, *_ = model_eval_para(para_train_dataloader, model, device)
        # dev_acc_para, dev_f1_para, *_ = model_eval_para(para_dev_dataloader, model, device)
        #
        # train_acc_sst, train_f1_sst, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc_sst, dev_f1_sst, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        #
        # dev_acc = dev_acc_sst + dev_acc_para + dev_acc_sts
        # train_acc = train_acc_sst + train_acc_para + train_acc_sts
        #
        # if dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc
        save_model(model, optimizer, args, config, args.filepath)

        print(
            f"STS Epoch {epoch}: train loss :: {train_loss :.3f}, ") #train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--filepath", type=str, help="filepath", default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    # test_multitask(args)
