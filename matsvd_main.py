from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_constant_schedule, RobertaTokenizerFast, T5EncoderModel, get_linear_schedule_with_warmup, \
    RobertaModel, T5Config, RobertaTokenizer
from tqdm import tqdm
from matsvd_model import Model
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

logger = logging.getLogger(__name__)


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 statement_mask,
                 labels,
                 func_labels,
                 num_statements,
                 adj,
                 func_index):
        self.input_ids = input_ids
        self.statement_mask = statement_mask
        self.labels = labels
        self.func_labels = func_labels
        self.num_statements = num_statements
        self.adj = adj
        self.func_index = func_index


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "val":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        self.vul_funcs = 0
        self.benign_funcs = 0
        self.vul_statements = 0
        self.benign_statements = 0

        ## 读取json文件内容 ###
        jsonfile_path = 'data/vul2_wc.json'

        with open(jsonfile_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)

        with open('data/top1.txt','r') as file:
            json_idx = [line.strip() for line in file.readlines()]  # 里面元素的类型是string
            # print("json里面记录的idx的个数 ： ", len(json_idx))
        
        logger.info("*** 读取数据： ***")
        for index in tqdm(dataset.keys(), bar_format='{l_bar}{bar} [{elapsed}]', ncols=100):  # 这里的 index 是 string 类型
            if index not in json_idx:  ## 只读取json_idx中的样本
                continue
            sample_dict = dataset[index]
            processed_code = sample_dict['processed_code']
            vulnerable_lines_index = sample_dict['vulnerable_lines_index']
            self.examples.append(convert_examples_to_features(processed_code, vulnerable_lines_index, tokenizer, args, file_type, int(index), data_type='json'))
        # print("json文件中的vul函数数量 ：", sum(1 for ex in self.examples if ex.func_labels == 1))

        ## 读取csv文件内容 ###
        df_all = pd.read_csv(file_path)
        df_vul = df_all[df_all["function_label"] == 1].reset_index(drop=True)
        df_non_vul = df_all[df_all["function_label"] == 0].reset_index(drop=True)
        df = pd.concat((df_vul, df_non_vul))
        df = df.reset_index(drop=True)

        labels = df["statement_label"].tolist()  # [:300]
        source = df["func_before"].tolist()   # [:300]
        indexs = df["index"].tolist()   # 里面元素的类型是int

        print("\n*******\n", f"total non-vul funcs in {file_type} data: {len(df_non_vul)}")
        print(f"total vul funcs in {file_type} data: {len(df_vul)}", "\n*******\n")

        with open('data/dele_idx2.txt','r') as file:
            dele_idx = [int(line.strip()) for line in file.readlines()]  # 里面元素的类型是int
            # print("dele_idx2 的长度 ： ", len(dele_idx))

        vulnum = 0
        for i in tqdm(range(len(source)), bar_format='{l_bar}{bar} [{elapsed}]', ncols=100):
            if indexs[i] in dele_idx:
                continue
            self.examples.append(convert_examples_to_features(source[i], labels[i], tokenizer, args, file_type, indexs[i],data_type='csv'))
            if self.examples[-1].func_labels == 1:
                vulnum += 1
        # print("csv文件中的vul函数数量 ：", vulnum)

        for s in self.examples:
            if s.func_labels == 1:
                self.vul_funcs += 1
                self.vul_statements += sum(1 for lab in s.labels[:s.num_statements] if lab == 1)
                self.benign_statements += sum(1 for lab in s.labels[:s.num_statements] if lab == 0)
            else:
                self.benign_funcs += 1
                self.benign_statements += s.num_statements

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].statement_mask), torch.tensor(
            self.examples[i].labels).float(), torch.tensor(self.examples[i].func_labels), torch.tensor(
            self.examples[i].num_statements), torch.tensor(self.examples[i].adj), torch.tensor(
            self.examples[i].func_index)


def convert_examples_to_features(source, labels, tokenizer, args, data_split, func_index, data_type):
    if data_type == "csv":
        labels = labels.strip("[").strip("]")
        labels = labels.split(",")
        labels = [int(l.strip()) for l in labels]
        assert len(labels) == args.num_labels
    elif data_type == 'json':
        ll = [0] * args.num_labels
        if labels.strip():
            vulnerable_lines_index = labels.split(",")  # ['73', '77', '81']
            for lineidx in vulnerable_lines_index:
                if int(lineidx) < args.num_labels:
                    ll[int(lineidx)] = 1
        labels = ll
        assert len(labels) == args.num_labels

    source = source.split("\n")
    source = source[:args.num_labels]
    padding_statement = [tokenizer.pad_token_id for _ in range(22)]  # 长度为 22
    num_statements = len(source)
    input_ids = []
    for stat in source:
        ids_ = tokenizer.encode(str(stat),
                                truncation=True,
                                max_length=20,
                                padding='max_length',
                                add_special_tokens=False)
        if len(ids_) == 0:
            ids_ = padding_statement
        else:
            ids_ = [tokenizer.cls_token_id] + ids_ + [tokenizer.sep_token_id]  # 长度为 22
        input_ids.append(ids_)

    if len(input_ids) < args.num_labels:
        for _ in range(args.num_labels - len(input_ids)):
            input_ids.append(padding_statement)

    statement_mask = []  # 长度为 155
    for statement in input_ids:
        if statement == padding_statement:
            statement_mask.append(0)
        else:
            statement_mask.append(1)

    if 1 in labels:
        func_labels = 1
    else:
        func_labels = 0

    # 读取edge_json文件
    if data_type == "csv":
        edgejson_dir = "edge_json/processed_test/"
    elif data_type == "json":
        edgejson_dir = "edge_json/vul2/"

    edges = []
    filepath = os.path.join(edgejson_dir, str(func_index) + '_edge.json')

    try:
        with open(filepath, "r") as f:
            edges = json.load(f)
    except FileNotFoundError:
        pass

    adj = [statement_mask[:] for _ in range(155)]  # 155*155 的嵌套list

    if len(edges) != 0:
        edge_connection = {}
        for e in edges:
            try:
                line_a = e[0]
                line_b = e[1]
                if line_a <= 154 and line_b <= 154:
                    edge_connection = update_dict(edge_connection, key=line_a, value=line_b)
                    edge_connection = update_dict(edge_connection, key=line_b, value=line_a)
            except:
                continue

        for i in edge_connection.keys():
            connection = edge_connection[i]
            for idx in range(155):
                if idx not in connection:
                    adj[i][idx] = 0

    assert len(input_ids) == args.num_labels

    return InputFeatures(input_ids, statement_mask, labels, func_labels, num_statements, adj, func_index)


def update_dict(edge_connection, key, value):
    if key in edge_connection:
        edge_connection[key].add(value)
    else:
        edge_connection[key] = set()
        edge_connection[key].add(key)
        edge_connection[key].add(value)
    return edge_connection


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1

    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    model_to_save = model.module if hasattr(model, 'module') else model
    for key in model_to_save.state_dict():
        print(key, end=", ")
    print()

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_constant_schedule(optimizer)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader) * args.epochs * 0.1,
                                                num_training_steps=len(train_dataloader) * args.epochs)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num vul_funcs = %d", train_dataset.vul_funcs)
    logger.info("  Num benign_funcs = %d", train_dataset.benign_funcs)
    logger.info("  Num vul_statements = %d", train_dataset.vul_statements)
    logger.info("  Num benign_statements = %d", train_dataset.benign_statements)
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            input_ids, statement_mask, labels, func_labels, num_statements, adj, func_index = [x.to(args.device) for x
                                                                                               in batch]
            model.train()
            statement_loss, func_loss = model(input_ids_with_pattern=input_ids,
                                              statement_mask=statement_mask,
                                              labels=labels,
                                              func_labels=func_labels,
                                              adj=adj,
                                              func_index=func_index)
            loss = 0.5 * statement_loss + 0.5 * func_loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_f1 = test(args, model, tokenizer, eval_dataset, eval_when_training=True)
                    # Save model checkpoint
                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        logger.info("  " + "*" * 20)
                        logger.info("  Best line-level F1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def test(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    # eval_when_training = FALSE，说明在测试阶段执行此函数，需要先把model包装为并行模式
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Num vul_funcs = %d", eval_dataset.vul_funcs)
    # logger.info("  Num benign_funcs = %d", eval_dataset.benign_funcs)
    # logger.info("  Num vul_statements = %d", eval_dataset.vul_statements)
    # logger.info("  Num benign_statements = %d", eval_dataset.benign_statements)
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []
    func_level_trues = []
    func_level_preds = []
    for step, batch in enumerate(bar):
        with torch.no_grad():
            input_ids, statement_mask, labels, func_labels, num_statements, adj, func_index = [x.to(args.device) for x
                                                                                               in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                      statement_mask=statement_mask,
                                      adj=adj)
            preds = torch.where(probs > 0.5, 1, 0).tolist()

            func_preds = torch.argmax(func_probs, dim=-1).tolist()

            ### function-level ###
            func_labels = func_labels.cpu().numpy().tolist()
            func_level_trues += func_labels
            func_level_preds += func_preds

            for indx in range(len(preds)):
                sample = preds[indx]
                if func_preds[indx] == 1:
                    for s in range(num_statements[indx]):
                        p = sample[s]
                        y_preds.append(p)
                else:
                    for _ in range(num_statements[indx]):
                        y_preds.append(0)
            labels = labels.cpu().numpy().tolist()
            for indx in range(len(labels)):
                sample = labels[indx]
                for s in range(num_statements[indx]):
                    lab = sample[s]
                    y_trues.append(lab)

    func_f1 = f1_score(y_true=func_level_trues, y_pred=func_level_preds)
    func_acc = accuracy_score(y_true=func_level_trues, y_pred=func_level_preds)
    func_recall = recall_score(y_true=func_level_trues, y_pred=func_level_preds)
    func_pre = precision_score(y_true=func_level_trues, y_pred=func_level_preds)
    

    func_TP = sum((p == 1) and (t == 1) for p, t in zip(func_level_preds, func_level_trues))
    func_FP = sum((p == 1) and (t == 0) for p, t in zip(func_level_preds, func_level_trues))
    func_TN = sum((p == 0) and (t == 0) for p, t in zip(func_level_preds, func_level_trues))
    func_FN = sum((p == 0) and (t == 1) for p, t in zip(func_level_preds, func_level_trues))
    func_MCC = (func_TP * func_TN - func_FP * func_FN) / np.sqrt(
        (func_TP + func_FP) * (func_TP + func_FN) * (func_TN + func_FP) * (func_TN + func_FN))

    y_TP = sum((p == 1) and (t == 1) for p, t in zip(y_preds, y_trues))
    y_FP = sum((p == 1) and (t == 0) for p, t in zip(y_preds, y_trues))
    y_TN = sum((p == 0) and (t == 0) for p, t in zip(y_preds, y_trues))
    y_FN = sum((p == 0) and (t == 1) for p, t in zip(y_preds, y_trues))
    y_MCC = (y_TP * y_TN - y_FP * y_FN) / np.sqrt((y_TP + y_FP) * (y_TP + y_FN) * (y_TN + y_FP) * (y_TN + y_FN))

    logger.info("***** Function-level Test results *****")
    logger.info(f"True Positives: {str(func_TP)}")
    logger.info(f"False Positives: {str(func_FP)}")
    logger.info(f"True Negatives: {str(func_TN)}")
    logger.info(f"False Negatives: {str(func_FN)}")
    logger.info(f"func_MCC Score: {str(func_MCC)}")
    logger.info(f"F1 Score: {str(func_f1)}")
    logger.info(f"acc Score: {str(func_acc)}")
    logger.info(f"recall Score: {str(func_recall)}")
    logger.info(f"pre Score: {str(func_pre)}")

    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    recall = recall_score(y_true=y_trues, y_pred=y_preds)
    pre = precision_score(y_true=y_trues, y_pred=y_preds)

    logger.info("***** Line-level Test results *****")
    logger.info(f"True Positives: {str(y_TP)}")
    logger.info(f"False Positives: {str(y_FP)}")
    logger.info(f"True Negatives: {str(y_TN)}")
    logger.info(f"False Negatives: {str(y_FN)}")
    logger.info(f"F1 Score: {str(f1)}")
    logger.info(f"y_MCC Score: {str(y_MCC)}")
    logger.info(f"acc Score: {str(acc)}")
    logger.info(f"recall Score: {str(recall)}")
    logger.info(f"pre Score: {str(pre)}")
    return f1


def evaluate_localization(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    # eval_when_training = FALSE，说明在测试阶段执行此函数，需要先把model包装为并行模式
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation localization *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Num vul_funcs = %d", eval_dataset.vul_funcs)
    logger.info("  Num vul_statements = %d", eval_dataset.vul_statements)
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []

    ## 用来记录每个函数中的 line-level 的结果
    line_level_dict = {}

    for step, batch in enumerate(bar):
        with torch.no_grad():
            input_ids, statement_mask, labels, func_labels, num_statements, adj, func_index = [x.to(args.device) for x
                                                                                               in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                      statement_mask=statement_mask,
                                      adj=adj)
            preds = torch.where(probs > 0.5, 1, 0).tolist()

            ### function-level 的正确答案， 让模型只在已知带有漏洞的函数上检测漏洞语句 ###
            func_labels = func_labels.cpu().numpy().tolist()
            func_index = func_index.cpu().numpy().tolist()

            ### statement-level ###
            for indx in range(len(preds)):
                sample = preds[indx]
                if func_labels[indx] == 1:
                    for s in range(num_statements[indx]):
                        p = sample[s]
                        y_preds.append(p)

            labels = labels.cpu().numpy().tolist()
            for indx in range(len(labels)):
                sample = labels[indx]
                if func_labels[indx] == 1:
                    for s in range(num_statements[indx]):
                        lab = sample[s]
                        y_trues.append(lab)


    ### 遍历完所有函数后计算总的f1值
    y_TP = sum((p == 1) and (t == 1) for p, t in zip(y_preds, y_trues))
    y_FP = sum((p == 1) and (t == 0) for p, t in zip(y_preds, y_trues))
    y_TN = sum((p == 0) and (t == 0) for p, t in zip(y_preds, y_trues))
    y_FN = sum((p == 0) and (t == 1) for p, t in zip(y_preds, y_trues))

    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    recall = recall_score(y_true=y_trues, y_pred=y_preds)
    pre = precision_score(y_true=y_trues, y_pred=y_preds)
    y_MCC = (y_TP * y_TN - y_FP * y_FN) / np.sqrt((y_TP + y_FP) * (y_TP + y_FN) * (y_TN + y_FP) * (y_TN + y_FN))

    logger.info("***** localization Line-level Test results *****")
    logger.info(f"True Positives: {str(y_TP)}")
    logger.info(f"False Positives: {str(y_FP)}")
    logger.info(f"True Negatives: {str(y_TN)}")
    logger.info(f"False Negatives: {str(y_FN)}")
    logger.info(f"y_MCC Score: {str(y_MCC)}")
    logger.info(f"F1 Score: {str(f1)}")
    logger.info(f"acc Score: {str(acc)}")
    logger.info(f"recall Score: {str(recall)}")
    logger.info(f"pre Score: {str(pre)}")
    return f1

def evaluate_rank_metrics(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    # eval_when_training = FALSE，说明在测试阶段执行此函数，需要先把model包装为并行模式
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation ranking metrics *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Num vul_funcs = %d", eval_dataset.vul_funcs)
    logger.info("  Num vul_statements = %d", eval_dataset.vul_statements)
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))

    top1_vul_func = 0
    top3_vul_func = 0
    top5_vul_func = 0
    MFR_all = 0
    MAR_all = 0

    for step, batch in enumerate(bar):
        with torch.no_grad():
            input_ids, statement_mask, labels, func_labels, num_statements, adj, func_index = [x.to(args.device) for x
                                                                                               in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                      statement_mask=statement_mask,
                                      adj=adj)

            # probs 是 16*155的张量

            ### function-level 的正确答案， 让模型只在已知带有漏洞的函数上检测漏洞语句 ###
            func_labels = func_labels.cpu().numpy().tolist()
            func_index = func_index.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            num_statements = num_statements.cpu().numpy().tolist()

            _, top_1_index = torch.max(probs, dim=1)
            _, top_3_index = torch.topk(probs, k=3, sorted=True, dim=1)
            _, top_5_index = torch.topk(probs, k=5, sorted=True, dim=1)
            _, top_all_index = torch.sort(probs, dim=1,descending=True)

            ### 计算 top_1， top_3 ， top_5 ###
            for indx in range(len(probs)):
                if func_labels[indx] == 1:
                    sample_vulline_index = [i for i, lab in enumerate(labels[indx]) if lab == 1]
                    pred_1 = top_1_index[indx]  # 获得对应的漏洞函数中排在第一的行号
                    pred_3 = top_3_index[indx]   # 获得对应的漏洞函数中排在前3的行号
                    pred_5 = top_5_index[indx]   # 获得对应的漏洞函数中排在前5的行号

                    if pred_1 in sample_vulline_index:
                        top1_vul_func += 1

                    for line in pred_3:
                        if line in sample_vulline_index:
                            top3_vul_func += 1
                            break

                    for line in pred_5:
                        if line in sample_vulline_index:
                            top5_vul_func += 1
                            break

                    # 去掉 pred_all 中的填充语句
                    func_len = num_statements[indx]  # 函数的总长度
                    pred_all = top_all_index[indx].cpu().numpy().tolist()  # 获得对应的漏洞函数中所有的排序行号
                    pred_all = [z for z in pred_all if z < func_len]  # 将不属于函数中的索引去掉
                    assert len(pred_all) == func_len

                    # 计算 MFR
                    for i, line in enumerate(pred_all):
                        if line in sample_vulline_index:
                            MFR_all += i + 1      # 排名从1开始起算
                            break

                    # 计算 MAR
                    i_ar = 0
                    for i, line in enumerate(pred_all):
                        if line in sample_vulline_index:
                            i_ar += i + 1      # 排名从1开始起算,把所有漏洞语句的行号都加起来
                    i_ar = i_ar /len(sample_vulline_index)  # 获得每个漏洞函数的所有漏洞语句的平均排序行号
                    MAR_all += i_ar

    top_1_acc = top1_vul_func / eval_dataset.vul_funcs
    top_3_acc = top3_vul_func / eval_dataset.vul_funcs
    top_5_acc = top5_vul_func / eval_dataset.vul_funcs

    MFR = MFR_all / eval_dataset.vul_funcs
    MAR = MAR_all / eval_dataset.vul_funcs

    logger.info("***** ranking metrics Test results *****")
    logger.info(f"top_1_acc: {str(top_1_acc)}")
    logger.info(f"top_3_acc: {str(top_3_acc)}")
    logger.info(f"top_5_acc: {str(top_5_acc)}")
    logger.info(f"MFR: {str(MFR)}")
    logger.info(f"MAR: {str(MAR)}")

    return 0

class Test_result_object(object):
    def __init__(self,
                 line_preds,
                 line_labels,
                 func_preds,
                 func_labels,
                 num_statements,
                 func_index):
        self.func_index = func_index
        self.line_preds = line_preds
        self.line_labels = line_labels
        self.func_preds = func_preds
        self.func_labels = func_labels
        self.num_statements = num_statements
        
Test_result_object_list = []

def analyse_two_phase_detection(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # Eval!
    logger.info("\n" + "*"*30)
    logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Num vul_funcs = %d", eval_dataset.vul_funcs)
    # logger.info("  Num benign_funcs = %d", eval_dataset.benign_funcs)
    # logger.info("  Num vul_statements = %d", eval_dataset.vul_statements)
    # logger.info("  Num benign_statements = %d", eval_dataset.benign_statements)
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for step, batch in enumerate(bar):
        with torch.no_grad():
            input_ids, statement_mask, labels, func_labels, num_statements, adj, func_index = [x.to(args.device) for x in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                      statement_mask=statement_mask,
                                      adj=adj)
            preds = torch.where(probs > 0.5, 1, 0).tolist()

            func_preds = torch.argmax(func_probs, dim=-1).tolist()

            num_statements = num_statements.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()

            ### function-level ###
            func_labels = func_labels.cpu().numpy().tolist()

            for indx in range(len(preds)):
                line_labels = labels[indx][:num_statements[indx]]
                func_len = num_statements[indx]  # 函数长度
                if func_preds[indx] == 1:
                    sample = preds[indx]  # sample 是一个长度为155的向量，记录了单个函数每行语句的预测结果
                    line_preds = sample[:func_len]           # linepreds 是某一行的预测结果
                else:
                    line_preds = [0] * func_len
                tboject = Test_result_object(func_index=func_index[indx], func_preds=func_preds[indx], func_labels=func_labels[indx],
                                            line_preds=line_preds,line_labels=line_labels,num_statements=num_statements[indx])
                Test_result_object_list.append(tboject)

    # 以上是一个for循环，遍历test_dataset
    #######################################
    # 对每个区间进行统计
    bins = [0,40,80,120,160]

    
    for i in range(len(bins) - 1):
        lower_bound = bins[i]  # 区间的下界
        upper_bound = bins[i + 1]  # 区间的上界
        logger.info("\n" + "*"*30)
        logger.info(f"当前区间为: [{str(lower_bound)} , {str(upper_bound)})")

        # 遍历 Test_result_object_list 来统计代码长度满足当前区间的func数量
        tmp_func_preds = []
        tmp_func_trues = []

        tmp_line_preds = []
        tmp_line_trues = []

        for tob in Test_result_object_list:
            if lower_bound <= tob.num_statements < upper_bound:
                tmp_func_preds.append(tob.func_preds)
                tmp_func_trues.append(tob.func_labels)

                tmp_line_preds.extend(tob.line_preds)
                tmp_line_trues.extend(tob.line_labels)

        # 得到一个区间内的所有数据，进行计算
        func_f1 = f1_score(y_true=tmp_func_trues, y_pred=tmp_func_preds)
        func_acc = accuracy_score(y_true=tmp_func_trues, y_pred=tmp_func_preds)
        func_recall = recall_score(y_true=tmp_func_trues, y_pred=tmp_func_preds)
        func_pre = precision_score(y_true=tmp_func_trues, y_pred=tmp_func_preds)

        func_TP = sum((p == 1) and (t == 1) for p, t in zip(tmp_func_preds, tmp_func_trues))
        func_FP = sum((p == 1) and (t == 0) for p, t in zip(tmp_func_preds, tmp_func_trues))
        func_TN = sum((p == 0) and (t == 0) for p, t in zip(tmp_func_preds, tmp_func_trues))
        func_FN = sum((p == 0) and (t == 1) for p, t in zip(tmp_func_preds, tmp_func_trues))

        y_TP = sum((p == 1) and (t == 1) for p, t in zip(tmp_line_preds, tmp_line_trues))
        y_FP = sum((p == 1) and (t == 0) for p, t in zip(tmp_line_preds, tmp_line_trues))
        y_TN = sum((p == 0) and (t == 0) for p, t in zip(tmp_line_preds, tmp_line_trues))
        y_FN = sum((p == 0) and (t == 1) for p, t in zip(tmp_line_preds, tmp_line_trues))

        logger.info("***** 各个区间上的 Function-level Test results *****")
        logger.info(f"True Positives: {str(func_TP)}")
        logger.info(f"False Positives: {str(func_FP)}")
        logger.info(f"True Negatives: {str(func_TN)}")
        logger.info(f"False Negatives: {str(func_FN)}")
        logger.info(f"F1 Score: {str(func_f1)}")
        logger.info(f"acc Score: {str(func_acc)}")
        logger.info(f"recall Score: {str(func_recall)}")
        logger.info(f"pre Score: {str(func_pre)}")

        f1 = f1_score(y_true=tmp_line_trues, y_pred=tmp_line_preds)
        acc = accuracy_score(y_true=tmp_line_trues, y_pred=tmp_line_preds)
        recall = recall_score(y_true=tmp_line_trues, y_pred=tmp_line_preds)
        pre = precision_score(y_true=tmp_line_trues, y_pred=tmp_line_preds)

        logger.info("***** 各个区间上的 Line-level Test results *****")
        logger.info(f"True Positives: {str(y_TP)}")
        logger.info(f"False Positives: {str(y_FP)}")
        logger.info(f"True Negatives: {str(y_TN)}")
        logger.info(f"False Negatives: {str(y_FN)}")
        logger.info(f"F1 Score: {str(f1)}")
        logger.info(f"acc Score: {str(acc)}")
        logger.info(f"recall Score: {str(recall)}")
        logger.info(f"pre Score: {str(pre)}")

    return 0



def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("--train_data_file", default=None, type=str, required=False,
                    help="The input training data file (a csv file).")
    ps.add_argument("--eval_data_file", default=None, type=str, required=False,
                    help="The input training data file (a csv file).")
    ps.add_argument("--test_data_file", default=None, type=str, required=False,
                    help="The input training data file (a csv file).")
    ps.add_argument("--pretrain_language", default="", type=str, required=False,
                    help="python, go, ruby, php, javascript, java, c_cpp")
    ps.add_argument("--output_dir", default=None, type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")
    ps.add_argument("--model_type", default="roberta", type=str,
                    help="The model architecture to be fine-tuned.")
    ps.add_argument("--encoder_block_size", default=512, type=int,
                    help="")
    ps.add_argument("--max_line_length", default=64, type=int,
                    help="")
    ps.add_argument("--model_name", default="model.bin", type=str,
                    help="Saved model name.")
    ps.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                    help="Checkpoint model name.")
    ps.add_argument("--model_name_or_path", default=None, type=str,
                    help="The model checkpoint for weights initialization.")
    ps.add_argument("--config_name", default="", type=str,
                    help="Optional pretrained config name or path if not the same as model_name_or_path")
    ps.add_argument("--tokenizer_name", default="", type=str,
                    help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    ps.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
    ps.add_argument("--do_test", action='store_true',
                    help="Whether to run training.")
    ps.add_argument("--evaluate_during_training", action='store_true',
                    help="Run evaluation during training at each logging step.")
    ps.add_argument("--train_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for training.")
    ps.add_argument("--eval_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
    ps.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    ps.add_argument("--learning_rate", default=1e-4, type=float,
                    help="The initial learning rate for AdamW.")
    ps.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
    ps.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
    ps.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
    ps.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    ps.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
    ps.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
    ps.add_argument('--epochs', type=int, default=3,
                    help="training epochs")
    args = ps.parse_args()

    args.num_labels = 155

    # Setup logging
    file_handler = logging.FileHandler('matsvd.log', mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)

    # Set seed
    set_seed(args)

    token_encoder_config = T5Config.from_pretrained("./pretrained_model/Salesforce/codet5-base",
                                                    local_files_only=True,
                                                    output_hidden_states=True,
                                                    )
    statement_encoder_config = T5Config.from_pretrained("./pretrained_model/Salesforce/codet5-base",
                                                        local_files_only=True,
                                                        output_hidden_states=True,
                                                        output_attentions=True,
                                                        num_layers=4
                                                        )

    tokenizer = RobertaTokenizer.from_pretrained('./pretrained_model/Salesforce/codet5-base', local_files_only=True)
    token_aggregation_encoder = T5EncoderModel.from_pretrained('./pretrained_model/Salesforce/codet5-base',
                                                               config=token_encoder_config,
                                                               local_files_only=True, ignore_mismatched_sizes=True)
    statement_encoder = T5EncoderModel.from_pretrained('./pretrained_model/Salesforce/codet5-base',
                                                       config=statement_encoder_config,
                                                       local_files_only=True, ignore_mismatched_sizes=True)

    model = Model(token_aggregation_encoder, statement_encoder, tokenizer, args, statement_encoder_config,
                  hidden_dim=768)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='val')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset)
        evaluate_localization(args, model, tokenizer, test_dataset)
        evaluate_rank_metrics(args, model, tokenizer, test_dataset)
        analyse_two_phase_detection(args, model, tokenizer, test_dataset)


if __name__ == "__main__":
    main()
