import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import T5Config, T5EncoderModel

import hook_t5_mask_back_2layers

statement_aggregation_encoder_config = T5Config.from_pretrained("./pretrained_model/Salesforce/codet5-base",
                                                                local_files_only=True,
                                                                output_hidden_states=True,
                                                                num_layers=1
                                                                )


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_dim, args, tokenizer):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.Dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.statement_aggregation_encoder = T5EncoderModel.from_pretrained('./pretrained_model/Salesforce/codet5-base',
                                                                            config=statement_aggregation_encoder_config,
                                                                            local_files_only=True,
                                                                            ignore_mismatched_sizes=True)
        self.func_dense = nn.Linear(hidden_dim, hidden_dim)
        self.func_out_proj = nn.Linear(hidden_dim, 2)
        self.args = args
        self.tokenizer = tokenizer

    def forward(self, hidden, statement_mask):
        # hidden 的大小为 16*155*768
        x = self.Dropout(hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)

        # 在 hidden 的第二维度的起始位置放一个cls_embed
        cls = torch.tensor([self.tokenizer.cls_token_id]).to(self.args.device)
        cls_embed = self.statement_aggregation_encoder.shared(cls).to(self.args.device)  # 1*768
        cls_embed = cls_embed.unsqueeze(0).expand(hidden.shape[0], -1, -1)  # 16*1*768
        hidden = torch.cat((cls_embed, hidden), dim=1)  # 30*156*768

        # 在 statement_mask 的 第二个维度的起始位置放一个1
        one_tensor = torch.ones(statement_mask.shape[0], 1).to(self.args.device)  # 16*1
        statement_mask = torch.cat((one_tensor, statement_mask), dim=1)  # 16*156
        rep = self.statement_aggregation_encoder(inputs_embeds=hidden, attention_mask=statement_mask).last_hidden_state  # 16*156*768

        func_x = rep[:, 0, :]  # 16*768
        func_x = self.Dropout(func_x)
        func_x = self.func_dense(func_x)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x

class Model(nn.Module):
    def __init__(self, token_aggregation_encoder, statement_encoder, tokenizer, args, statement_encoder_config=None,hidden_dim=768, num_labels=155):
        super(Model, self).__init__()
        self.token_aggregation_encoder = token_aggregation_encoder
        self.statement_encoder = statement_encoder
        self.tokenizer = tokenizer
        self.args = args
        self.statement_encoder_config = statement_encoder_config
        # CLS head
        self.classifier = ClassificationHead(hidden_dim=hidden_dim,args=args,tokenizer=tokenizer)

    def forward(self, input_ids_with_pattern, statement_mask, labels=None, func_labels=None, adj=None,func_index=None):
        statement_mask = statement_mask[:, :self.args.num_labels]
        input_shape = input_ids_with_pattern.size()
        input_ids_with_pattern = input_ids_with_pattern.view(-1, input_shape[-1])  

        token_mask = input_ids_with_pattern.ne(self.tokenizer.pad_token_id).to(self.args.device)

        token_rep = self.token_aggregation_encoder(input_ids=input_ids_with_pattern, attention_mask=token_mask).last_hidden_state 

        token_rep = token_rep.view(input_shape[0], input_shape[1], input_shape[2], -1)   
        statement_embeds = token_rep[:, :, 0, :]  
        # 注册钩子函数
        a = hook_t5_mask_back_2layers.hook_class(self.statement_encoder, adj, self.statement_encoder_config)
        hook_T5Attention_handle = a.hookencoder()
        rep_xfmr = self.statement_encoder(inputs_embeds=statement_embeds,attention_mask=statement_mask).last_hidden_state
        hook_T5Attention_handle.remove()

        if self.training:
            logits, func_logits = self.classifier(rep_xfmr,statement_mask)
            loss_fct = nn.CrossEntropyLoss()
            statement_loss = loss_fct(logits, labels)
            func_loss = loss_fct(func_logits, func_labels)

            return statement_loss, func_loss
        else:
            logits, func_logits = self.classifier(rep_xfmr,statement_mask)
            probs = torch.sigmoid(logits)
            func_probs = torch.softmax(func_logits, dim=-1)

            return probs, func_probs





