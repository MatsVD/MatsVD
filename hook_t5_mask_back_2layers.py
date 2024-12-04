import torch
from torch import nn
import transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5EncoderModel, T5Config


class hook_class():
    def __init__(self, t5encoder, adj, config):
        self.t5encoder = t5encoder
        self.adj = adj
        self.config = config
        self.no_masked_position_bias = None
        self.hook_get_position_bias_handle = None

    def hookencoder(self):
        # 在第0层layer中设置hook函数，获取 no_masked_position_bias
        for layer in self.t5encoder.encoder.block[0].layer[0].modules():
            if isinstance(layer, transformers.models.t5.modeling_t5.T5Attention):
                self.hook_get_position_bias_handle = layer.register_forward_pre_hook(self.hook_get_position_bias)

        # 遍历t5encoder中的第2个encoder block中的第0个子层，注册T5Attention类的钩子函数
        # 在第2个encoder block的forward函数执行后，替换它的计算结果，相当于在第2个block中使用了adj掩盖了注意力
        for layer in self.t5encoder.encoder.block[2].layer[0].modules():
            if isinstance(layer, transformers.models.t5.modeling_t5.T5Attention):
                hook_T5Attention_handle = layer.register_forward_hook(self.hook_T5Attention_forward)
                return hook_T5Attention_handle

    def hook_get_position_bias(self, module, input):
        key_value_states = None
        position_bias = None

        hidden_states = input[0]

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        if position_bias is None:
            self.no_masked_position_bias = module.compute_bias(real_seq_length, key_length, device=hidden_states.device)
            # 没有加掩盖的 position_bias的大小为(1, n_heads, seq_length, key_length)
            # print(self.no_masked_position_bias)

        if self.no_masked_position_bias is not None:
            self.hook_get_position_bias_handle.remove()


    def hook_T5Attention_forward(self, module, input, output):

        # print("in hook_T5Attention_forward: ")

        hidden_states = input[0]
        dtype = hidden_states.dtype

        if self.adj.dim() == 3:  # adj 的大小为 64*155*155
            my_extended_attention_mask = self.adj[:, None, :, :]  # 大小为64*1*155*155
        my_extended_attention_mask = my_extended_attention_mask.to(dtype=dtype)
        my_extended_attention_mask = (1.0 - my_extended_attention_mask) * torch.finfo(dtype).min

        mask = my_extended_attention_mask
        key_value_states = None
        past_key_value = None
        layer_head_mask = None
        query_length = None
        use_cache = False
        output_attentions = self.config.output_attentions

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, module.n_heads, module.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, module.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        query_states = shape(module.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        key_states = project(hidden_states, module.k, key_value_states,
                             past_key_value[0] if past_key_value is not None else None)

        value_states = project(hidden_states, module.v, key_value_states,
                               past_key_value[1] if past_key_value is not None else None)   # (batch_size, n_heads, seq_length, dim_per_head)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # (batch_size, n_heads, seq_length, seq_length)

        '''
        self.no_masked_position_bias 的大小为 (1, n_heads, seq_length, key_length), 
        这里的 mask 已经被替换为使用 adj 计算出的 my_extended_attention_mask，大小为 (batch_size, 1, seq_length, seq_length)
        '''
        if self.no_masked_position_bias is not None:
            if mask is not None:
                position_bias = self.no_masked_position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if module.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(module.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # print(position_bias_masked)
        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores)  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=module.dropout, training=module.training
        )

        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = module.o(attn_output)

        present_key_value_state = (key_states, value_states) if (module.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


