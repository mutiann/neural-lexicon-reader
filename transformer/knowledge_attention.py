import torch
from torch import nn
from torch.nn import functional as F

from transformer.attention import split_heads, combine_heads, MultiheadAttention
from transformer.modules import FFNLayer
from transformer.common import attention_bias, get_sinusoid_encoding_table, impute


def gather_by_index(x, index):
    '''
    Args:
        x: [n_entry, ...]
        index: [...] -> (0, entry)

    Returns: [..., ...], a tensor selected from the contexts x according to the index
    '''
    selected_x = torch.index_select(x.view(x.shape[0], -1), dim=0, index=index.view([-1]))
    return selected_x.view(index.shape + x.shape[1:])


class KnowledgeAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_heads, dropout_rate=0.1):
        super(KnowledgeAttention, self).__init__()
        self.q_transform = nn.Linear(query_size, query_size, bias=False)
        self.k_transform = nn.Linear(key_size, query_size, bias=False)
        self.v_transform = nn.Linear(value_size, query_size, bias=False)
        self.output_transform = nn.Linear(query_size, query_size, bias=False)
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, values, bias):
        """

        Args:
            queries: [batch, length_q, query_size]
            keys: [batch, length_q, length_k, key_size]
            values: [batch, length_q, length_k, value_size]
            bias: [batch, length_q, length_k]
        Returns:
            context: [batch, length_q, query_size]
            align: [batch, num_heads, length_k, length_q]
        """
        q = self.q_transform(queries)
        k = self.k_transform(keys)
        k = k.reshape([k.shape[0], -1, k.shape[-1]])
        v = self.v_transform(values)
        v = v.reshape([v.shape[0], -1, v.shape[-1]])
        q = split_heads(q, self.num_heads) # [batch, num_heads, length_q, depth_k']
        k = split_heads(k, self.num_heads) # [batch, num_heads, length_q * length_k, depth_k']
        k = k.reshape([k.shape[0], self.num_heads, q.shape[2], -1, k.shape[-1]])
        v = split_heads(v, self.num_heads) # [batch, num_heads, length_q * length_k, depth_v']
        v = v.reshape([v.shape[0], self.num_heads, q.shape[2], -1, v.shape[-1]])
        key_depth_per_head = self.key_size // self.num_heads
        q = q * key_depth_per_head ** -0.5
        logits = torch.matmul(k, q.unsqueeze(-1)).squeeze(-1) # [batch, num_heads, length_q, length_k]
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        align = weights.permute(0, 1, 3, 2)
        weights = self.attn_dropout(weights)
        context = torch.matmul(weights.unsqueeze(-2), v).squeeze(-2)

        context = combine_heads(context)
        context = self.output_transform(context)
        return {'outputs': context, 'align': align}

class KnowledgeableTransformerEncoder(nn.Module):
    def __init__(self, input_size, hparams):
        super(KnowledgeableTransformerEncoder, self).__init__()
        self.self_attentions = nn.ModuleList()
        self.attn_layer_norms = nn.ModuleList()
        self.knowledge_attentions = nn.ModuleList()
        self.knowledge_layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_layer_norms = nn.ModuleList()
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(hparams.transformer_dropout_rate)
        self.hp = hparams

        if hparams.use_key_encoder > 0:
            import copy
            from transformer.modules import TransformerEncoder
            hp_ = copy.copy(hparams)
            hp_.n_encoder_layer = hparams.key_encode_layers
            self.key_linear = FFNLayer(hparams.knowledge_key_size, hparams.knowledge_key_size, hparams.encoder_hidden)
            self.key_encoder = TransformerEncoder(hparams.encoder_hidden, hp_)
            key_size = hparams.encoder_hidden
        else:
            key_size = hparams.knowledge_key_size

        hidden_size = hparams.encoder_hidden
        for layer in range(hparams.n_encoder_layer):
            in_size = input_size if layer == 0 else hidden_size
            self.attn_layer_norms.append(nn.LayerNorm(in_size, eps=1e-6))
            self.self_attentions.append(MultiheadAttention(key_size=in_size,
                                                           value_size=in_size,
                                                           is_self_attention=True,
                                                           num_heads=hparams.n_attention_head,
                                                           dropout_rate=hparams.transformer_dropout_rate))

            if hparams.knowledge_start_layer <= layer < hparams.knowledge_end_layer:
                self.knowledge_layer_norms.append(nn.LayerNorm(in_size, eps=1e-6))
                self.knowledge_attentions.append(KnowledgeAttention(key_size=key_size,
                                                                    value_size=hparams.knowledge_value_size,
                                                                    query_size=hidden_size,
                                                                    num_heads=hparams.knowledge_attention_head,
                                                                    dropout_rate=hparams.transformer_dropout_rate))
            else:
                self.knowledge_layer_norms.append(nn.Identity())
                self.knowledge_attentions.append(nn.Identity())

            self.ffn_layer_norms.append(nn.LayerNorm(hidden_size, eps=1e-6))
            self.ffn_layers.append(FFNLayer(hidden_size, hidden_size * 4, hidden_size,
                                            dropout_rate=hparams.transformer_dropout_rate))

        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def prepare_inputs(self, inputs, input_lengths, contexts, context_lengths):
        mask = torch.arange(inputs.shape[1], device=input_lengths.device)[None, :] < input_lengths[:, None]
        inputs = inputs * mask.unsqueeze(-1)
        bias = attention_bias(mask, "masking").to(inputs.device)

        mask = torch.arange(contexts.shape[1], device=context_lengths.device)[None, :] < context_lengths[:, None]
        contexts = contexts * mask.unsqueeze(-1)
        context_bias = attention_bias(mask, "masking").to(contexts.device)

        pe = get_sinusoid_encoding_table(length=inputs.shape[1], channels=inputs.shape[2], device=inputs.device)
        inputs += pe * self.pe_scale
        inputs = self.dropout(inputs)
        return inputs, bias, contexts, context_bias

    def forward(self, inputs, input_lengths, keys, contexts, context_lengths, indices):
        '''

        Args:
            inputs: [batch, length_q, input_size]
            input_lengths: [batch]
            keys: [n_entry, length_k, key_size]
            contexts: [n_entry, length_k, value_size]
            context_lengths: [n_entry]
            indices: [batch, length_q] -> (0, n_entry)
        Returns:
            outputs: [batch, length_q]
            align: [batch, n_heads, length_q, length_context]
        '''
        if self.hp.use_key_encoder:
            keys = self.key_linear(keys)
            keys = self.key_encoder(keys, context_lengths)
            if self.hp.use_identical_key_context:
                contexts = keys
        x, x_bias, _, context_bias = self.prepare_inputs(inputs, input_lengths, contexts, context_lengths)

        keys = gather_by_index(keys, indices)  # [batch, length_q, length_k, key_size]
        contexts = gather_by_index(contexts, indices)  # [batch, length_q, length_k, value_size]
        # [batch, 1, length_q, length_k]
        context_bias = gather_by_index(context_bias, indices).squeeze(2).squeeze(2).unsqueeze(1)

        attn_align = []
        for i in range(len(self.self_attentions)):
            y = self.self_attentions[i](queries=self.attn_layer_norms[i](x),
                                        memories=None,
                                        bias=x_bias)
            x = x + self.dropout(y["outputs"])

            if self.hp.knowledge_start_layer <= i < self.hp.knowledge_end_layer:
                y = self.knowledge_attentions[i](queries=self.knowledge_layer_norms[i](x),
                                                 keys=keys, values=contexts,
                                                 bias=context_bias)
                attn_align.append(y['align'])
                x = x + self.dropout(y["outputs"])

            y = self.ffn_layers[i](self.ffn_layer_norms[i](x))
            x = x + self.dropout(y)
        outputs = self.output_layer_norm(x)

        outputs = impute(outputs, input_lengths)
        return outputs, attn_align
