import torch.nn as nn
from math import sqrt
from transformers import AutoTokenizer,AutoConfig
from bertviz.transformers_neuron_view import BertModel
import torch.nn.functional as F
import torch

def scaled_dot_product_attention(query, key, value):
  dim_k = query.size(-1)
  # batch matrix-matrix product
  scores = torch.bmm(query,key.transpose(1,2))/sqrt(dim_k)
  attention_weights = F.softmax(scores,dim=-1)
  return torch.bmm(attention_weights,value)


class AttentionHead(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super().__init__()
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)

  def forward(self, hidden_state):
    attention_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
    return attention_outputs

class MultiHeadAttention(nn.Module):
  def __init__(self,config):
    super().__init__()
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embed_dim//num_heads

    self.heads = nn.ModuleList(
      [AttentionHead(embed_dim,head_dim) for _ in range(num_heads)]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)
    # print(f"output_linear: {self.output_linear}")
  def forward(self,hidden_state):
    x =torch.cat(
      [ h(hidden_state) for h in self.heads], dim=-1
    )
    # print(f"MultiHeadAttention: x.shape={x.shape}")

    x= self.output_linear(x)
    return x


class FeedForward(nn.Module):
  def __init__(self,config):
    super().__init__()
    print(f"input layer size: {config.hidden_size}")
    print(f"first layer size: {config.intermediate_size}")
    self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
  def forward(self,x):
    x = self.linear_1(x)
    x = self.gelu(x)
    x = self.linear_2(x)
    x = self.dropout(x)
    return x

class TransformerEncoderLayer(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
    self.attention = MultiHeadAttention(config)
    self.FeedForward = FeedForward(config)

  def forward(self,x):
    hidden_state = self.layer_norm_1(x)
    x = x+self.attention(hidden_state)
    x = x+ self.FeedForward(self.layer_norm_2(x))
    return x

#learnable embeddings
class Embeddings (nn.Module):
  def __init__(self,config):
    super().__init__()
    self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
    self.dropout = nn.Dropout()
  
  def forward(self,input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype= torch.long).unsqueeze(0)


if __name__ == "__main__":
  model_ckpt = "bert-base-uncased"
  tokenizer  = AutoTokenizer.from_pretrained(model_ckpt)
  # model = BertModel.from_pretrained(model_ckpt)
  config = AutoConfig.from_pretrained(model_ckpt) # bert-base-uncased
  token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

  text  = "time flies like an arrow"
  inputs = tokenizer(text,return_tensors="pt",add_special_tokens=False)
  inputs_embeds = token_emb(inputs.input_ids)

  multihead_attn = MultiHeadAttention(config)
  attn_output = multihead_attn(inputs_embeds)
  print(f"attention output: {attn_output.size()}")
