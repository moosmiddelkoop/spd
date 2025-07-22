# Model definitions for modular addition experiments
# Vendored from progress-measures-paper repository
# https://github.com/mechanistic-interpretability-grokking/progress-measures-paper
# 
# NOTE: This file has been modified from the original to use standard PyTorch modules
# (nn.Linear, nn.Embedding) instead of raw nn.Parameter for SPD compatibility.
# The mathematical behavior is preserved see test_model.py for details.

import random
from dataclasses import dataclass

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen = True)
class Config:
    lr: float = 1e-3
    weight_decay: float = 1.0
    p: int = 113
    d_model: int = 128
    fn_name: str = 'add'  # ['add', 'subtract', 'x2xyy2','rand']
    frac_train: float = 0.3
    num_epochs: int = 50000
    save_models: bool = False
    save_every: int = 100
    stopping_thresh: int = -1
    seed: int = 0

    num_layers: int = 1
    batch_style: str = 'full'
    d_vocab: int = p+1
    n_ctx: int = 3
    d_mlp: int = 4*d_model
    num_heads: int = 4

    act_type: str = 'ReLU'  # ['ReLU', 'GeLU']
    device: t.device = t.device("cuda")
    use_ln: bool = False
    take_metrics_every_n_epochs: int = 1000

    @property
    def d_head(self):
        return self.d_model // self.num_heads

    @property
    def random_answers(self):
        return np.random.randint(low=0, high=self.p, size=(self.p, self.p))

    @property 
    def fns_dict(self):
        return {
            'add': lambda x,y:(x+y) % self.p,
            'subtract': lambda x,y:(x-y) % self.p,
            'x2xyy2': lambda x,y:(x**2+x*y+y**2) % self.p,
            'rand': lambda x,y:self.random_answers[x][y]
            }

    @property
    def fn(self):
        return self.fns_dict[self.fn_name]

    def is_train_is_test(self, train):
        '''Creates an array of Boolean indices according to whether each data point is in train or test.
        Used to index into the big batch of all possible data'''
        is_train = []
        is_test = []
        for x in range(self.p):
            for y in range(self.p):
                if (x, y, 113) in train:
                    is_train.append(True)
                    is_test.append(False)
                else:
                    is_train.append(False)
                    is_test.append(True)
        is_train = np.array(is_train)
        is_test = np.array(is_test)
        return (is_train, is_test)

class HookPoint(nn.Module):
    '''A helper class to get access to intermediate activations'''
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x

class Embed(nn.Module):
    '''Define network architecture
    I defined my own transformer from scratch so I'd fully understand each component 
    - I expect this wasn't necessary or particularly important, and a bunch of this replicates existing Pyt functionality
    '''
    def __init__(self, d_vocab, d_model):
        super().__init__()
        # Use nn.Embedding for SPD compatibility
        self.embedding = nn.Embedding(d_vocab, d_model)
        # Initialize to match original initialization
        with t.no_grad():
            self.embedding.weight.data = (t.randn(d_vocab, d_model) / np.sqrt(d_model))
    
    def forward(self, x):
        return self.embedding(x)

#| export
class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        # Use nn.Linear for SPD compatibility  
        self.linear = nn.Linear(d_model, d_vocab, bias=False)
        # Initialize to match original initialization
        with t.no_grad():
            self.linear.weight.data = (t.randn(d_vocab, d_model) / np.sqrt(d_vocab))
    
    def forward(self, x):
        return self.linear(x)

#| export
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        # Use nn.Embedding for SPD compatibility
        self.pos_embedding = nn.Embedding(max_ctx, d_model)
        # Initialize to match original
        with t.no_grad():
            self.pos_embedding.weight.data = (t.randn(max_ctx, d_model) / np.sqrt(d_model))
    
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        positions = t.arange(seq_len, device=x.device)
        # Expand positions to have batch dimension for SPD compatibility
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)  # (batch_size, seq_len, d_model)
        return x + pos_emb

class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

#| export
class SingleHeadedAttention(nn.Module):
    """A single attention head with separate Q, K, V, O projections."""
    
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        
        # Separate linear layers for Q, K, V, O - SPD can target these individually
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.o_proj = nn.Linear(d_head, d_model, bias=False)  # Output projection back to d_model
        
        # Initialize to match original
        with t.no_grad():
            self.q_proj.weight.data = (t.randn(d_head, d_model) / np.sqrt(d_model))
            self.k_proj.weight.data = (t.randn(d_head, d_model) / np.sqrt(d_model))
            self.v_proj.weight.data = (t.randn(d_head, d_model) / np.sqrt(d_model))
            self.o_proj.weight.data = (t.randn(d_model, d_head) / np.sqrt(d_model))
        
        # Hook points for this head
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()
    
    def forward(self, x, mask):
        """Forward pass for a single attention head.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (seq_len, seq_len)
            
        Returns:
            Head output (batch, seq_len, d_model)
        """
        q = self.hook_q(self.q_proj(x))  # (batch, seq_len, d_head)
        k = self.hook_k(self.k_proj(x))  # (batch, seq_len, d_head)
        v = self.hook_v(self.v_proj(x))  # (batch, seq_len, d_head)
        
        # Compute attention scores
        scores = t.einsum('bqd,bkd->bqk', q, k)  # (batch, seq_len, seq_len)
        scores = self.hook_attn_pre(scores / np.sqrt(self.d_head))
        
        # Apply causal mask
        seq_len = x.shape[1]
        scores = t.tril(scores) - 1e10 * (1 - mask[:seq_len, :seq_len])
        
        # Softmax and apply to values
        attn_weights = self.hook_attn(F.softmax(scores, dim=-1))  # (batch, seq_len, seq_len)
        z = self.hook_z(t.einsum('bqk,bkd->bqd', attn_weights, v))  # (batch, seq_len, d_head)
        
        # Output projection
        output = self.o_proj(z)  # (batch, seq_len, d_model)
        
        return output


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.num_heads = num_heads
        
        # Create separate attention heads - SPD can target each individually
        self.heads = nn.ModuleList([
            SingleHeadedAttention(d_model, d_head) for _ in range(num_heads)
        ])
        
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        
        # Hook points for the overall attention output
        self.hook_attn_out = HookPoint()

    def forward(self, x):
        # Run each attention head separately
        head_outputs = []
        for head in self.heads:
            head_out = head(x, self.mask)  # Each head outputs (batch, seq_len, d_model)
            head_outputs.append(head_out)
        
        # Sum head outputs (mathematically equivalent to original concat + linear)
        combined_output = sum(head_outputs)  # (batch, seq_len, d_model)
        
        return self.hook_attn_out(combined_output)

#| export
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.act_type = act_type
        
        # Use nn.Linear for SPD compatibility
        self.W_in = nn.Linear(d_model, d_mlp, bias=True)
        self.W_out = nn.Linear(d_mlp, d_model, bias=True)
        
        # Initialize to match original
        with t.no_grad():
            self.W_in.weight.data = (t.randn(d_mlp, d_model) / np.sqrt(d_model))
            self.W_in.bias.data = t.zeros(d_mlp)
            self.W_out.weight.data = (t.randn(d_model, d_mlp) / np.sqrt(d_model))
            self.W_out.bias.data = t.zeros(d_model)
        
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = self.hook_pre(self.W_in(x))
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = self.W_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.hook_resid_pre(x))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.embed = Embed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.pos_embed = PosEmbed(max_ctx = config.n_ctx, d_model = config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model = config.d_model,
            d_mlp = config.d_mlp,
            d_head = config.d_head,
            num_heads = config.num_heads,
            n_ctx = config.n_ctx,
            act_type = config.act_type,
            model=[self]) for i in range(config.num_layers)])
        self.unembed = Unembed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

def gen_train_test(config: Config):
    '''Generate train and test split'''
    num_to_generate = config.p
    pairs = [(i, j, num_to_generate) for i in range(num_to_generate) for j in range(num_to_generate)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.frac_train*len(pairs))
    return pairs[:div], pairs[div:]