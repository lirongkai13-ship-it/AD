"""
三分支并行时空编码 + 残差门控融合

encoder_mode:
  "base_parallel"         = H_node + H_space → fusion → USAD
  "tri_branch_residual_gate" = H_node + H_space + H_global(门控残差) → USAD

temporal_mode (控制 H_node):
  "per_variable_conv"              = A: Conv1d(k=3)×2
  "per_variable_dilated_conv"      = B: Dilated Conv dil=1,2,4
  "per_variable_residual_ms_tcn"   = C: Conv+残差TCN dil=1,2,4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


# ═══════════════════════════════════════════════════
# Dynamic Pearson + Prior Boost (不变)
# ═══════════════════════════════════════════════════
class DynamicPearsonPriorGraph(nn.Module):
    def __init__(self, n_vars, prior_edge_index, prior_weights, boost=0.3, threshold=0.3):
        super().__init__()
        self.n_vars=n_vars; self.boost=boost; self.threshold=threshold
        A=torch.zeros(n_vars,n_vars)
        for i in range(prior_edge_index.shape[1]):
            s,d=prior_edge_index[0,i].item(),prior_edge_index[1,i].item()
            w=prior_weights[i].item()
            if s<n_vars and d<n_vars: A[s,d]=max(A[s,d],w); A[d,s]=max(A[d,s],w)
        self.register_buffer("A_prior",A); self.register_buffer("prior_mask",A>0)

    def forward(self,x):
        b,w,n=x.shape; xc=x-x.mean(dim=1,keepdim=True)
        cov=torch.bmm(xc.transpose(1,2),xc)/(w-1)
        std=torch.sqrt(torch.var(x,dim=1,unbiased=True)+1e-8)
        C=torch.nan_to_num(cov/(std.unsqueeze(1)*std.unsqueeze(2)+1e-8),nan=0.0)
        Ad=C.abs().mean(0); Af=Ad.clone()
        ov=self.prior_mask.to(x.device)&(Ad>=self.threshold)
        op=self.prior_mask.to(x.device)&(Ad<self.threshold)
        bv=torch.tensor(self.boost,dtype=Af.dtype,device=Af.device)
        Af[ov]=Af[ov]+bv; Af[op]=self.A_prior.to(device=x.device,dtype=Af.dtype)[op]
        dm=~torch.eye(n,dtype=torch.bool,device=x.device)
        em=(Af.abs()>=self.threshold)&dm
        if em.any(): s,d=torch.where(em); return torch.stack([s,d],0)
        return torch.zeros(2,0,dtype=torch.long,device=x.device)


class PriorNodeEmbedding(nn.Module):
    def __init__(self,nv,pei,pw,hd):
        super().__init__()
        P=torch.zeros(nv,nv)
        for i in range(pei.shape[1]): s,d=pei[0,i].item(),pei[1,i].item(); P[s,d]=max(P[s,d],pw[i].item())
        P_n=P/P.sum(1,keepdim=True).clamp(1)
        self.register_buffer("P_norm",P_n)
        self.node_embed=nn.Parameter(torch.randn(nv,hd)*0.1); self.proj=nn.Linear(hd,hd)
    def forward(self): return self.proj(torch.matmul(self.P_norm,self.node_embed))


# ═══════════════════════════════════════════════════
# Branch 1: Per-variable temporal (A/B/C, 全保留)
# ═══════════════════════════════════════════════════
class PerVariableConv(nn.Module):
    def __init__(self,ws=60,out_dim=32,hidden=16,dropout=0.2):
        super().__init__()
        self.c1=nn.Conv1d(1,hidden,3,padding=1); self.c2=nn.Conv1d(hidden,out_dim,3,padding=1)
        self.drop=nn.Dropout(dropout); self.pool=nn.AdaptiveAvgPool1d(1)
    def forward(self,x): h=F.relu(self.c1(x)); h=self.drop(h); h=F.relu(self.c2(h)); return self.pool(h).squeeze(-1)

class PerVariableDilatedConv(nn.Module):
    def __init__(self,ws=60,out_dim=32,hidden=16,dropout=0.2):
        super().__init__()
        h=hidden//3+1
        self.b1=nn.Conv1d(1,h,3,padding=1,dilation=1); self.b2=nn.Conv1d(1,h,3,padding=2,dilation=2)
        self.b3=nn.Conv1d(1,h,3,padding=4,dilation=4); self.pool=nn.AdaptiveAvgPool1d(1)
        self.fuse=nn.Sequential(nn.Dropout(dropout),nn.Linear(h*3,out_dim))
    def forward(self,x):
        h1=self.pool(F.relu(self.b1(x))).squeeze(-1); h2=self.pool(F.relu(self.b2(x))).squeeze(-1)
        h3=self.pool(F.relu(self.b3(x))).squeeze(-1); return self.fuse(torch.cat([h1,h2,h3],-1))

class ResidualMSTCN(nn.Module):
    def __init__(self,ic,oc,ks=3,dil=1,drop=0.1):
        super().__init__()
        p=(ks-1)*dil//2
        self.c1=nn.Conv1d(ic,oc,ks,padding=p,dilation=dil); self.c2=nn.Conv1d(oc,oc,1)
        self.drop=nn.Dropout(drop); self.res=nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(self,x): r=self.res(x); h=F.relu(self.c1(x)); h=self.drop(h); return F.relu(self.c2(h))+r

class PerVariableResidualMSTCN(nn.Module):
    def __init__(self,ws=60,out_dim=32,hidden=12,dropout=0.1):
        super().__init__()
        self.base=nn.Conv1d(1,hidden,3,padding=1)
        self.tcn1=ResidualMSTCN(hidden,hidden,3,1,dropout); self.tcn2=ResidualMSTCN(hidden,hidden,3,2,dropout)
        self.tcn3=ResidualMSTCN(hidden,hidden,3,4,dropout); self.gamma=nn.Parameter(torch.tensor(0.1))
        self.pool=nn.AdaptiveAvgPool1d(1); self.fuse=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden,out_dim))
    def forward(self,x):
        h=F.relu(self.base(x)); h=h+self.gamma*self.tcn3(self.tcn2(self.tcn1(h)))
        return self.fuse(self.pool(h).squeeze(-1))


# ═══════════════════════════════════════════════════
# Branch 3: 全局时间注意力 (MTAD-like)
# ═══════════════════════════════════════════════════
class GlobalTemporalAttentionEncoder(nn.Module):
    """时间步之间 self-attention → 聚合 → 广播到节点 [B,N,d]"""
    def __init__(self, num_variables=51, window_size=60, d=32, d_attn=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(num_variables, d_attn)   # N → d_attn
        self.attn = nn.MultiheadAttention(d_attn, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_attn)
        # 时间聚合 → 节点级
        self.proj_out = nn.Sequential(
            nn.Linear(d_attn, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        """x:[B,W,N] → [B,N,d]"""
        b,w,n = x.shape
        h = self.proj_in(x)                           # [B, W, d_attn]
        h_attn, _ = self.attn(h, h, h)                # [B, W, d_attn]
        h = self.norm(h + h_attn)                     # 残差+LN
        h_pool = h.mean(dim=1)                         # [B, d_attn] 时间聚合
        h_global = self.proj_out(h_pool)               # [B, d]
        return h_global.unsqueeze(1).expand(b, n, -1)  # 广播到 [B, N, d]


# ═══════════════════════════════════════════════════
# 残差门控融合
# ═══════════════════════════════════════════════════
class ResidualGatedFusion(nn.Module):
    """H_fuse = H_base + gamma * gate * H_global
       gate = gate_scale * sigmoid(MLP(H_global))
       gate_scale limits max injection strength of global branch"""
    def __init__(self, d=32, gamma_init=0.05, dropout=0.1, gamma_mode='fixed', gate_scale=1.0):
        super().__init__()
        self.gamma_mode = gamma_mode
        self.gate_scale = gate_scale
        if gamma_mode == 'learnable':
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))
        self.gate_mlp = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, d))

    def forward(self, h_base, h_global):
        raw_gate = torch.sigmoid(self.gate_mlp(h_global))
        g = self.gate_scale * raw_gate
        return h_base + self.gamma * g * h_global


# ═══════════════════════════════════════════════════
# 主编码器
# ═══════════════════════════════════════════════════
class TriBranchEncoder(nn.Module):
    def __init__(self, nv=51, ws=60, prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3,
                 temporal_mode="per_variable_conv",
                 encoder_mode="base_parallel",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0):
        super().__init__()
        self.nv=nv; self.ws=ws; self.use_flatten=use_flatten
        self.temporal_mode=temporal_mode; self.encoder_mode=encoder_mode
        self.gate_scale=gate_scale

        # ── Branch 1: Per-variable temporal ──
        if temporal_mode=="per_variable_conv":
            self.temporal_enc=PerVariableConv(ws,hidden_dim,hidden_dim//2,dropout)
        elif temporal_mode=="per_variable_dilated_conv":
            self.temporal_enc=PerVariableDilatedConv(ws,hidden_dim,hidden_dim//2,dropout)
        elif temporal_mode=="per_variable_residual_ms_tcn":
            self.temporal_enc=PerVariableResidualMSTCN(ws,hidden_dim,hidden_dim//2,dropout)
        else: raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        # ── Branch 2: Spatial (GATv2 + prior gate) ──
        self.dyn_graph=DynamicPearsonPriorGraph(nv,prior_edge_index,prior_weights,boost)
        self.prior_embed=PriorNodeEmbedding(nv,prior_edge_index,prior_weights,hidden_dim)
        self.gat=GATv2Block(ws,hidden_dim,hidden_dim,heads=gat_heads,dropout=dropout)
        self.gate=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.Sigmoid())

        # ── Branch 3: Global temporal attention (仅 tri_branch 模式) ──
        if encoder_mode=="tri_branch_residual_gate":
            self.global_temp=GlobalTemporalAttentionEncoder(nv,ws,hidden_dim,hidden_dim,dropout=dropout)
            self.gated_fusion=ResidualGatedFusion(hidden_dim,dropout=dropout,
                                                   gamma_mode=gamma_mode, gamma_init=gamma_value,
                                                   gate_scale=gate_scale)
        else:
            self.global_temp=None; self.gated_fusion=None

        # ── Base fusion: H_node + H_space ──
        self.base_fusion=nn.Sequential(
            nn.Linear(hidden_dim*2,hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim))

        # ── Latent ──
        if use_flatten:
            fd=nv*hidden_dim; md=fd//16
            self.latent_proj=nn.Sequential(nn.Linear(fd,md),nn.ReLU(),nn.Linear(md,latent_dim))
        else:
            self.latent_proj=nn.Linear(hidden_dim,latent_dim)

    def forward(self,x):
        b,w,n=x.shape

        # Branch 1: H_node [B,N,d]
        xv=x.permute(0,2,1).reshape(b*n,1,w)
        h_node=self.temporal_enc(xv).reshape(b,n,-1)

        # Branch 2: H_space [B,N,d]
        edges=self.dyn_graph(x)
        hg=self.gat(x.permute(0,2,1),edges)
        hp=self.prior_embed().unsqueeze(0).expand(b,-1,-1)
        g=self.gate(torch.cat([hg,hp],-1))
        h_space=g*hg+(1-g)*hp

        # Base fusion: H_node + H_space
        h_base=self.base_fusion(torch.cat([h_node,h_space],-1))  # [B,N,d]

        # Branch 3: 可选残差门控
        if self.encoder_mode=="tri_branch_residual_gate" and self.gated_fusion is not None:
            h_global=self.global_temp(x)                         # [B,N,d]
            h_fuse=self.gated_fusion(h_base,h_global)             # [B,N,d]
        else:
            h_fuse=h_base

        # Latent
        z=h_fuse.reshape(b,n*h_fuse.shape[-1]) if self.use_flatten else h_fuse.mean(1)
        return self.latent_proj(z), edges


# ═══════════════════════════════════════════════════
# 完整模型
# ═══════════════════════════════════════════════════
class TriBranch_USAD(nn.Module):
    def __init__(self, nv, ws, static_edge_index, prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, gru_hidden=32, tcn_channels=32, tcn_blocks=1,
                 dropout=0.2, latent_dim=64, use_flatten=True,
                 temporal_mode="per_variable_conv", encoder_mode="base_parallel",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0):
        super().__init__()
        df=nv*ws
        self.encoder=TriBranchEncoder(
            nv,ws,prior_edge_index,prior_weights,hidden_dim,gat_heads,dropout,
            latent_dim,use_flatten,temporal_mode=temporal_mode,encoder_mode=encoder_mode,
            gamma_mode=gamma_mode, gamma_value=gamma_value, gate_scale=gate_scale)
        dh=latent_dim*2
        self.decoder1=nn.Sequential(nn.Linear(latent_dim,dh),nn.ReLU(),nn.Dropout(dropout),nn.Linear(dh,df))
        self.decoder2=nn.Sequential(nn.Linear(latent_dim,dh),nn.ReLU(),nn.Dropout(dropout),nn.Linear(dh,df))

    def forward(self,x,edge_index):
        b,w,n=x.shape; z,edges=self.encoder(x)
        r1=self.decoder1(z).view(b,w,n); r2=self.decoder2(z).view(b,w,n)
        with torch.no_grad():
            z2,_=self.encoder(r1); z2=z2.detach()
        return r1,r2,self.decoder2(z2).view(b,w,n)

    def forward_eval(self,x,edge_index):
        b,w,n=x.shape; z,_=self.encoder(x)
        return self.decoder1(z).view(b,w,n)
