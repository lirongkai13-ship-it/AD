"""
方案B: 并行时空USAD + MTAD-like 全局时间分支 + 节点广播

与 parallel_usad_prior 唯一区别:
  时间分支从 per-variable Conv1d 改为 GRU过[B,W,N] + 节点广播

temporal_mode:
  "per_variable" = 原版逐变量Conv1d
  "mtad_global"   = 新GRU全局+广播 (default)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


# ═══════════════════════════════════════════════════════════════
# 1. Dynamic Pearson + Prior Boost Fusion (不变)
# ═══════════════════════════════════════════════════════════════
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
        if em.any():
            s,d=torch.where(em); return torch.stack([s,d],0)
        return torch.zeros(2,0,dtype=torch.long,device=x.device)


# ═══════════════════════════════════════════════════════════════
# 2a. 原版 per-variable 时间编码 (保留为可选)
# ═══════════════════════════════════════════════════════════════
class PerVariableEncoder(nn.Module):
    def __init__(self, window_size=60, out_dim=32, hidden_channels=16, dropout=0.2):
        super().__init__()
        self.c1=nn.Conv1d(1,hidden_channels,3,padding=1)
        self.c2=nn.Conv1d(hidden_channels,out_dim,3,padding=1)
        self.drop=nn.Dropout(dropout); self.pool=nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        h=F.relu(self.c1(x)); h=self.drop(h); h=F.relu(self.c2(h))
        return self.pool(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# 2b. MTAD-like 全局时间编码 [NEW]
# ═══════════════════════════════════════════════════════════════
class MTADLikeGlobalTemporalEncoder(nn.Module):
    """GRU over [B,W,N] → h_temp [B,d] — 只处理B条序列, 不展开B*N"""
    def __init__(self, num_variables=51, window_size=60, out_dim=32,
                 temporal_hidden=64, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(num_variables, temporal_hidden, num_layers=1,
                          batch_first=True, bidirectional=False)
        self.proj = nn.Sequential(
            nn.Linear(temporal_hidden, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        """x: [B, W, N] → h_temp: [B, out_dim]"""
        _, h_last = self.gru(x)           # [1, B, temporal_hidden]
        return self.proj(h_last[-1])       # [B, out_dim]


# ═══════════════════════════════════════════════════════════════
# 3. 节点广播 + 节点嵌入差异化 [NEW]
# ═══════════════════════════════════════════════════════════════
class TemporalNodeProjector(nn.Module):
    """h_temp [B,d] + NodeEmbed [N,d] → H_time [B,N,d]"""
    def __init__(self, num_variables=51, d=32, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Embedding(num_variables, d)
        # concat(global_temp, node_embed) → 2d → d
        self.fuse = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, h_temp, device):
        """h_temp: [B,d] → H_time: [B,N,d]"""
        b = h_temp.shape[0]
        n = self.node_embed.num_embeddings
        # 广播全局特征到每个节点
        h_global = h_temp.unsqueeze(1).expand(b, n, -1)        # [B,N,d]
        # 节点嵌入
        ids = torch.arange(n, device=device)
        e_node = self.node_embed(ids).unsqueeze(0).expand(b, n, -1)  # [B,N,d]
        # 融合
        return self.fuse(torch.cat([h_global, e_node], dim=-1))     # [B,N,d]


# ═══════════════════════════════════════════════════════════════
# 4. PriorNodeEmbedding, NodeLevelFusion (不变)
# ═══════════════════════════════════════════════════════════════
class PriorNodeEmbedding(nn.Module):
    def __init__(self,nv,pei,pw,hd):
        super().__init__()
        P=torch.zeros(nv,nv)
        for i in range(pei.shape[1]): s,d=pei[0,i].item(),pei[1,i].item(); P[s,d]=max(P[s,d],pw[i].item())
        P_n=P/P.sum(1,keepdim=True).clamp(1)
        self.register_buffer("P_norm",P_n)
        self.node_embed=nn.Parameter(torch.randn(nv,hd)*0.1); self.proj=nn.Linear(hd,hd)
    def forward(self): return self.proj(torch.matmul(self.P_norm,self.node_embed))

class NodeLevelFusion(nn.Module):
    def __init__(self,ds=32,dt=32,df=32,dropout=0.2):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(ds+dt,df),nn.ReLU(),nn.Dropout(dropout),nn.Linear(df,df))
    def forward(self,hs,ht): return self.mlp(torch.cat([hs,ht],-1))


# ═══════════════════════════════════════════════════════════════
# 5. ParallelSpatioTemporalEncoder (方案B版)
# ═══════════════════════════════════════════════════════════════
class ParallelSpatioTemporalEncoder(nn.Module):
    def __init__(self, num_variables=51, window_size=60,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3,
                 temporal_mode="mtad_global"):
        super().__init__()
        self.nv=num_variables; self.ws=window_size
        self.use_flatten=use_flatten; self.temporal_mode=temporal_mode

        # ── 空间 (不变) ──
        self.dyn_graph=DynamicPearsonPriorGraph(num_variables,prior_edge_index,prior_weights,boost=boost)
        self.prior_embed=PriorNodeEmbedding(num_variables,prior_edge_index,prior_weights,hidden_dim)
        self.gat=GATv2Block(window_size,hidden_dim,hidden_dim,heads=gat_heads,dropout=dropout)
        self.gate=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.Sigmoid())

        # ── 时间分支 ──
        if temporal_mode=="mtad_global":
            self.temporal_enc = MTADLikeGlobalTemporalEncoder(
                num_variables=num_variables, window_size=window_size,
                out_dim=hidden_dim, temporal_hidden=hidden_dim*2, dropout=dropout)
            self.temporal_projector = TemporalNodeProjector(
                num_variables=num_variables, d=hidden_dim, dropout=dropout)
        else:
            self.temporal_enc = PerVariableEncoder(
                window_size=window_size, out_dim=hidden_dim,
                hidden_channels=hidden_dim//2, dropout=dropout)
            self.temporal_projector = None

        # ── 融合 ──
        self.fusion=NodeLevelFusion(hidden_dim,hidden_dim,hidden_dim,dropout)

        # ── Latent ──
        if use_flatten:
            fd=num_variables*hidden_dim; md=fd//16
            self.latent_proj=nn.Sequential(nn.Linear(fd,md),nn.ReLU(),nn.Linear(md,latent_dim))
        else:
            self.latent_proj=nn.Linear(hidden_dim,latent_dim)

    def forward(self, x):
        b,w,n=x.shape

        # ── 空间 ──
        edges=self.dyn_graph(x)
        hg=self.gat(x.permute(0,2,1),edges)
        hp=self.prior_embed().unsqueeze(0).expand(b,-1,-1)
        g=self.gate(torch.cat([hg,hp],-1)); hs=g*hg+(1-g)*hp

        # ── 时间 ──
        if self.temporal_mode=="mtad_global":
            h_temp=self.temporal_enc(x)                          # [B,d]
            ht=self.temporal_projector(h_temp, x.device)          # [B,N,d]
        else:
            xv=x.permute(0,2,1).reshape(b*n,1,w)
            ht=self.temporal_enc(xv).reshape(b,n,-1)              # [B,N,d]

        # ── 融合 ──
        hf=self.fusion(hs,ht)
        z=hf.reshape(b,n*hf.shape[-1]) if self.use_flatten else hf.mean(1)
        return self.latent_proj(z), edges


# ═══════════════════════════════════════════════════════════════
# 6. ParallelB_USAD
# ═══════════════════════════════════════════════════════════════
class ParallelB_USAD(nn.Module):
    def __init__(self, num_variables, window_size, static_edge_index,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2,
                 latent_dim=64, use_flatten=True, temporal_mode="mtad_global"):
        super().__init__()
        df=window_size*num_variables
        self.encoder=ParallelSpatioTemporalEncoder(
            num_variables=num_variables, window_size=window_size,
            prior_edge_index=prior_edge_index, prior_weights=prior_weights,
            hidden_dim=hidden_dim, gat_heads=gat_heads, dropout=dropout,
            latent_dim=latent_dim, use_flatten=use_flatten, temporal_mode=temporal_mode)
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
