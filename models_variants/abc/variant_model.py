"""
并行时空USAD + 三模式时间分支 A/B/C

temporal_mode:
  "per_variable_conv"           = A: Conv1d(k=3)×2              (当前最佳 F1=0.7524)
  "per_variable_dilated_conv"   = B: Conv1d dil=1,2,4            (NEW)
  "per_variable_residual_ms_tcn"= C: Conv1d(k=3)+残差TCN dil=1,2,4 (NEW)

所有分支 per-variable [B*N,1,W], Conv1d only, 无 GRU/Transformer.
空间/先验/融合/Decoder 完全不变.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


# ═══════════════════════════════════════════════════
# DynamicPearsonPriorGraph, PriorNodeEmbedding, NodeLevelFusion (不变)
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
        if em.any():
            s,d=torch.where(em); return torch.stack([s,d],0)
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

class NodeLevelFusion(nn.Module):
    def __init__(self,ds=32,dt=32,df=32,dropout=0.2):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(ds+dt,df),nn.ReLU(),nn.Dropout(dropout),nn.Linear(df,df))
    def forward(self,hs,ht): return self.mlp(torch.cat([hs,ht],-1))


# ═══════════════════════════════════════════════════
# A 方案: 原版 Conv1d(k=3)×2  (F1=0.7524)
# ═══════════════════════════════════════════════════
class PerVariableConv(nn.Module):
    def __init__(self, ws=60, out_dim=32, hidden=16, dropout=0.2):
        super().__init__()
        self.c1=nn.Conv1d(1,hidden,3,padding=1)
        self.c2=nn.Conv1d(hidden,out_dim,3,padding=1)
        self.drop=nn.Dropout(dropout); self.pool=nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        h=F.relu(self.c1(x)); h=self.drop(h)
        h=F.relu(self.c2(h)); return self.pool(h).squeeze(-1)


# ═══════════════════════════════════════════════════
# B 方案: Dilated Conv dil=1,2,4
# ═══════════════════════════════════════════════════
class PerVariableDilatedConv(nn.Module):
    """三层不同膨胀率的Conv1d, 捕获多尺度时间感受野"""
    def __init__(self, ws=60, out_dim=32, hidden=16, dropout=0.2):
        super().__init__()
        h=hidden//3+1  # 每分支通道
        self.b1=nn.Conv1d(1,h,3,padding=1,dilation=1)
        self.b2=nn.Conv1d(1,h,3,padding=2,dilation=2)
        self.b3=nn.Conv1d(1,h,3,padding=4,dilation=4)
        self.pool=nn.AdaptiveAvgPool1d(1)
        concat=h*3
        self.fuse=nn.Sequential(nn.Dropout(dropout),nn.Linear(concat,out_dim))

    def forward(self,x):
        h1=self.pool(F.relu(self.b1(x))).squeeze(-1)
        h2=self.pool(F.relu(self.b2(x))).squeeze(-1)
        h3=self.pool(F.relu(self.b3(x))).squeeze(-1)
        return self.fuse(torch.cat([h1,h2,h3],-1))


# ═══════════════════════════════════════════════════
# C 方案: Residual Multi-Scale TCN (轻量)
# ═══════════════════════════════════════════════════
class ResidualMSTCN(nn.Module):
    """小残差块, 用于多尺度分支"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        p=(kernel_size-1)*dilation//2
        self.c1=nn.Conv1d(in_ch,out_ch,kernel_size,padding=p,dilation=dilation)
        self.c2=nn.Conv1d(out_ch,out_ch,1)
        self.drop=nn.Dropout(dropout)
        self.res=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()

    def forward(self,x):
        r=self.res(x)
        h=F.relu(self.c1(x)); h=self.drop(h)
        return F.relu(self.c2(h))+r


class PerVariableResidualMSTCN(nn.Module):
    """base Conv1d(k=3) + 三路残差TCN(dil=1,2,4) + gamma加权"""
    def __init__(self, ws=60, out_dim=32, hidden=12, dropout=0.1):
        super().__init__()
        self.base=nn.Conv1d(1,hidden,3,padding=1)       # 基础特征
        self.tcn1=ResidualMSTCN(hidden,hidden,3,1,dropout)
        self.tcn2=ResidualMSTCN(hidden,hidden,3,2,dropout)
        self.tcn3=ResidualMSTCN(hidden,hidden,3,4,dropout)
        self.gamma=nn.Parameter(torch.tensor(0.1))        # 残差缩放, 初始小值
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.fuse=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden,out_dim))

    def forward(self,x):
        h_base=F.relu(self.base(x))                         # [B*N, hidden, W]
        # 三路残差, gamma 控制幅度
        h_tcn=self.tcn1(h_base)
        h_tcn=self.tcn2(h_tcn)
        h_tcn=self.tcn3(h_tcn)
        h=h_base+self.gamma*h_tcn                           # 残差连接, 小步长
        return self.fuse(self.pool(h).squeeze(-1))


# ═══════════════════════════════════════════════════
# ParallelSpatioTemporalEncoder (支持 temporal_mode)
# ═══════════════════════════════════════════════════
class ParallelSpatioTemporalEncoder(nn.Module):
    def __init__(self, nv=51, ws=60, prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3,
                 temporal_mode="per_variable_conv"):
        super().__init__()
        self.nv=nv; self.ws=ws; self.use_flatten=use_flatten
        self.temporal_mode=temporal_mode

        # 空间
        self.dyn_graph=DynamicPearsonPriorGraph(nv,prior_edge_index,prior_weights,boost)
        self.prior_embed=PriorNodeEmbedding(nv,prior_edge_index,prior_weights,hidden_dim)
        self.gat=GATv2Block(ws,hidden_dim,hidden_dim,heads=gat_heads,dropout=dropout)
        self.gate=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.Sigmoid())

        # 时间分支选择
        if temporal_mode=="per_variable_conv":
            self.temporal_enc=PerVariableConv(ws,hidden_dim,hidden_dim//2,dropout)
        elif temporal_mode=="per_variable_dilated_conv":
            self.temporal_enc=PerVariableDilatedConv(ws,hidden_dim,hidden_dim//2,dropout)
        elif temporal_mode=="per_variable_residual_ms_tcn":
            self.temporal_enc=PerVariableResidualMSTCN(ws,hidden_dim,hidden_dim//2,dropout)
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        self.fusion=NodeLevelFusion(hidden_dim,hidden_dim,hidden_dim,dropout)
        if use_flatten:
            fd=nv*hidden_dim; md=fd//16
            self.latent_proj=nn.Sequential(nn.Linear(fd,md),nn.ReLU(),nn.Linear(md,latent_dim))
        else:
            self.latent_proj=nn.Linear(hidden_dim,latent_dim)

    def forward(self,x):
        b,w,n=x.shape
        edges=self.dyn_graph(x)
        hg=self.gat(x.permute(0,2,1),edges)
        hp=self.prior_embed().unsqueeze(0).expand(b,-1,-1)
        g=self.gate(torch.cat([hg,hp],-1)); hs=g*hg+(1-g)*hp
        xv=x.permute(0,2,1).reshape(b*n,1,w)
        ht=self.temporal_enc(xv).reshape(b,n,-1)
        hf=self.fusion(hs,ht)
        z=hf.reshape(b,n*hf.shape[-1]) if self.use_flatten else hf.mean(1)
        return self.latent_proj(z),edges


class ABC_USAD(nn.Module):
    def __init__(self, nv, ws, static_edge_index, prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, gru_hidden=32, tcn_channels=32, tcn_blocks=1,
                 dropout=0.2, latent_dim=64, use_flatten=True,
                 temporal_mode="per_variable_conv"):
        super().__init__()
        df=nv*ws
        self.encoder=ParallelSpatioTemporalEncoder(
            nv,ws,prior_edge_index,prior_weights,hidden_dim,gat_heads,dropout,
            latent_dim,use_flatten,temporal_mode=temporal_mode)
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
