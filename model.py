import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualGATv2Layer(nn.Module):
    """
    不依赖 torch_geometric 的简化 GATv2 层。
    输入 x: [B, N, Fin]
    edge_index: [2, E]
    输出: [B, N, Fout * heads] 或 [B, N, Fout]
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.lin_l = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_r = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))

        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        b, n, _ = x.shape
        src, dst = edge_index[0], edge_index[1]

        h_l = self.lin_l(x).view(b, n, self.heads, self.out_dim)
        h_r = self.lin_r(x).view(b, n, self.heads, self.out_dim)

        h_src = h_l[:, src, :, :]  # [B, E, H, D]
        h_dst = h_r[:, dst, :, :]  # [B, E, H, D]

        # GATv2: attention over W_l h_i + W_r h_j
        e = F.leaky_relu(h_src + h_dst, negative_slope=0.2)
        e = (e * self.att.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)  # [B, E, H]

        # 对每个目标节点做 softmax
        alpha = torch.zeros_like(e)
        for node in range(n):
            mask = dst == node
            if mask.any():
                alpha[:, mask, :] = torch.softmax(e[:, mask, :], dim=1)

        alpha = self.dropout(alpha)

        out = torch.zeros(b, n, self.heads, self.out_dim, device=x.device)
        msg = h_src * alpha.unsqueeze(-1)

        # 聚合到目标节点
        for edge_pos in range(src.numel()):
            out[:, dst[edge_pos], :, :] += msg[:, edge_pos, :, :]

        if self.concat:
            out = out.reshape(b, n, self.heads * self.out_dim)
        else:
            out = out.mean(dim=2)

        out = out + self.bias
        return out


class GATv2Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = ManualGATv2Layer(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.gat2 = ManualGATv2Layer(
            hidden_dim * heads,
            out_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        residual = self.res_proj(x)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index)
        h = self.norm(h + residual)
        return h


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.res = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        # x: [B, C, T]
        residual = self.res(x)
        out = self.conv(x)

        # 裁掉右侧多余 padding，保持因果卷积长度不变
        out = out[:, :, : x.size(-1)]

        out = F.relu(out)
        out = self.dropout(out)
        return out + residual


class GATv2TCNGRUDetector(nn.Module):
    """
    输入 x: [B, W, N]
    输出:
      pred: [B, N]        预测未来 horizon 点
      recon: [B, W, N]    重构当前窗口
    """

    def __init__(
        self,
        num_variables,
        window_size,
        hidden_dim=64,
        gat_heads=4,
        gru_hidden=64,
        tcn_channels=64,
        dropout=0.2,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.window_size = window_size

        # 每个变量的窗口序列作为节点特征：[N, W]
        self.gat = GATv2Block(
            in_dim=window_size,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            heads=gat_heads,
            dropout=dropout,
        )

        # GAT 输出 [B, N, hidden]，转为 [B, hidden, N] 做变量维 TCN
        self.tcn1 = TCNBlock(
            hidden_dim,
            tcn_channels,
            kernel_size=3,
            dilation=1,
            dropout=dropout,
        )
        self.tcn2 = TCNBlock(
            tcn_channels,
            tcn_channels,
            kernel_size=3,
            dilation=2,
            dropout=dropout,
        )

        # 把变量维序列送入 GRU
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_variables),
        )

        self.recon_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, window_size * num_variables),
        )

    def forward(self, x, edge_index):
        # x: [B, W, N]
        b, w, n = x.shape

        # 每个变量作为一个节点，每个节点特征是长度为 W 的时间窗口
        node_feat = x.transpose(1, 2)  # [B, N, W]

        h = self.gat(node_feat, edge_index)  # [B, N, hidden]

        # TCN 沿变量维处理空间融合后的表示
        z = h.transpose(1, 2)  # [B, hidden, N]
        z = self.tcn1(z)
        z = self.tcn2(z)

        # GRU 输入 [B, N, C]，把变量序列视作融合序列
        z = z.transpose(1, 2)
        _, h_last = self.gru(z)
        h_last = h_last[-1]

        pred = self.pred_head(h_last)
        recon = self.recon_head(h_last).view(b, w, n)

        return pred, recon