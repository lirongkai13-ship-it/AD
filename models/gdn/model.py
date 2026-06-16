"""GDN [Deng 2019 AAAI]: Graph Deviation Network"""
import torch, math
import torch.nn as nn


class GDN(nn.Module):
    """传感器嵌入 + LSTM + 注意力偏差评分"""
    def __init__(self, n_vars=51, window=60, hidden=96, top_k=30, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.window = window

        # 传感器嵌入
        self.embeddings = nn.Parameter(torch.randn(n_vars, hidden) * 0.1)

        # LSTM 编码
        self.lstm = nn.LSTM(n_vars, hidden, 2, batch_first=True, dropout=dropout)

        # 每个传感器的预测头（输出 1 个值 = 该传感器在下一时刻的预测）
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, n_vars),
        )

        # LSTM 解码器：逐步生成完整窗口（参数量大幅低于全连接）
        self.dec_lstm = nn.LSTM(n_vars, hidden, 1, batch_first=True)
        self.dec_out = nn.Linear(hidden, n_vars)
        self.dec_init = nn.Linear(hidden * 2, hidden)  # combined → 初始隐藏态
        self.dec_cell = nn.Linear(hidden * 2, hidden)  # combined → 初始细胞态

    def forward(self, x):
        b, w, n = x.shape

        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, W, H]
        h_last = lstm_out[:, -1, :]          # [B, H]

        # 传感器注意力
        attn = torch.softmax(
            torch.matmul(h_last, self.embeddings.T) / math.sqrt(h_last.shape[1]),
            dim=-1,
        )  # [B, N]
        ctx = torch.matmul(attn, self.embeddings)  # [B, H]

        # 融合向量 → 初始化 LSTM 解码器
        combined = torch.cat([h_last, ctx], dim=1)  # [B, 2H]
        h0 = self.dec_init(combined).unsqueeze(0)    # [1, B, H]
        c0 = self.dec_cell(combined).unsqueeze(0)    # [1, B, H]

        # 逐步解码
        dec_input = x[:, -1:, :]  # [B, 1, N] 窗口最后一步作为起始
        outputs = []
        for t in range(w):
            dec_out, (h0, c0) = self.dec_lstm(dec_input, (h0, c0))
            pred_t = self.dec_out(dec_out)        # [B, 1, N]
            outputs.append(pred_t)
            dec_input = pred_t                     # 自回归

        recon = torch.cat(outputs, dim=1)  # [B, W, N]
        return recon
