"""批量生成所有模型架构图"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "model_diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

# 统一输入形状
B, W, N = 1, 60, 51
x_sample = torch.randn(B, W, N)
ei_dummy = torch.tensor([[i, i] for i in range(10)], dtype=torch.long).T  # 示例边

# ─── 主模型注册 ───
def baseline_model():
    from model import GATv2TCNGRUDetector
    return GATv2TCNGRUDetector(N, W, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Baseline_GATv2_TCN_GRU"

def temporal_attn_model():
    from models_variants.temporal_attn.variant_model import GATv2_TA_TCN_GRU
    return GATv2_TA_TCN_GRU(N, W, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Temporal_Attention"

def prior_fusion_model():
    from models_variants.prior_fusion.variant_model import GATv2_PriorFusion_TCN_GRU
    n_prior_ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    n_prior_w = torch.tensor([1.0, 1.0])
    return GATv2_PriorFusion_TCN_GRU(N, W, n_prior_ei, n_prior_w, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Prior_Fusion"

def prior_dynamic_model():
    from models_variants.prior_dynamic.variant_model import GATv2_PD_TCN_GRU
    n_prior_ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    n_prior_w = torch.tensor([1.0, 1.0])
    return GATv2_PD_TCN_GRU(N, W, n_prior_ei, n_prior_w, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Prior_Dynamic"

def ms_tcn_model():
    from models_variants.ms_tcn.variant_model import GATv2TCNGRUDetector_MS
    return GATv2TCNGRUDetector_MS(N, W, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "MultiScale_TCN"

def dynamic_graph_model():
    from models_variants.dynamic_graph.variant_model import GATv2_DG_TCN_GRU
    return GATv2_DG_TCN_GRU(N, W, ei_dummy, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Dynamic_Pearson"

def dynamic_prior_feat_model():
    from models_variants.dynamic_prior_feat.variant_model import GATv2_DynPrior_TCN_GRU
    n_prior_ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    n_prior_w = torch.tensor([1.0, 1.0])
    return GATv2_DynPrior_TCN_GRU(N, W, n_prior_ei, n_prior_w, ei_dummy, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Dynamic_Prior_Feat"

def usad_dual_model():
    from models_variants.usad_dual.variant_model import USAD_DualDecoder
    return USAD_DualDecoder(N, W, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "USAD_Dual"

def dynamic_usad_model():
    from models_variants.dynamic_usad.variant_model import DynPearson_USAD
    return DynPearson_USAD(N, W, ei_dummy, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Dynamic_USAD"

def dynamic_usad_prior_model():
    from models_variants.dynamic_usad_prior.variant_model import DynPrior_USAD
    n_prior_ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    n_prior_w = torch.tensor([1.0, 1.0])
    return DynPrior_USAD(N, W, ei_dummy, n_prior_ei, n_prior_w, 32, 2, 32, 32, 1, 0.2), (x_sample, ei_dummy), "Dynamic_USAD_Prior"

# ─── 外部对比模型 ───
def lstm_ae_model():
    from models.lstm_ae.model import LSTMAE
    m = LSTMAE(N, W, 64, 2, 0.1)
    return m, (x_sample,), "LSTM-AE"

def dagmm_model():
    from models.dagmm.model import DAGMM
    m = DAGMM(N, W, 64, 16, 4, 0.1)
    return m, (x_sample,), "DAGMM"

def usad_solo_model():
    from models.usad.model import USAD
    m = USAD(N, W, 64, 32, 0.1)
    return m, (x_sample,), "USAD"

def mtad_gat_model():
    from models.mtad_gat.model import MTADGAT
    m = MTADGAT(N, W, 48, 2, 32, 0.1)
    return m, (x_sample, ei_dummy), "MTAD-GAT"

def mad_gan_model():
    from models.mad_gan.model import MADGAN
    m = MADGAN(N, W, 32, 64)
    return m, (x_sample,), "MAD-GAN"

def ano_trans_model():
    from models.ano_trans.model import AnomalyTransformer
    m = AnomalyTransformer(N, W, 64, 4, 2, 0.1)
    return m, (x_sample,), "AnomalyTransformer"

def tranad_model():
    from models.tranad.model import TranAD
    m = TranAD(N, W, 48, 4, 2, 0.1)
    return m, (x_sample,), "TranAD"

def timesnet_model():
    from models.timesnet.model import TimesNet
    m = TimesNet(N, W, 256, 4, 0.1)
    return m, (x_sample,), "TimesNet"

def dcdetector_model():
    from models.dcdetector.model import DCdetector
    m = DCdetector(N, W, 96, 4, 0.1)
    return m, (x_sample,), "DCdetector"

def gdn_model():
    from models.gdn.model import GDN
    m = GDN(N, W, 96, 30, 0.1)
    return m, (x_sample,), "GDN"

def can_model():
    from models.can.model import CAN
    m = CAN(N, W, 96, 4, 0.1)
    return m, (x_sample,), "CAN"

def gcn_model():
    from models.gcn.model import GCN_TCN_GRU
    m = GCN_TCN_GRU(N, W, 32, 32, 32, 1, 0.2)
    return m, (x_sample, ei_dummy), "GCN-TCN-GRU"


ALL_MODELS = [
    # 我们的模型
    ("Ours_Baseline", baseline_model),
    ("Ours_TemporalAttn", temporal_attn_model),
    ("Ours_PriorFusion", prior_fusion_model),
    ("Ours_PriorDynamic", prior_dynamic_model),
    ("Ours_MS-TCN", ms_tcn_model),
    ("Ours_DynamicGraph", dynamic_graph_model),
    ("Ours_DynPriorFeat", dynamic_prior_feat_model),
    ("Ours_USAD", usad_dual_model),
    ("Ours_DynUSAD", dynamic_usad_model),
    ("Ours_DynUSAD_Prior", dynamic_usad_prior_model),
    # 对比模型
    ("Ext_LSTM-AE", lstm_ae_model),
    ("Ext_DAGMM", dagmm_model),
    ("Ext_USAD", usad_solo_model),
    ("Ext_MTAD-GAT", mtad_gat_model),
    ("Ext_MAD-GAN", mad_gan_model),
    ("Ext_AnoTrans", ano_trans_model),
    ("Ext_TranAD", tranad_model),
    ("Ext_TimesNet", timesnet_model),
    ("Ext_DCdetector", dcdetector_model),
    ("Ext_GDN", gdn_model),
    ("Ext_CAN", can_model),
    ("Ext_GCN", gcn_model),
]


def main():
    from torchview import draw_graph

    results = []
    for label, factory in ALL_MODELS:
        print(f"\n{'='*50}\n{label}\n{'='*50}")
        try:
            model, args, safe_name = factory()
            # 生成图
            g = draw_graph(
                model, input_data=args,
                depth=3,
                graph_dir='TB',       # top-to-bottom
                roll=True,            # 展开循环
                expand_nested=True,
                save_graph=True,
                filename=safe_name,
                directory=OUT_DIR,
            )
            print(f"  ✅ {safe_name}.gv.pdf 已生成")
            results.append({"label": label, "status": "OK"})
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"label": label, "status": "FAILED", "error": str(e)})

    print(f"\n{'='*50}")
    print(f"完成: {sum(1 for r in results if r['status']=='OK')}/{len(results)}")
    for r in results:
        print(f"  {r['label']}: {r['status']}")
    with open(os.path.join(OUT_DIR, "_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
