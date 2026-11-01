# FP32 / fake-quant / INT8 evaluation of the same final QAT weights (Section 4.2).
# usage: python tools/eval_fp32_fq_int8.py <qat_final_state.pth> <out prefix> -- <option.py args>
import sys, copy, numpy as np, torch
state_path, out = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, ".")
from option import args
import data_v2, utils.utility as utility
from utils.functions import evaluation
from model import make_model_student
args.test_only = True
ckpt = utility.checkpoint(args)
loader = data_v2.ImageDataManager(args)
torch.backends.quantized.engine = "qnnpack"
model = make_model_student(args, ckpt).to("cpu")
model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
torch.quantization.prepare_qat(model, inplace=True)
print(model.load_state_dict(torch.load(state_path, map_location="cpu")), flush=True)
model.eval()
model.apply(torch.quantization.disable_observer)  # keep the quantization ranges learned in training; do not calibrate on test images
if hasattr(model, "disable_klw_for_inference"):
    model.disable_klw_for_inference()


def evaluate(m, tag, dev):
    m = m.to(dev).eval()
    def extract(dl):
        feats, pids, cams = [], [], []
        for d in dl:
            x, pid, cam = d[0].to(dev), d[1], d[2]
            f = m(x)[2].data.cpu() + m(x.flip(3))[2].data.cpu()
            f = f / torch.norm(f, p=2, dim=1, keepdim=True)
            feats.append(f.view(f.size(0), -1)); pids += list(np.asarray(pid)); cams += list(np.asarray(cam))
        return torch.cat(feats), np.asarray(pids), np.asarray(cams)
    with torch.no_grad():
        qf, qp, qc = extract(loader.query_loader)
        gf, gp, gc = extract(loader.test_loader)
    dist = 1 - torch.mm(qf, gf.t()).numpy()
    cmc, mAP = evaluation(dist, qp, gp, qc, gc, 50)
    ap = np.array([evaluation(dist[i:i+1], qp[i:i+1], gp, qc[i:i+1], gc, 50, use_cython=False)[1] for i in range(len(qp))])
    print(f"{tag}: mAP {mAP:.4f} R1 {cmc[0]:.4f} R5 {cmc[4]:.4f} R10 {cmc[9]:.4f}", flush=True)
    np.savez(f"{out}_{tag}.npz", ap=ap, qpids=qp, qf=qf.numpy(), gf=gf.numpy())


evaluate(model, "fakequant", "cuda")
fp = copy.deepcopy(model)
fp.apply(torch.quantization.disable_fake_quant)
evaluate(fp, "fp32", "cuda")
int8 = torch.quantization.convert(copy.deepcopy(model).to("cpu"), inplace=False)
evaluate(int8, "int8", "cpu")
