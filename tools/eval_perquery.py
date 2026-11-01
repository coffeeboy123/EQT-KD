# Re-evaluate a saved checkpoint and dump per-query AP / CMC for bootstrap analysis.
# usage: python tools/eval_perquery.py <qat|fp32> <ckpt> <out.npz> -- <option.py args...>
import sys, numpy as np, torch
mode, ckpt_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
DEV = "cuda" if (mode == "fp32" and torch.cuda.is_available()) else "cpu"
sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, ".")
from option import args
import data_v2, utils.utility as utility
from utils.functions import evaluation
args.test_only = True
ckpt = utility.checkpoint(args)
loader = data_v2.ImageDataManager(args)
if mode == "qat":
    from model import make_model_student
    model = make_model_student(args, ckpt)
    model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    model = torch.quantization.prepare_qat(model, inplace=True)
else:
    from model import make_model
    model = make_model(args, ckpt)
model.to(DEV).eval()
if mode == "qat":
    # best checkpoints hold the already-converted INT8 state dict
    if hasattr(model, "disable_klw_for_inference"): model.disable_klw_for_inference()
    torch.backends.quantized.engine = "qnnpack"
    model = torch.quantization.convert(model, inplace=False)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    print("load:", model.load_state_dict(sd, strict=False))  # keep _metadata for packed params
else:
    ckpt.load_pretrained_weights(model, ckpt_path)

def feat(x):
    # most models return a tuple whose third element is the descriptor; PCB returns the stacked descriptor itself
    o = model(x)
    return (o if torch.is_tensor(o) else o[2]).data.cpu()

def extract(dl):
    feats, pids, cams = [], [], []
    for d in dl:
        x, pid, cam = d[0].to(DEV), d[1], d[2]
        f = feat(x) + feat(x.flip(3))
        if f.dim() == 3:
            f = f / torch.norm(f, p=2, dim=1, keepdim=True); f = f.view(f.size(0), -1)
        else:
            f = f / torch.norm(f, p=2, dim=1, keepdim=True)
        feats.append(f); pids += list(np.asarray(pid)); cams += list(np.asarray(cam))
    return torch.cat(feats), np.asarray(pids), np.asarray(cams)

with torch.no_grad():
    qf, qp, qc = extract(loader.query_loader)
    gf, gp, gc = extract(loader.test_loader)
dist = 1 - torch.mm(qf, gf.t()).numpy()
cmc, mAP = evaluation(dist, qp, gp, qc, gc, 50)
ap, hit = [], []
for i in range(len(qp)):
    c, a = evaluation(dist[i:i+1], qp[i:i+1], gp, qc[i:i+1], gc, 50, use_cython=False)
    ap.append(a); hit.append(c[:10])
ap, hit = np.array(ap), np.array(hit)
print("overall mAP %.4f R1 %.4f R5 %.4f R10 %.4f | per-query mean mAP %.4f R1 %.4f n=%d" % (mAP, cmc[0], cmc[4], cmc[9], ap.mean(), hit[:,0].mean(), len(ap)))
np.savez(out, ap=ap, cmc=hit, qpids=qp)
