# loss/kd_logic_loss_qaware.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLogicLossQAware(nn.Module):
    def __init__(self, temperature: float = 2.0,
                 alpha: float = 1.0,
                 eps: float = 1e-6,
                 w_min: float = 1.0,
                 w_max: float = 5.0,
                 norm: str = "l2",
                 per_sample: bool = False,
                 detach_error: bool = True):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.norm = norm
        self.per_sample = per_sample
        self.detach_error = detach_error

    def _compute_error_weight(self, s_post, s_pre):
        if self.norm == "l1":
            err = (s_post - s_pre).abs().sum(dim=-1)   # (B,H)
        else:
            err = torch.norm(s_post - s_pre, p=2, dim=-1)  # (B,H)

        if self.detach_error:
            err = err.detach()

        denom = err.mean().clamp_min(self.eps)
        norm_err = err / denom                           # (B,H)

        w = 1.0 + self.alpha * norm_err                  # (B,H)
        if not self.per_sample:
            w = w.mean(dim=0, keepdim=True)              # (1,H)
        w = w.clamp_(self.w_min, self.w_max)             # (B,H) or (1,H)
        return w

    # ★ w_learned 추가 (optional)
    def forward(self, s_post: torch.Tensor, t_logits: torch.Tensor, s_pre: torch.Tensor, w_learned: torch.Tensor = None):
        T = self.T

        def _u3(x):
            return x.unsqueeze(1) if x.dim() == 2 else x

        s_post = _u3(s_post)
        t_logits = _u3(t_logits)
        s_pre = _u3(s_pre)

        B, H, C = s_post.shape
        assert t_logits.shape == (B, H, C) and s_pre.shape == (B, H, C), "Shape mismatch"

        # 1) 학습 가중치가 있으면 그걸 쓰고 (+1), 없으면 규칙 가중치를 계산하고 (+1)
        if w_learned is not None:
            # w_learned: (B,H) 기대
            w = w_learned + 1.0
        else:
            w = self._compute_error_weight(s_post, s_pre)
            w = w + 1.0

        with torch.no_grad():
            pt = torch.softmax(t_logits / T, dim=-1)

        log_ps = torch.log_softmax(s_post / T, dim=-1)

        kl_all = F.kl_div(log_ps, pt, reduction="none").sum(dim=-1)  # (B,H)

        # 가중 적용
        kl_all = kl_all * w

        kl_per_head = kl_all.mean(dim=0)   # (H,)
        loss = kl_per_head.sum() * (T * T)
        return loss
