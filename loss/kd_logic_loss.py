# loss/kd_logic_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLogicLoss(nn.Module):
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = float(temperature)

    def forward(self, s_logits: torch.Tensor, t_logits: torch.Tensor):
        """
        s_logits, t_logits: (B, C) 또는 (B, H, C)
        """
        T = self.T

        # (B,C) → (B,1,C) 통일
        if s_logits.dim() == 2:
            s_logits = s_logits.unsqueeze(1)
            t_logits = t_logits.unsqueeze(1)

        B, H, C = s_logits.shape
        assert t_logits.shape == (B, H, C), "Shape mismatch"

        # teacher 확률 분포
        with torch.no_grad():
            pt = torch.softmax(t_logits / T, dim=-1)   # (B,H,C)

        log_ps = torch.log_softmax(s_logits / T, dim=-1)  # (B,H,C)

        # per-sample, per-head KL
        kl_all = F.kl_div(log_ps, pt, reduction="none")   # (B,H,C)
        kl_all = kl_all.sum(dim=-1)                       # (B,H) 클래스 축 합
        kl_per_head = kl_all.mean(dim=0)                  # (H,) 배치 평균

        # 모든 헤드 합산
        loss = kl_per_head.sum() * (T * T)
        return loss
