import torch
import torch.nn as nn
import torch.nn.functional as F

def rot6d_from_R(R):
    # R: (..., 3, 3)
    return torch.stack([R[..., 0, 0], R[..., 1, 0], R[..., 2, 0],
                        R[..., 0, 1], R[..., 1, 1], R[..., 2, 1]], dim=-1)

class PairwiseJointHead(nn.Module):
    def __init__(self, sem_dim=128, hidden=256):
        super().__init__()
        in_dim = 0
        # semantics
        in_dim += sem_dim * 2 + 1  # s_i, s_j, cos_sim
        # geometry
        in_dim += 3  # Δc (parent frame)
        in_dim += 3  # log λ_i
        in_dim += 3  # log λ_j
        in_dim += 6  # R_rel 6D
        in_dim += 3  # optional: contact centroid in parent frame (or zeros)
        in_dim += 2  # optional: d_min, contact area proxy (or zeros)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.type_head = nn.Linear(hidden, 3)     # revolute, prismatic, fixed
        self.exist_head = nn.Linear(hidden, 1)    # for K>2 edge existence
        self.alpha_head = nn.Linear(hidden, 3)    # axis coeffs in parent frame
        self.offset_head = nn.Linear(hidden, 3)   # anchor offset in parent frame
        self.limits_head = nn.Linear(hidden, 2)   # [min, max] normalized

    def forward(self, s_i, s_j, c_i, c_j, R_i, R_j, loglam_i, loglam_j,
                contact_centroid=None, d_min=None, contact_area=None):
        # s_i, s_j: (B, D)
        # c_i, c_j: (B, 3)
        # R_i, R_j: (B, 3, 3) orthonormal
        # loglam_i, loglam_j: (B, 3)
        B = s_i.shape[0]
        if contact_centroid is None:
            contact_centroid = torch.zeros(B, 3, device=s_i.device)
        if d_min is None:
            d_min = torch.zeros(B, 1, device=s_i.device)
        if contact_area is None:
            contact_area = torch.zeros(B, 1, device=s_i.device)

        # express in parent (i) frame
        d_c = c_j - c_i                           # (B, 3)
        d_c_parent = torch.einsum('bij,bj->bi', R_i.transpose(1, 2), d_c)
        R_rel = torch.einsum('bij,bjk->bik', R_i.transpose(1, 2), R_j)
        R_rel6 = rot6d_from_R(R_rel)
        cc_parent = torch.einsum('bij,bj->bi', R_i.transpose(1, 2), contact_centroid - c_i)

        cos_sim = F.cosine_similarity(s_i, s_j, dim=-1, eps=1e-6).unsqueeze(-1)

        x = torch.cat([
            s_i, s_j, cos_sim,
            d_c_parent, loglam_i, loglam_j,
            R_rel6, cc_parent, d_min, contact_area
        ], dim=-1)

        h = self.mlp(x)
        type_logits = self.type_head(h)
        exist_logit = self.exist_head(h)
        alpha = self.alpha_head(h)                # axis coeffs (parent frame)
        offset = self.offset_head(h)              # anchor offset (parent frame)
        limits = self.limits_head(h)              # normalized min/max

        # Decode to world frame (for loss/export)
        # Axis u = R_i @ normalize(alpha)
        alpha_n = F.normalize(alpha, dim=-1, eps=1e-8)
        u_world = torch.einsum('bij,bj->bi', R_i, alpha_n)
        # Anchor p = c_i + R_i @ offset
        p_world = c_i + torch.einsum('bij,bj->bi', R_i, offset)

        return {
            'type_logits': type_logits,
            'exist_logit': exist_logit,
            'axis_world': u_world,
            'anchor_world': p_world,
            'limits': limits,
            'intermediates': {
                'alpha_parent': alpha_n,
                'offset_parent': offset,
                'R_rel6': R_rel6,
                'd_c_parent': d_c_parent
            }
        }


# ----------------------------- Training utilities -----------------------------
from typing import Dict, Optional


def _dot_abs(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, dim=dim, eps=eps)
    b_n = F.normalize(b, dim=dim, eps=eps)
    return (a_n * b_n).sum(dim=dim).abs()


def axis_angle_loss(u_pred: torch.Tensor, u_gt: torch.Tensor) -> torch.Tensor:
    # 1 - |u · u*|
    return (1.0 - _dot_abs(u_pred, u_gt)).mean()


def point_to_line_distance(points: torch.Tensor, p: torch.Tensor, u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # points: (B,N,3) or (B,3)
    # p: (B,3), u: (B,3), unit not required
    if points.dim() == 2:
        points = points.unsqueeze(1)  # (B,1,3)
    u_n = F.normalize(u, dim=-1, eps=eps).unsqueeze(1)  # (B,1,3)
    p_to_x = points - p.unsqueeze(1)  # (B,N,3)
    cross = torch.linalg.cross(p_to_x, u_n, dim=-1)
    dist = torch.linalg.norm(cross, dim=-1)  # (B,N)
    return dist


def anchor_loss(p_pred: torch.Tensor,
                u_pred: torch.Tensor,
                axis_point_gt: Optional[torch.Tensor] = None,
                axis_dir_gt: Optional[torch.Tensor] = None,
                child_points_gt: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Prefer GT axis line if available, else use child points wrt predicted line
    if axis_point_gt is not None and axis_dir_gt is not None:
        # Distance of predicted anchor to GT axis line
        d = point_to_line_distance(p_pred, axis_point_gt, axis_dir_gt)  # (B,1)
        return d.mean()
    if child_points_gt is not None:
        # Distance of sampled child points to predicted axis line
        d = point_to_line_distance(child_points_gt, p_pred, u_pred)  # (B,N)
        if weights is not None:
            w = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            return (d * w).sum(dim=-1).mean()
        return d.mean()
    return torch.tensor(0.0, device=p_pred.device)


def limits_loss(lim_pred: torch.Tensor, lim_gt: torch.Tensor, margin: float = 1e-3) -> torch.Tensor:
    # lim_*: (B,2) normalized [min, max]
    l1 = F.smooth_l1_loss(lim_pred, lim_gt)
    # Hinge to enforce min < max
    hinge = F.relu(lim_pred[:, 0] - lim_pred[:, 1] + margin).mean()
    return l1 + hinge


def type_loss(type_logits: torch.Tensor, type_gt: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(type_logits, type_gt)


def exist_loss(exist_logit: torch.Tensor, exist_gt: torch.Tensor) -> torch.Tensor:
    exist_gt = exist_gt.float().view_as(exist_logit)
    return F.binary_cross_entropy_with_logits(exist_logit, exist_gt)


def compute_total_loss(outputs: Dict[str, torch.Tensor],
                       gt: Dict[str, torch.Tensor],
                       weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = {}
    w_type = weights.get('type', 1.0)
    w_axis = weights.get('axis', 1.0)
    w_anchor = weights.get('anchor', 1.0)
    w_limits = weights.get('limits', 0.5)
    w_exist = weights.get('exist', 0.1)

    losses = {}
    if 'type' in gt and outputs.get('type_logits') is not None:
        losses['type'] = type_loss(outputs['type_logits'], gt['type']) * w_type
    if 'axis_world' in outputs and 'axis_world' in gt:
        losses['axis'] = axis_angle_loss(outputs['axis_world'], gt['axis_world']) * w_axis
    # Anchor uses either GT axis line or child points
    losses['anchor'] = anchor_loss(
        outputs.get('anchor_world'),
        outputs.get('axis_world'),
        axis_point_gt=gt.get('axis_point_world'),
        axis_dir_gt=gt.get('axis_world'),
        child_points_gt=gt.get('child_points_world'),
        weights=gt.get('child_points_weights')
    ) * w_anchor
    if 'limits' in gt and outputs.get('limits') is not None:
        losses['limits'] = limits_loss(outputs['limits'], gt['limits']) * w_limits
    if 'exist' in gt and outputs.get('exist_logit') is not None:
        losses['exist'] = exist_loss(outputs['exist_logit'], gt['exist']) * w_exist

    losses['total'] = sum(losses.values()) if len(losses) > 0 else torch.tensor(0.0, device=next(iter(outputs.values())).device)
    return losses


def forward_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        batch['s_i'], batch['s_j'],
        batch['c_i'], batch['c_j'],
        batch['R_i'], batch['R_j'],
        batch['loglam_i'], batch['loglam_j'],
        batch.get('contact_centroid'),
        batch.get('d_min'),
        batch.get('contact_area'),
    )


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class JointTrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0,
                 loss_weights: Optional[Dict[str, float]] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_weights = loss_weights or {}

    def step(self, batch: Dict[str, torch.Tensor], train: bool = True) -> Dict[str, torch.Tensor]:
        batch = to_device(batch, self.device)
        outputs = forward_from_batch(self.model, batch)
        # Build GT dict with tolerant keys
        gt = {
            k: batch[k] for k in ['type', 'axis_world', 'limits', 'exist', 'axis_point_world'] if k in batch
        }
        # Optional child points for anchor supervision
        if 'child_points_world' in batch:
            gt['child_points_world'] = batch['child_points_world']
        if 'child_points_weights' in batch:
            gt['child_points_weights'] = batch['child_points_weights']

        losses = compute_total_loss(outputs, gt, self.loss_weights)

        if train:
            self.opt.zero_grad(set_to_none=True)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.opt.step()
        return {'losses': losses, 'outputs': outputs}

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        agg = {}
        for batch in loader:
            res = self.step(batch, train=True)
            for k, v in res['losses'].items():
                agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
        n = max(1, len(loader))
        return {k: v / n for k, v in agg.items()}

    def eval_epoch(self, loader) -> Dict[str, float]:
        self.model.eval()
        agg = {}
        with torch.no_grad():
            for batch in loader:
                res = self.step(batch, train=False)
                for k, v in res['losses'].items():
                    agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
        n = max(1, len(loader))
        return {k: v / n for k, v in agg.items()}


# ----------------------------- CLI: simple training script -----------------------------
if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    try:
        from .dataset_pairs import PairNPZDataset, pair_collate  # if used as a package
    except Exception:
        try:
            from dataset_pairs import PairNPZDataset, pair_collate  # script mode
        except Exception:
            PairNPZDataset = None
            pair_collate = None

    parser = argparse.ArgumentParser(description='Train PairwiseJointHead (K=2 or edge batches)')
    parser.add_argument('--train_index', type=str, default=None, help='Path to train index (txt/json) or directory of NPZs')
    parser.add_argument('--val_index', type=str, default=None, help='Path to val index (optional)')
    parser.add_argument('--sem-dim', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save', type=str, default='./pjh_checkpoint.pt')
    args = parser.parse_args()

    model = PairwiseJointHead(sem_dim=args.sem_dim, hidden=args.hidden)
    trainer = JointTrainer(model, lr=args.lr, weight_decay=args.weight_decay,
                           loss_weights={'type': 1.0, 'axis': 1.0, 'anchor': 1.0, 'limits': 0.5, 'exist': 0.1})

    if PairNPZDataset is None or args.train_index is None:
        print('Dataset not provided or PairNPZDataset unavailable. Exiting after model init.')
        exit(0)

    train_ds = PairNPZDataset(args.train_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False,
                              collate_fn=pair_collate)
    if args.val_index:
        val_ds = PairNPZDataset(args.val_index)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False,
                                collate_fn=pair_collate)
    else:
        val_loader = None

    for epoch in range(1, args.epochs + 1):
        tr = trainer.train_epoch(train_loader)
        msg = f'Epoch {epoch:03d} | ' + ' '.join([f'{k}:{v:.4f}' for k, v in tr.items()])
        if val_loader is not None:
            va = trainer.eval_epoch(val_loader)
            msg += ' | ' + ' '.join([f'val_{k}:{v:.4f}' for k, v in va.items()])
        print(msg)

    torch.save({'model': model.state_dict(), 'config': vars(args)}, args.save)
    print(f'Saved checkpoint to {args.save}')