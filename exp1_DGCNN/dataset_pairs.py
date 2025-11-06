import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _ensure_tensor(x: Any, dtype=torch.float32) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    x = np.asarray(x)
    if x.dtype.kind in {'i', 'u'} and dtype == torch.long:
        return torch.from_numpy(x).long()
    return torch.from_numpy(x.astype(np.float32))


def _read_index(path: str) -> List[str]:
    p = Path(path)
    if p.is_dir():
        files = sorted([str(pp) for pp in p.rglob('*.npz')])
        return files
    if p.suffix.lower() == '.txt':
        out = []
        with open(p, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                out.append(line)
        return out
    if p.suffix.lower() == '.json':
        with open(p, 'r') as f:
            j = json.load(f)
        if isinstance(j, list):
            return j
        if isinstance(j, dict):
            if 'files' in j and isinstance(j['files'], list):
                return j['files']
            if 'paths' in j and isinstance(j['paths'], list):
                return j['paths']
        raise ValueError('JSON index must be a list or dict with key "files" or "paths"')
    # Otherwise assume it's a single npz
    if p.suffix.lower() == '.npz' and p.exists():
        return [str(p)]
    raise FileNotFoundError(f'Cannot interpret index path: {path}')


def _reshape(x: Optional[torch.Tensor], shape: tuple) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.reshape(shape)


class PairNPZDataset(Dataset):
    """
    Dataset of precomputed part pair samples stored as .npz files.

    Required per sample (keys in npz):
      - s_i (D), s_j (D)
      - c_i (3), c_j (3)
      - R_i (3,3), R_j (3,3)
      - loglam_i (3), loglam_j (3)

    Optional supervision keys:
      - type (int {0,1,2})
      - axis_world (3)
      - axis_point_world (3)
      - limits (2)
      - exist (0/1)
      - child_points_world (N,3)
      - child_points_weights (N)

    Optional contact keys:
      - contact_centroid (3)
      - d_min (scalar)
      - contact_area (scalar)
    """

    def __init__(self, index_path: str):
        super().__init__()
        self.files = _read_index(index_path)
        if len(self.files) == 0:
            raise RuntimeError(f'No .npz files found for index: {index_path}')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as data:
            item: Dict[str, torch.Tensor] = {}
            # required
            item['s_i'] = _ensure_tensor(data['s_i'])
            item['s_j'] = _ensure_tensor(data['s_j'])
            item['c_i'] = _ensure_tensor(data['c_i']).reshape(3)
            item['c_j'] = _ensure_tensor(data['c_j']).reshape(3)
            item['R_i'] = _ensure_tensor(data['R_i']).reshape(3, 3)
            item['R_j'] = _ensure_tensor(data['R_j']).reshape(3, 3)
            item['loglam_i'] = _ensure_tensor(data['loglam_i']).reshape(3)
            item['loglam_j'] = _ensure_tensor(data['loglam_j']).reshape(3)

            # optional contact
            item['contact_centroid'] = _ensure_tensor(data['contact_centroid']) if 'contact_centroid' in data else None
            if item['contact_centroid'] is not None:
                item['contact_centroid'] = item['contact_centroid'].reshape(3)
            item['d_min'] = _ensure_tensor(data['d_min']) if 'd_min' in data else None
            if item['d_min'] is not None:
                item['d_min'] = item['d_min'].reshape(1)
            item['contact_area'] = _ensure_tensor(data['contact_area']) if 'contact_area' in data else None
            if item['contact_area'] is not None:
                item['contact_area'] = item['contact_area'].reshape(1)

            # supervision
            if 'type' in data:
                item['type'] = _ensure_tensor(data['type'], dtype=torch.long).reshape(())
            if 'axis_world' in data:
                item['axis_world'] = _ensure_tensor(data['axis_world']).reshape(3)
            if 'axis_point_world' in data:
                item['axis_point_world'] = _ensure_tensor(data['axis_point_world']).reshape(3)
            if 'limits' in data:
                item['limits'] = _ensure_tensor(data['limits']).reshape(2)
            if 'exist' in data:
                # store as float for BCE
                exist = _ensure_tensor(data['exist']).reshape(())
                item['exist'] = exist.float()
            if 'child_points_world' in data:
                cp = _ensure_tensor(data['child_points_world'])  # (N,3)
                cp = cp.reshape(-1, 3)
                item['child_points_world'] = cp
                if 'child_points_weights' in data:
                    w = _ensure_tensor(data['child_points_weights']).reshape(-1)
                else:
                    w = torch.ones(cp.shape[0], dtype=torch.float32)
                item['child_points_weights'] = w

        # Fill missing optional contact with None; collate will handle
        item.setdefault('contact_centroid', None)
        item.setdefault('d_min', None)
        item.setdefault('contact_area', None)
        return item


def pair_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Determine max N for variable-length child points
    keys = set().union(*[b.keys() for b in batch])
    out: Dict[str, Any] = {}

    def stack_or_default(key: str, default: Optional[torch.Tensor] = None):
        vals = [b.get(key, None) for b in batch]
        if all(v is None for v in vals):
            return None
        # replace None with default/zeros of proper shape
        # infer shape from first tensor
        ref = next((v for v in vals if isinstance(v, torch.Tensor)), None)
        if ref is None:
            return None
        rep = default if default is not None else torch.zeros_like(ref)
        vals = [v if isinstance(v, torch.Tensor) else rep for v in vals]
        return torch.stack(vals, dim=0)

    # Simple fixed-size tensors
    for key in ['s_i', 's_j', 'c_i', 'c_j', 'loglam_i', 'loglam_j']:
        out[key] = stack_or_default(key)
    for key in ['R_i', 'R_j']:
        out[key] = stack_or_default(key)

    # Optionals: contact_* may be None
    out['contact_centroid'] = stack_or_default('contact_centroid')
    out['d_min'] = stack_or_default('d_min')
    out['contact_area'] = stack_or_default('contact_area')

    # Supervision
    if any('type' in b for b in batch):
        out['type'] = torch.stack([b['type'] for b in batch if 'type' in b], dim=0)
        # In case some items lack 'type', pad with -1 and later ignore? For now, assume consistent.
    if any('axis_world' in b for b in batch):
        out['axis_world'] = torch.stack([b['axis_world'] for b in batch if 'axis_world' in b], dim=0)
    if any('axis_point_world' in b for b in batch):
        out['axis_point_world'] = torch.stack([b['axis_point_world'] for b in batch if 'axis_point_world' in b], dim=0)
    if any('limits' in b for b in batch):
        out['limits'] = torch.stack([b['limits'] for b in batch if 'limits' in b], dim=0)
    if any('exist' in b for b in batch):
        out['exist'] = torch.stack([b['exist'] for b in batch if 'exist' in b], dim=0)

    # Variable length child points -> pad within batch
    has_cp = any('child_points_world' in b for b in batch)
    if has_cp:
        Ns = [b['child_points_world'].shape[0] if 'child_points_world' in b else 0 for b in batch]
        maxN = max(Ns)
        cp_list = []
        w_list = []
        for b, N in zip(batch, Ns):
            if 'child_points_world' in b:
                cp = b['child_points_world']
                w = b.get('child_points_weights', torch.ones(cp.shape[0], dtype=torch.float32))
            else:
                cp = torch.zeros(0, 3, dtype=torch.float32)
                w = torch.zeros(0, dtype=torch.float32)
            # pad
            pad_cp = torch.zeros(maxN, 3, dtype=torch.float32)
            pad_w = torch.zeros(maxN, dtype=torch.float32)
            if N > 0:
                pad_cp[:N] = cp
                pad_w[:N] = w
            cp_list.append(pad_cp)
            w_list.append(pad_w)
        out['child_points_world'] = torch.stack(cp_list, dim=0)
        out['child_points_weights'] = torch.stack(w_list, dim=0)

    return out


__all__ = ['PairNPZDataset', 'pair_collate']
