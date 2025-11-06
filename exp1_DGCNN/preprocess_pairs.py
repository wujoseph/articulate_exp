import os
import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Optional dependency for PLY reading
try:
    from plyfile import PlyData  # pip install plyfile
except Exception:  # pragma: no cover
    PlyData = None

# Try importing dataset utils (PartNet Mobility helpers)
try:
    # When running within the repo, add ../lib to sys.path
    import sys as _sys
    _repo_root = Path(__file__).resolve().parents[1]
    _lib_path = _repo_root / 'lib'
    if str(_lib_path) not in _sys.path:
        _sys.path.append(str(_lib_path))
    from dataset_utils import get_class_ids  # type: ignore
except Exception as _e:
    get_class_ids = None  # will error if user requests category filtering
    print(f"[WARN] dataset_utils.get_class_ids not available: {_e}")


@dataclass
class PreprocessConfig:
    sem_dim: int  # must match your PartField feature dimension (or target)
    # Dataset coordinates are already normalized; keep unchanged to match mobility_v2.json axes/origins
    normalize: bool = False
    neighbor_k: int = 4  # for K>2 graphs
    create_both_directions_when_k2: bool = True
    eps: float = 1e-6


# ---------------------------- IO helpers ----------------------------

def read_ply_xyz(path: str) -> np.ndarray:
    """Read Nx3 XYZ from a .ply file. Requires plyfile.
    """
    if PlyData is None:
        raise ImportError('plyfile is required. Install with: pip install plyfile')
    with open(path, 'rb') as f:
        ply = PlyData.read(f)
    # Try common property names
    for name in ['vertex', 'point', 'vertices']:
        if name in ply.elements:
            el = ply[name]
            break
    else:
        el = ply.elements[0]
    x = np.asarray(el.data['x'], dtype=np.float32)
    y = np.asarray(el.data['y'], dtype=np.float32)
    z = np.asarray(el.data['z'], dtype=np.float32)
    pts = np.stack([x, y, z], axis=1)
    return pts


def read_labels_txt(path: str) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.int32)
    return arr.reshape(-1)


def read_features_npy(path: str) -> np.ndarray:
    feats = np.load(path)
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)
    return feats.astype(np.float32)


# ---------------------------- Geometry utilities ----------------------------

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center by object centroid and scale by bbox diagonal L.
    Returns: points_norm, centroid, L
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    mins = centered.min(axis=0)
    maxs = centered.max(axis=0)
    diag = np.linalg.norm(maxs - mins) + 1e-12
    return centered / diag, centroid.astype(np.float32), float(diag)


def pca_frame(points: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Return R (3x3, det=+1) and axis scales (3,) as bbox lengths along PCA axes.
    Scales = (max - min) along each principal axis (robust and interpretable).
    """
    P = points
    Pc = P - P.mean(axis=0, keepdims=True)
    cov = np.cov(Pc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending by eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # ensure right-handed
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1.0
    # project to axes to get bbox lengths
    proj = P @ eigvecs  # (N,3)
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    scales = (maxs - mins).astype(np.float32)
    scales = np.maximum(scales, eps)
    return eigvecs.astype(np.float32), scales


def subsample(points: np.ndarray, max_n: int) -> np.ndarray:
    n = points.shape[0]
    if n <= max_n:
        return points
    idx = np.random.choice(n, size=max_n, replace=False)
    return points[idx]


# ---------------------------- Part tokens ----------------------------

def build_part_tokens(points_norm: np.ndarray, labels: np.ndarray, feats: np.ndarray,
                      eps: float) -> Dict[int, Dict[str, np.ndarray]]:
    part_ids = np.unique(labels)
    tokens: Dict[int, Dict[str, np.ndarray]] = {}
    for pid in part_ids.tolist():
        idx = np.where(labels == pid)[0]
        if idx.size == 0:
            continue
        Pk = points_norm[idx]
        Fk = feats[idx]
        s_k = Fk.mean(axis=0).astype(np.float32)
        c_k = Pk.mean(axis=0).astype(np.float32)
        R_k, scales_k = pca_frame(Pk, eps=eps)
        loglam_k = np.log(scales_k + eps).astype(np.float32)
        tokens[pid] = {
            's': s_k,
            'c': c_k,
            'R': R_k,
            'loglam': loglam_k,
            'points': Pk.astype(np.float32),
        }
    return tokens


# ---------------------------- Pair selection ----------------------------

def select_pairs(part_ids: List[int], centroids: Dict[int, np.ndarray], neighbor_k: int,
                 both_dirs_if_two: bool) -> List[Tuple[int, int]]:
    K = len(part_ids)
    if K <= 1:
        return []
    if K == 2:
        i, j = part_ids[0], part_ids[1]
        return [(i, j), (j, i)] if both_dirs_if_two else [(i, j)]
    # K>2: m-NN by centroid distance
    C = np.stack([centroids[pid] for pid in part_ids], axis=0)
    pairs: List[Tuple[int, int]] = []
    for a, pid in enumerate(part_ids):
        d = np.linalg.norm(C - C[a:a+1], axis=1)
        order = np.argsort(d)
        # skip self (first)
        nb = [part_ids[b] for b in order[1:neighbor_k+1]]
        for q in nb:
            pairs.append((pid, q))
    return pairs


# ---------------------------- Pair feature writer ----------------------------

def write_pair_npz(out_path: str,
                   si: np.ndarray, sj: np.ndarray,
                   ci: np.ndarray, cj: np.ndarray,
                   Ri: np.ndarray, Rj: np.ndarray,
                   loglami: np.ndarray, loglamj: np.ndarray) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_dict: Dict[str, np.ndarray] = {
        's_i': si.astype(np.float32),
        's_j': sj.astype(np.float32),
        'c_i': ci.astype(np.float32),
        'c_j': cj.astype(np.float32),
        'R_i': Ri.astype(np.float32),
        'R_j': Rj.astype(np.float32),
        'loglam_i': loglami.astype(np.float32),
        'loglam_j': loglamj.astype(np.float32),
    }
    np.savez_compressed(out_path, **save_dict)
    return out_path


# ---------------------------- Mobility grouping (decompose by joints) ----------------------------

def _load_mobility_json(path: str) -> List[dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('mobility_v2.json must be a list of nodes')
    return data


def _build_union_find(ids: List[int]):
    parent: Dict[int, int] = {i: i for i in ids}
    rank: Dict[int, int] = {i: 0 for i in ids}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return find, union


def parts_to_joint_groups(mobility_nodes: List[dict]) -> Dict[int, int]:
    """Map primitive part id -> rigid group id by merging along static joints.
    Non-static (hinge/slider/...) remain separate groups.
    """
    nodes: Dict[int, dict] = {}
    for n in mobility_nodes:
        nodes[int(n['id'])] = n

    node_ids = list(nodes.keys())
    find, union = _build_union_find(node_ids)

    # Merge child with parent when joint is 'static'
    for n in mobility_nodes:
        nid = int(n['id'])
        pid = int(n.get('parent', -1))
        joint = str(n.get('joint', 'static')).lower()
        if pid >= 0 and joint == 'static':
            union(nid, pid)

    # Assign each primitive 'parts[].id' to its representative group
    part_to_group: Dict[int, int] = {}
    for n in mobility_nodes:
        rep = find(int(n['id']))
        for p in n.get('parts', []):
            pid = int(p.get('id'))
            part_to_group[pid] = rep
    return part_to_group


# ---------------------------- Object processing ----------------------------

def resolve_object_paths(dataset_root: str, obj_id: str) -> Tuple[str, str, str, str]:
    """Resolve paths for this repo's layout:
    dataset/<obj_id>/
      partfield/part_feat_<obj_id>_0.npy
      point_sample/ply-10000.ply # notice it's ply, not pts-10000.ply
      point_sample/sample-points-all-label-10000.txt
      mobility_v2.json
    """
    base = Path(dataset_root) / obj_id
    partfield = base / 'partfield' / f'part_feat_{obj_id}_0.npy'
    ply = base / 'point_sample' / 'ply-10000.ply'
    labels = base / 'point_sample' / 'sample-points-all-label-10000.txt'
    mobility = base / 'mobility_v2.json'
    if not partfield.exists():
        raise FileNotFoundError(f'PartField not found: {partfield}')
    if not ply.exists():
        raise FileNotFoundError(f'PLY not found: {ply}')
    if not labels.exists():
        raise FileNotFoundError(f'Labels not found: {labels}')
    if not mobility.exists():
        raise FileNotFoundError(f'Mobility json not found: {mobility}')
    return str(ply), str(labels), str(partfield), str(mobility)


def process_object(obj_id: str,
                   dataset_root: str,
                   out_dir: str,
                   cfg: PreprocessConfig) -> List[str]:
    # Paths
    ply_path, labels_path, feats_path, mobility_path = resolve_object_paths(dataset_root, obj_id)
    # Load (coords already normalized in dataset)
    P = read_ply_xyz(ply_path).astype(np.float32)  # (N,3)
    L_raw = read_labels_txt(labels_path).astype(np.int64)  # (N,) raw part ids (not zero-based)
    F = read_features_npy(feats_path).astype(np.float32)  # (N,D)
    assert P.shape[0] == L_raw.shape[0] == F.shape[0], 'Point/label/feature size mismatch'
    if F.shape[1] != cfg.sem_dim:
        raise ValueError(f'Feature dim {F.shape[1]} != sem_dim {cfg.sem_dim}. Set cfg.sem_dim accordingly.')

    # DO NOT normalize P; preserve alignment with mobility_v2.json
    P_use = P

    # Build grouped labels using mobility graph (merge static joints)
    mobility_nodes = _load_mobility_json(mobility_path)
    part_to_group = parts_to_joint_groups(mobility_nodes)

    # Remap group ids to contiguous [0..G-1]
    group_ids_sorted = sorted(set(part_to_group.values()))
    group_to_idx = {g: i for i, g in enumerate(group_ids_sorted)}

    # Create coarse labels, drop points lacking mobility mapping
    labels_coarse = np.full(L_raw.shape, fill_value=-1, dtype=np.int32)
    for i, lbl in enumerate(L_raw.tolist()):
        g = part_to_group.get(int(lbl))
        if g is not None:
            labels_coarse[i] = group_to_idx[g]
    mask = labels_coarse >= 0
    if mask.sum() == 0:
        print(f'[WARN] {obj_id}: no points matched mobility parts; skipping object')
        return []

    P_sub = P_use[mask]
    F_sub = F[mask]
    L_sub = labels_coarse[mask]

    # Tokens on grouped labels
    tokens = build_part_tokens(P_sub, L_sub, F_sub, eps=cfg.eps)
    part_ids = sorted(tokens.keys())
    if len(part_ids) < 2:
        return []
    centroids = {pid: tokens[pid]['c'] for pid in part_ids}

    # Pairs
    pairs = select_pairs(part_ids, centroids, cfg.neighbor_k, cfg.create_both_directions_when_k2)

    saved: List[str] = []
    base = Path(out_dir) / obj_id

    for (pi, pj) in pairs:
        ti = tokens[pi]
        tj = tokens[pj]
        out_path = base / f'edge_{pi}_{pj}.npz'
        path = write_pair_npz(
            str(out_path),
            ti['s'], tj['s'],
            ti['c'], tj['c'],
            ti['R'], tj['R'],
            ti['loglam'], tj['loglam'],
        )
        saved.append(path)

    return saved


# ---------------------------- Dataset processing ----------------------------

def process_category(dataset_root: str, category: str, output_root: str,
                     cfg: PreprocessConfig, index_txt_out: Optional[str] = None) -> List[str]:
    if get_class_ids is None:
        raise ImportError('dataset_utils.get_class_ids not available. Ensure projects/articulate_exp/lib is on PYTHONPATH.')
    obj_ids = get_class_ids(category, dataset_root)
    os.makedirs(output_root, exist_ok=True)

    all_paths: List[str] = []
    for obj_id in obj_ids:
        try:
            paths = process_object(obj_id, dataset_root, output_root, cfg)
        except Exception as e:
            print(f'[WARN] Skip {obj_id}: {e}')
            continue
        all_paths.extend(paths)
        print(f'[OK] {obj_id}: wrote {len(paths)} pairs')

    if index_txt_out is None:
        index_txt_out = str(Path(output_root) / f'{category}_pairs_index.txt')
    with open(index_txt_out, 'w') as f:
        for p in all_paths:
            f.write(p + '\n')
    print(f'[INFO] Wrote index: {index_txt_out} ({len(all_paths)} pairs)')
    return all_paths


# ---------------------------- Entry function (call all steps) ----------------------------

def run_all(dataset_root: str, category: str, output_root: str, sem_dim: int,
            neighbor_k: int = 4) -> List[str]:
    """
    Preprocess one PartNet Mobility category into per-pair NPZ files.

    - dataset_root: PartNet Mobility dataset root that contains <obj_id>/meta.json etc.,
      and your per-object assets under <obj_id>/partfield and <obj_id>/point_sample.
    - category: e.g., "Faucet". Only objects in this class are processed.
    - output_root: where NPZs are written: output_root/<obj_id>/edge_i_j.npz
    - sem_dim: PartField feature dimension.
    """
    cfg = PreprocessConfig(
        sem_dim=sem_dim,
        neighbor_k=neighbor_k,
    )
    return process_category(dataset_root, category, output_root, cfg)


# ---------------------------- CLI ----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess dataset/<obj_id> to pair NPZs (category-filtered)')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root of PartNet Mobility dataset (contains <obj_id>/)')
    parser.add_argument('--category', type=str, required=True, help='Category name to process (e.g., Faucet)')
    parser.add_argument('--output_root', type=str, required=True, help='Output folder for NPZ pairs')
    parser.add_argument('--neighbor-k', type=int, default=4)
    args = parser.parse_args()

    paths = run_all(
        dataset_root=args.dataset_root,
        category=args.category,
        output_root=args.output_root,
        sem_dim=448, # this is what PartField defined
        neighbor_k=args.neighbor_k,
    )
    print(f'Done. Total pairs: {len(paths)}')
