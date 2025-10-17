import os
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Read atom locations from XYZ file 
#
def read_xyz_file(filename):
    """Read XYZ file and separate Carbon and Hydrogen atoms"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip the first two lines (atom count and comment)
    atoms = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, np.array([x, y, z])))
    
    print(atoms)
    # Separate C and H atoms
    c_atoms = [atom[1] for atom in atoms if atom[0] == '6']
    h_atoms = [atom[1] for atom in atoms if atom[0] == '1']
    
    return np.array(c_atoms), np.array(h_atoms)

#
# Read VTK file to get grid points and magnetic field vectors
#
def read_vtk_file(filename):
    """Read VTK file and extract grid points and magnetic field vectors"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the dimensions line
    dim_line = None
    for i, line in enumerate(lines):
        if line.startswith('DIMENSIONS'):
            dim_line = i
            break
    
    if dim_line is None:
        raise ValueError("Could not find DIMENSIONS in VTK file")
    
    # Parse dimensions
    dims = list(map(int, lines[dim_line].split()[1:4]))
    
    # Find POINTS section
    points_start = None
    for i, line in enumerate(lines):
        if line.startswith('POINTS'):
            points_start = i + 1
            break
    
    if points_start is None:
        raise ValueError("Could not find POINTS in VTK file")
    
    # Read grid points
    grid_points = []
    i = points_start
    points_count = int(lines[points_start-1].split()[1])
    
    for j in range(points_count):
        if i + j >= len(lines):
            break
        coords = list(map(float, lines[i + j].split()))
        grid_points.append(coords)
    
    grid_points = np.array(grid_points)
    
    # Find VECTORS section
    vectors_start = None
    for i, line in enumerate(lines):
        if line.startswith('VECTORS'):
            vectors_start = i + 1
            break
    
    if vectors_start is None:
        raise ValueError("Could not find VECTORS in VTK file")
    
    # Read magnetic field vectors
    magnetic_vectors = []
    for j in range(points_count):
        if vectors_start + j >= len(lines):
            break
        vector = list(map(float, lines[vectors_start + j].split()))
        magnetic_vectors.append(vector)
    
    magnetic_vectors = np.array(magnetic_vectors)
    
    return grid_points, magnetic_vectors, dims

#
# Write VTK file with grid points and magnetic field vectors
#
def write_vtk_file(filename, grid_points, vectors, dims):
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Predicted magnetic field\n")
        f.write("ASCII\n\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {dims[0]} {dims[1]} {dims[2]}\n")
        f.write(f"POINTS {len(grid_points)} float\n")
        for pt in grid_points:
            f.write(f"    {pt[0]: .3f} {pt[1]: .3f} {pt[2]: .3f}\n")
        f.write(f"\nPOINT_DATA {len(vectors)}\n")
        f.write("VECTORS point_vectors float\n")
        for v in vectors:
            f.write(f"    {v[0]: .3f} {v[1]: .3f} {v[2]: .3f}\n")

#
# RBF layer for distance expansion
#
class RBF(nn.Module):
    def __init__(self, num_centers=16, cutoff=6.0):
        super().__init__()
        self.num_centers = num_centers
        centers = torch.linspace(0.0, cutoff, num_centers)
        self.register_buffer('centers', centers)
        self.gamma = nn.Parameter(torch.tensor(10.0), requires_grad=False)

    def forward(self, d):
        # d: (...,) distances
        d_exp = d.unsqueeze(-1)  # (..., 1)
        c = self.centers.view([*(([1] * (d_exp.dim() - 1))), -1])  # (..., C)
        diff = d_exp - c
        return torch.exp(-self.gamma * (diff ** 2))  # (..., C)


class AtomQueryFieldNet(nn.Module):
    """
    Simple equivariant-inspired model:
      - For each query point q and atom a compute vector r = q - r_a
      - Use radial basis expansion of |r| and atom scalar features to produce a scalar weight w(q,a)
      - Contribution to B at q from atom a is w(q,a) * (r / |r|)
      - Sum contributions over atoms and pass through small MLP
    This produces outputs that rotate with coordinates (constructed from r vectors).
    """
    def __init__(self, atom_feat_dim=2, rbf_centers=16, hidden=64, cutoff=6.0):
        super().__init__()
        self.rbf = RBF(num_centers=rbf_centers, cutoff=cutoff)
        self.atom_mlp = nn.Sequential(
            nn.Linear(atom_feat_dim + rbf_centers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # scalar weight -> multiply unit vector
        self.scalar_out = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        # final small projection on aggregated vector
        self.final_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.cutoff = cutoff

    def forward(self, atom_pos, atom_feat, query_pos):
        """
        atom_pos: (Na,3) tensor
        atom_feat: (Na, F) tensor (scalar features per atom)
        query_pos: (Nq,3) tensor
        returns: preds (Nq,3)
        """
        # Expand to (Nq, Na, 3)
        rel = query_pos.unsqueeze(1) - atom_pos.unsqueeze(0)  # (Nq,Na,3)
        dist = torch.norm(rel + 1e-12, dim=-1)  # (Nq,Na)
        # Mask atoms outside cutoff (optional)
        mask = (dist <= self.cutoff).float()  # (Nq,Na)

        # RBF embedding
        rbf = self.rbf(dist)  # (Nq,Na,C)
        # Prepare atom_feat broadcasted
        atom_feat_b = atom_feat.unsqueeze(0).expand(query_pos.size(0), -1, -1)  # (Nq,Na,F)
        atom_input = torch.cat([atom_feat_b, rbf], dim=-1)  # (Nq,Na,F+C)
        # Pass through atom MLP -> per (q,a) embedding
        h = self.atom_mlp(atom_input)  # (Nq,Na,H)
        scalar_w = self.scalar_out(h).squeeze(-1)  # (Nq,Na)
        scalar_w = scalar_w * mask  # zero out contributions from far atoms

        # Unit direction vectors
        dir_vec = rel / (dist.unsqueeze(-1) + 1e-12)  # (Nq,Na,3)
        # Weighted sum of directions
        weighted = (scalar_w.unsqueeze(-1) * dir_vec).sum(dim=1)  # (Nq,3)
        # Final small projection
        out = self.final_proj(weighted)  # (Nq,3)
        return out


# ---- utility to build simple atom features (one-hot C/H) ----
def atom_features_from_xyz(c_atoms, h_atoms):
    """
    Returns:
      atom_pos: (Na,3) numpy array
      atom_feat: (Na,2) numpy array one-hot [is_C, is_H]
    """
    pos_list = []
    feats = []
    for p in c_atoms:
        pos_list.append(p)
        feats.append([1.0, 0.0])
    for p in h_atoms:
        pos_list.append(p)
        feats.append([0.0, 1.0])
    if len(pos_list) == 0:
        return np.zeros((0,3)), np.zeros((0,2))
    return np.vstack(pos_list), np.vstack(feats)


# ---- Training / run script ----
def main():
    # Paths (edit if needed)
    vtk_file = "../VTK/Tacne/benz_xy.vtk"
    xyz_file = "../VTK/Tacne/benz.xyz"
    out_vtk = "../VTK/Tacne/predicted_gnn.vtk"

    # Read inputs using copied routines
    print("Reading data...")
    c_atoms, h_atoms = read_xyz_file(xyz_file)
    grid_points, magnetic_vectors, dims = read_vtk_file(vtk_file)

    atom_pos_np, atom_feat_np = atom_features_from_xyz(c_atoms, h_atoms)
    if atom_pos_np.size == 0:
        raise RuntimeError("No atoms found from XYZ parser.")

    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    atom_pos = torch.tensor(atom_pos_np, dtype=torch.float32, device=device)
    atom_feat = torch.tensor(atom_feat_np, dtype=torch.float32, device=device)
    query_pos = torch.tensor(grid_points, dtype=torch.float32, device=device)
    targets = torch.tensor(magnetic_vectors, dtype=torch.float32, device=device)

    # Model
    model = AtomQueryFieldNet(atom_feat_dim=atom_feat.shape[1], rbf_centers=32, hidden=128, cutoff=8.0)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    # Training loop (small number of epochs; increase if needed)
    epochs = 1000
    batch_query = query_pos  # we process all queries together (single molecule)
    best_loss = 1e9
    model.train()
    for epoch in range(1, epochs + 1):
        optim.zero_grad()
        pred = model(atom_pos, atom_feat, batch_query)  # (Nq,3)
        loss = loss_fn(pred, targets)
        loss.backward()
        optim.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}  Loss: {loss.item():.6e}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            # optional: save best weights
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best (if saved)
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        pred_vecs = model(atom_pos, atom_feat, query_pos).cpu().numpy()

    # Save predicted vectors to VTK using copied routine
    write_vtk_file(out_vtk, grid_points, pred_vecs, dims)
    print(f"Saved predicted field to: {out_vtk}")
    # Also save numpy for inspection
    np.savez('gnn_prediction.npz', grid=grid_points, target=magnetic_vectors, pred=pred_vecs)

if __name__ == "__main__":
    main()
