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
    
    # Get atom count from first line
    atom_count = int(lines[0].strip())
    
    # Parse atoms
    atoms = []
    for line in lines[2:2+atom_count]:
        parts = line.strip().split()
        if len(parts) >= 4:
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, np.array([x, y, z])))
    
    # Separate C and H atoms
    c_atoms = [atom[1] for atom in atoms if atom[0] == '6' or atom[0] == 'C']
    h_atoms = [atom[1] for atom in atoms if atom[0] == '1' or atom[0] == 'H']
    
    return np.array(c_atoms) if c_atoms else np.empty((0,3)), np.array(h_atoms) if h_atoms else np.empty((0,3))

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
            f.write(f"    {pt[0]: .6f} {pt[1]: .6f} {pt[2]: .6f}\n")
        f.write(f"\nPOINT_DATA {len(vectors)}\n")
        f.write("VECTORS point_vectors float\n")
        for v in vectors:
            f.write(f"    {v[0]: .6f} {v[1]: .6f} {v[2]: .6f}\n")

#
# RBF layer for distance expansion
#
class RBF(nn.Module):
    def __init__(self, num_centers=16, cutoff=6.0):
        super().__init__()
        self.num_centers = num_centers
        centers = torch.linspace(0.0, cutoff, num_centers)
        self.register_buffer('centers', centers)
        # Make gamma learnable
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, d):
        # d: (...,) distances
        d_exp = d.unsqueeze(-1)  # (..., 1)
        c = self.centers.view([*(([1] * (d_exp.dim() - 1))), -1])  # (..., C)
        diff = d_exp - c
        return torch.exp(-self.gamma * (diff ** 2))  # (..., C)


class PaddedAtomQueryFieldNet(nn.Module):
    """
    Fixed-size network with padding for variable number of atoms.
    Maximum atoms = 40 (based on requirement)
    """
    def __init__(self, max_atoms=40, atom_feat_dim=2, rbf_centers=32, hidden=128, cutoff=8.0):
        super().__init__()
        self.max_atoms = max_atoms
        self.cutoff = cutoff
        
        self.rbf = RBF(num_centers=rbf_centers, cutoff=cutoff)
        
        # Process each atom independently
        self.atom_mlp = nn.Sequential(
            nn.Linear(atom_feat_dim + rbf_centers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Scalar weight output
        self.scalar_out = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Learnable embedding for padding atoms (will be ignored via mask)
        self.register_buffer('padding_feat', torch.zeros(1, atom_feat_dim))

    def forward(self, atom_pos, atom_feat, query_pos, atom_mask):
        """
        atom_pos: (max_atoms, 3) - padded atom positions
        atom_feat: (max_atoms, F) - padded atom features
        query_pos: (Nq, 3) - query points
        atom_mask: (max_atoms,) - 1 for real atoms, 0 for padding
        """
        # Expand dimensions for broadcasting
        rel = query_pos.unsqueeze(1) - atom_pos.unsqueeze(0)  # (Nq, max_atoms, 3)
        dist = torch.norm(rel + 1e-12, dim=-1)  # (Nq, max_atoms)
        
        # Create mask: real atoms within cutoff
        cutoff_mask = (dist <= self.cutoff).float()  # (Nq, max_atoms)
        mask = atom_mask.unsqueeze(0) * cutoff_mask  # (Nq, max_atoms)
        
        # RBF embedding
        rbf = self.rbf(dist)  # (Nq, max_atoms, C)
        
        # Broadcast atom features
        atom_feat_b = atom_feat.unsqueeze(0).expand(query_pos.size(0), -1, -1)  # (Nq, max_atoms, F)
        
        # Concatenate and process
        atom_input = torch.cat([atom_feat_b, rbf], dim=-1)  # (Nq, max_atoms, F+C)
        h = self.atom_mlp(atom_input)  # (Nq, max_atoms, H)
        scalar_w = self.scalar_out(h).squeeze(-1)  # (Nq, max_atoms)
        
        # Apply mask (zero out padding and atoms beyond cutoff)
        scalar_w = scalar_w * mask
        
        # Unit direction vectors
        dir_vec = rel / (dist.unsqueeze(-1) + 1e-12)  # (Nq, max_atoms, 3)
        
        # Weighted sum (padded atoms contribute zero due to mask)
        weighted = (scalar_w.unsqueeze(-1) * dir_vec).sum(dim=1)  # (Nq, 3)
        
        # Final projection
        out = self.final_proj(weighted)  # (Nq, 3)
        
        return out


def atom_features_from_xyz(c_atoms, h_atoms, max_atoms=40):
    """
    Returns padded atom positions and features.
    Also returns mask indicating real atoms.
    """
    pos_list = []
    feats = []
    
    # Add Carbon atoms
    for p in c_atoms:
        pos_list.append(p)
        feats.append([1.0, 0.0])  # [is_C, is_H]
    
    # Add Hydrogen atoms
    for p in h_atoms:
        pos_list.append(p)
        feats.append([0.0, 1.0])
    
    n_atoms = len(pos_list)
    if n_atoms == 0:
        return np.zeros((max_atoms, 3)), np.zeros((max_atoms, 2)), np.zeros(max_atoms)
    
    # Convert to arrays
    atom_pos = np.vstack(pos_list)
    atom_feat = np.vstack(feats)
    
    # Create mask
    mask = np.ones(max_atoms)
    mask[n_atoms:] = 0
    
    # Pad to max_atoms
    if n_atoms < max_atoms:
        padded_pos = np.zeros((max_atoms, 3))
        padded_feat = np.zeros((max_atoms, 2))
        padded_pos[:n_atoms] = atom_pos
        padded_feat[:n_atoms] = atom_feat
    else:
        # Truncate if more than max_atoms (shouldn't happen with max_atoms=40)
        padded_pos = atom_pos[:max_atoms]
        padded_feat = atom_feat[:max_atoms]
        mask = np.ones(max_atoms)
    
    return padded_pos, padded_feat, mask


def load_molecule_data(xyz_file, vtk_file, max_atoms=40):
    """Load and prepare data for a single molecule"""
    c_atoms, h_atoms = read_xyz_file(xyz_file)
    grid_points, magnetic_vectors, dims = read_vtk_file(vtk_file)
    
    atom_pos_pad, atom_feat_pad, atom_mask = atom_features_from_xyz(c_atoms, h_atoms, max_atoms)
    
    return {
        'atom_pos': atom_pos_pad,
        'atom_feat': atom_feat_pad,
        'atom_mask': atom_mask,
        'query_pos': grid_points,
        'target': magnetic_vectors,
        'dims': dims,
        'name': os.path.basename(xyz_file).replace('.xyz', '')
    }


def train_epoch(model, optimizer, molecule_data, device):
    """Train for one epoch on a single molecule"""
    model.train()
    
    # Move data to device
    atom_pos = torch.tensor(molecule_data['atom_pos'], dtype=torch.float32, device=device)
    atom_feat = torch.tensor(molecule_data['atom_feat'], dtype=torch.float32, device=device)
    atom_mask = torch.tensor(molecule_data['atom_mask'], dtype=torch.float32, device=device)
    query_pos = torch.tensor(molecule_data['query_pos'], dtype=torch.float32, device=device)
    targets = torch.tensor(molecule_data['target'], dtype=torch.float32, device=device)
    
    optimizer.zero_grad()
    pred = model(atom_pos, atom_feat, query_pos, atom_mask)
    loss = nn.MSELoss()(pred, targets)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, molecule_data, device):
    """Evaluate on a molecule"""
    model.eval()
    
    with torch.no_grad():
        atom_pos = torch.tensor(molecule_data['atom_pos'], dtype=torch.float32, device=device)
        atom_feat = torch.tensor(molecule_data['atom_feat'], dtype=torch.float32, device=device)
        atom_mask = torch.tensor(molecule_data['atom_mask'], dtype=torch.float32, device=device)
        query_pos = torch.tensor(molecule_data['query_pos'], dtype=torch.float32, device=device)
        targets = torch.tensor(molecule_data['target'], dtype=torch.float32, device=device)
        
        pred = model(atom_pos, atom_feat, query_pos, atom_mask)
        loss = nn.MSELoss()(pred, targets)
        
        return loss.item(), pred.cpu().numpy()


def main():
    # Configuration
    MAX_ATOMS = 40
    RBF_CENTERS = 32
    HIDDEN_SIZE = 128
    CUTOFF = 8.0
    EPOCHS = 2000
    LEARNING_RATE = 1e-3
    
    # Data paths
    base_path = "../VTK/Tacne"
    train_molecules = [
        #('benz.xyz', 'benz_xy.vtk'),
        ('DBP.xyz', 'DBP_xy.vtk')
    ]
    #test_molecule = ('ZnPorf.xyz', 'ZnPorf_xy.vtk')
    test_molecule = ('benz.xyz', 'benz_xy.vtk')
    #test_molecule = ('DBP.xyz', 'DBP_xy.vtk')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training data
    print("\nLoading training data...")
    train_data = []
    for xyz_file, vtk_file in train_molecules:
        print(f"  Loading {xyz_file} and {vtk_file}")
        data = load_molecule_data(
            os.path.join(base_path, xyz_file),
            os.path.join(base_path, vtk_file),
            MAX_ATOMS
        )
        train_data.append(data)
        print(f"    Atoms: {int(data['atom_mask'].sum())}")
        print(f"    Query points: {len(data['query_pos'])}")
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_molecule_data(
        os.path.join(base_path, test_molecule[0]),
        os.path.join(base_path, test_molecule[1]),
        MAX_ATOMS
    )
    print(f"  {test_molecule[0]}: {int(test_data['atom_mask'].sum())} atoms")
    print(f"  Query points: {len(test_data['query_pos'])}")
    
    # Create model
    model = PaddedAtomQueryFieldNet(
        max_atoms=MAX_ATOMS,
        atom_feat_dim=2,
        rbf_centers=RBF_CENTERS,
        hidden=HIDDEN_SIZE,
        cutoff=CUTOFF
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    # Training loop
    print("\nStarting training...")
    best_test_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # Train on each molecule
        epoch_loss = 0
        for mol_data in train_data:
            loss = train_epoch(model, optimizer, mol_data, device)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / len(train_data)
        
        # Evaluate on test set every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            test_loss, _ = evaluate(model, test_data, device)
            
            print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.6e} | Test Loss: {test_loss:.6e}")
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), 'best_padded_model.pth')
                print(f"  → New best model saved! (Test loss: {test_loss:.6e})")
            
            # Learning rate scheduling
            scheduler.step(test_loss)
    
    # Load best model and evaluate thoroughly
    print(f"\nBest test loss: {best_test_loss:.6e}")
    model.load_state_dict(torch.load('best_padded_model.pth'))
    
    # Final evaluation on all molecules
    print("\nFinal Evaluation:")
    print("-" * 50)
    
    for i, mol_data in enumerate(train_data):
        loss, pred = evaluate(model, mol_data, device)
        print(f"Train - {mol_data['name']}: MSE = {loss:.6e}")
        
        # Save predictions for training molecules
        out_vtk = os.path.join(base_path, f"pred_{mol_data['name']}.vtk")
        write_vtk_file(out_vtk, mol_data['query_pos'], pred, mol_data['dims'])
        print(f"  Saved: {out_vtk}")
    
    # Test molecule
    test_loss, test_pred = evaluate(model, test_data, device)
    print(f"Test  - {test_data['name']}: MSE = {test_loss:.6e}")
    
    # Save test predictions
    out_vtk = os.path.join(base_path, f"pred_{test_data['name']}.vtk")
    write_vtk_file(out_vtk, test_data['query_pos'], test_pred, test_data['dims'])
    print(f"Saved: {out_vtk}")
    
    # Save all predictions as numpy for analysis
    np.savez('all_predictions.npz',
             #train_benz_pred=evaluate(model, train_data[0], device)[1],
             train_dbp_pred=evaluate(model, train_data[0], device)[1],
             #test_znporf_pred=test_pred
             #test_dpb_pred=test_pred)
             test_benz_pred=test_pred)


if __name__ == "__main__":
    main()

