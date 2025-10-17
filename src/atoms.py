import numpy as np
from scipy.spatial import KDTree
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

def build_features(grid_points, c_atoms, h_atoms):
    """
    Build features for each grid point:
    - First 6 features: distances to 6 nearest Carbon atoms (sorted)
    - Next 6 features: distances to 6 nearest Hydrogen atoms (sorted)
    """
    # Build KD-trees for efficient nearest neighbor search
    c_tree = KDTree(c_atoms)
    h_tree = KDTree(h_atoms)
    
    features = []
    
    for point in grid_points:
        point_features = []
        
        # Get distances to C atoms (6 nearest)
        c_dists, c_indices = c_tree.query(point, k=min(6, len(c_atoms)))
        # Sort by distance
        c_dists_sorted = np.sort(c_dists)
        point_features.extend(c_dists_sorted)
        
        # If there are fewer than 6 C atoms, pad with zeros
        if len(c_dists_sorted) < 6:
            point_features.extend([0.0] * (6 - len(c_dists_sorted)))
        
        # Get distances to H atoms (6 nearest)
        h_dists, h_indices = h_tree.query(point, k=min(6, len(h_atoms)))
        # Sort by distance
        h_dists_sorted = np.sort(h_dists)
        point_features.extend(h_dists_sorted)
        
        # If there are fewer than 6 H atoms, pad with zeros
        if len(h_dists_sorted) < 6:
            point_features.extend([0.0] * (6 - len(h_dists_sorted)))
        
        features.append(point_features)
    
    return np.array(features)

#
# Machine Learning Part
#
def train_rf_on_polar_targets(features, polar_targets):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, polar_targets, test_size=0.2, random_state=42
    )

    # Train Random Forest Regressor for both outputs
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    print(f"Test MSE (intensity, angle): {mse}")

    return rf

#
# VTK Writing Part
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
# Main Execution
#
def main():
    # File paths
    vtk_file = "../VTK/Tacne/benz_xy.vtk"  # Replace with your VTK file path
    xyz_file = "../VTK/Tacne/benz.xyz"         # Replace with your XYZ file path
    
    # Read data
    print("Reading XYZ file...")
    c_atoms, h_atoms = read_xyz_file(xyz_file)
    print(f"Found {len(c_atoms)} Carbon atoms and {len(h_atoms)} Hydrogen atoms")
    
    print("Reading VTK file...")
    grid_points, magnetic_vectors, dims = read_vtk_file(vtk_file)
    print(f"Grid dimensions: {dims}")
    print(f"Number of grid points: {len(grid_points)}")
    print(f"Number of magnetic vectors: {len(magnetic_vectors)}")
    
    # Build features
    print("Building features...")
    features = build_features(grid_points, c_atoms, h_atoms)
    
    # Compute polar targets (intensity and angle in XY plane)
    bx = magnetic_vectors[:, 0]
    by = magnetic_vectors[:, 1]
    polar_intensity_xy = np.sqrt(bx**2 + by**2)
    polar_angle_xy = np.arctan2(by, bx)
    polar_targets = np.stack([polar_intensity_xy, polar_angle_xy], axis=1)
    
    
    # Create dataset
    dataset = {
        'features': features,           # Shape: (1089, 12)
        'targets': magnetic_vectors,    # Shape: (1089, 3)
        'grid_points': grid_points,     # Shape: (1089, 3)
        'c_atoms': c_atoms,             # Shape: (6, 3)
        'h_atoms': h_atoms              # Shape: (6, 3)
    }
    
    # Print some statistics
    print("\nFeature statistics:")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {magnetic_vectors.shape}")
    
    print("\nFirst 5 samples:")
    for i in range(min(5, len(features))):
        print(f"Sample {i}:")
        print(f"  Features: {features[i]}")
        print(f"  Target: {magnetic_vectors[i]}")
        print(f"  Grid point: {grid_points[i]}")
        print()
    
    # Feature descriptions
    feature_names = []
    for i in range(6):
        feature_names.append(f"dist_C_{i+1}")
    for i in range(6):
        feature_names.append(f"dist_H_{i+1}")
    
    print("Feature names:", feature_names)
    
    # Save dataset
    np.savez('magnetic_field_dataset.npz', 
             features=features, 
             targets=magnetic_vectors,
             grid_points=grid_points,
             polar_targets=polar_targets,
             c_atoms=c_atoms,
             h_atoms=h_atoms,
             feature_names=feature_names)
    
    print("Dataset saved to 'magnetic_field_dataset.npz'")
    
    print("\nTraining Random Forest model to predict polar_targets (intensity, angle)...")
    rf = train_rf_on_polar_targets(features, polar_targets)
    
    # Predict polar targets for all grid points
    polar_pred = rf.predict(features)
    # Convert polar (intensity, angle) to (Bx, By, Bz=0)
    bx_pred = polar_pred[:, 0] * np.cos(polar_pred[:, 1])
    by_pred = polar_pred[:, 0] * np.sin(polar_pred[:, 1])
    bz_pred = np.zeros_like(bx_pred)
    vectors_pred = np.stack([bx_pred, by_pred, bz_pred], axis=1)

    # Save to VTK
    write_vtk_file("../VTK/Tacne/predicted.vtk", grid_points, vectors_pred, dims)
    print("Predicted vectors saved to predicted.vtk")    
    

if __name__ == "__main__":
    main()
