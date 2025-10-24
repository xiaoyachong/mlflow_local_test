"""
Local test: Compare traditional PCA vs Neural PCA using local models only
Uses dataset-level normalization (1st-99th percentile) instead of per-image normalization
"""
import numpy as np
import torch
import os
import sys
from scipy.ndimage import zoom
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from models.vae.vae import ConvVAE

def resize_image(img, target_size=(128, 128)):
    """Resize image using scipy zoom"""
    h, w = img.shape
    zoom_h, zoom_w = target_size[0] / h, target_size[1] / w
    return zoom(img, (zoom_h, zoom_w), order=1)

def preprocess_with_dataset_norm(img, data_min, data_max):
    """
    Preprocess image using dataset-level normalization
    
    Args:
        img: Input image array
        data_min: Minimum value from dataset (1st percentile)
        data_max: Maximum value from dataset (99th percentile)
    """
    # Clip to dataset range
    img = np.clip(img, data_min, data_max)
    
    # Normalize to [0, 1]
    img = (img - data_min) / (data_max - data_min)
    
    # Convert to tensor: (H, W) -> (1, H, W)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    
    return tensor

def get_latent_features(img_tensor, vae_model, device):
    """Extract latent features from VAE"""
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        _, mu, _ = vae_model(img_tensor)
        latent = mu.cpu().numpy()
    
    return latent

def get_pca_coords(latent, pca_model):
    """Get PCA coordinates from latent features"""
    coords = pca_model.transform(latent)
    return coords

def get_neural_pca_coords(latent, scaler, neural_model, device):
    """Get Neural PCA coordinates from latent features"""
    # Scale latent features
    latent_scaled = scaler.transform(latent)
    
    # Apply neural network
    with torch.no_grad():
        coords = neural_model(torch.tensor(latent_scaled, dtype=torch.float32).to(device)).cpu().numpy()
    
    return coords

class SimpleDimRedApproximator(torch.nn.Module):
    """Neural network for dimensionality reduction approximation"""
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=2):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, h))
            layers.append(torch.nn.ReLU())
            input_dim = h
        layers.append(torch.nn.Linear(input_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))

def plot_comparison(results, output_dir='local_comparison_results'):
    """Create comprehensive visualizations comparing PCA vs Neural PCA"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    coords_pca = np.array([r['coords_pca'] for r in results])
    coords_neural = np.array([r['coords_neural'] for r in results])
    print("=========:",coords_pca.shape,coords_neural.shape)
    bin_nums = np.array([r['bin_num'] for r in results])
    
    # Get unique bins and colors
    unique_bins = sorted(set(bin_nums))
    from matplotlib import cm
    colors_tab10 = cm.tab10(np.linspace(0, 1, 10))
    bin_colors = {bin_num: colors_tab10[i % 10] for i, bin_num in enumerate(unique_bins)}
    
    # Figure 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot PCA results
    for bin_num in unique_bins:
        mask = bin_nums == bin_num
        axes[0].scatter(coords_pca[mask, 0], coords_pca[mask, 1], 
                       c=[bin_colors[bin_num]], s=30, alpha=0.6, label=f'Bin {bin_num}')
    axes[0].set_title('Traditional PCA', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Neural PCA results
    for bin_num in unique_bins:
        mask = bin_nums == bin_num
        axes[1].scatter(coords_neural[mask, 0], coords_neural[mask, 1], 
                       c=[bin_colors[bin_num]], s=30, alpha=0.6, label=f'Bin {bin_num}')
    axes[1].set_title('Neural PCA', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'pca_vs_neural_side_by_side.png')
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig1_path}")
    
    # Figure 2: Overlay comparison
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for bin_num in unique_bins:
        mask = bin_nums == bin_num
        ax.scatter(coords_pca[mask, 0], coords_pca[mask, 1], 
                  c=[bin_colors[bin_num]], s=50, alpha=0.5, marker='o', 
                  label=f'Bin {bin_num} (PCA)')
        ax.scatter(coords_neural[mask, 0], coords_neural[mask, 1], 
                  c=[bin_colors[bin_num]], s=50, alpha=0.5, marker='x')
    
    # Draw lines connecting corresponding points (sample a few)
    sample_indices = np.random.choice(len(results), min(50, len(results)), replace=False)
    for i in sample_indices:
        ax.plot([coords_pca[i, 0], coords_neural[i, 0]], 
               [coords_pca[i, 1], coords_neural[i, 1]], 
               'gray', alpha=0.2, linewidth=0.5)
    
    ax.set_title('Overlay Comparison\n(o = PCA, x = Neural PCA)', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'pca_vs_neural_overlay.png')
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig2_path}")
    
    # Figure 3: Difference analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Coordinate differences
    coord_diff = np.abs(coords_pca - coords_neural)
    euclidean_dists = np.linalg.norm(coords_pca - coords_neural, axis=1)
    
    # Histogram of PC1 differences
    axes[0, 0].hist(coord_diff[:, 0], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(coord_diff[:, 0]), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(coord_diff[:, 0]):.6f}')
    axes[0, 0].set_title('PC1 Absolute Difference', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Absolute Difference', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of PC2 differences
    axes[0, 1].hist(coord_diff[:, 1], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(coord_diff[:, 1]), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(coord_diff[:, 1]):.6f}')
    axes[0, 1].set_title('PC2 Absolute Difference', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Absolute Difference', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of Euclidean distances
    axes[1, 0].hist(euclidean_dists, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(euclidean_dists), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(euclidean_dists):.6f}')
    axes[1, 0].set_title('Euclidean Distance', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Distance', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter: Distance vs Bin
    for bin_num in unique_bins:
        mask = bin_nums == bin_num
        axes[1, 1].scatter([bin_num]*np.sum(mask), euclidean_dists[mask], 
                          c=[bin_colors[bin_num]], s=20, alpha=0.6, label=f'Bin {bin_num}')
    axes[1, 1].set_title('Euclidean Distance by Bin', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Bin Number', fontsize=11)
    axes[1, 1].set_ylabel('Euclidean Distance', fontsize=11)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'pca_vs_neural_difference.png')
    plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig3_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples plotted: {len(results)}")
    print(f"Number of bins: {len(unique_bins)}")
    print(f"\nCoordinate Differences:")
    print(f"  PC1 - Mean: {np.mean(coord_diff[:, 0]):.6f}, Max: {np.max(coord_diff[:, 0]):.6f}")
    print(f"  PC2 - Mean: {np.mean(coord_diff[:, 1]):.6f}, Max: {np.max(coord_diff[:, 1]):.6f}")
    print(f"  Euclidean - Mean: {np.mean(euclidean_dists):.6f}, Max: {np.max(euclidean_dists):.6f}")
    print(f"\nFiles saved in: {output_dir}/")

def main():
    # Configuration
    data_path = "shot_averages/combined_all_bin_averages.npy"
    vae_weights_path = "../../models/trs_vae/vae_model_128_weights.npz"
    pca_model_path = "../../models/trs_vae/pca_model.joblib"
    neural_pca_model_path = "../../models/trs_vae/neural_pca_model.pth"
    scaler_path = "../../models/trs_vae/scaler.pkl"
    output_dir = "local_comparison_results"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("LOCAL MODEL COMPARISON: PCA vs NEURAL PCA")
    print("="*80)
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"VAE weights: {vae_weights_path}")
    print(f"PCA model: {pca_model_path}")
    print(f"Neural PCA model: {neural_pca_model_path}")
    print(f"Scaler: {scaler_path}")
    
    # Load data
    print(f"\nLoading data...")
    combined_dict = np.load(data_path, allow_pickle=True).item()
    keys = list(combined_dict.keys())
    images = list(combined_dict.values())
    print(f"✅ Loaded {len(images)} images")
    
    # Resize all images first
    print(f"\nResizing images to 128x128...")
    resized_images = [resize_image(img, target_size=(128, 128)) for img in tqdm(images, desc="Resizing")]
    resized_samples = np.array(resized_images)
    
    # Calculate dataset-level normalization parameters (1st-99th percentile)
    print(f"\nCalculating dataset-level normalization parameters...")
    data_min = np.percentile(resized_samples, 1)  # 1st percentile
    data_max = np.percentile(resized_samples, 99)  # 99th percentile
    print(f"✅ Data min (1st percentile): {data_min:.6f}")
    print(f"✅ Data max (99th percentile): {data_max:.6f}")
    
    # Load VAE model
    print(f"\nLoading VAE model...")
    vae_model = ConvVAE(latent_dim=256, image_size=(128, 128))
    weights_npz = np.load(vae_weights_path)
    state_dict = {key: torch.tensor(weights_npz[key]) for key in weights_npz.files}
    vae_model.load_state_dict(state_dict, strict=True)
    vae_model.eval()
    vae_model = vae_model.to(device)
    print(f"✅ VAE model loaded")
    
    # Load PCA model
    print(f"\nLoading PCA model...")
    pca_model = joblib.load(pca_model_path)
    print(f"✅ PCA model loaded")
    print(f"   PCA n_components: {pca_model.n_components_}")
    
    # Load Neural PCA model and scaler
    print(f"\nLoading Neural PCA model...")
    scaler = joblib.load(scaler_path)
    checkpoint = torch.load(neural_pca_model_path, map_location=device)
    
    # Determine output_dim from checkpoint if available
    output_dim = checkpoint.get('output_dim', 2)
    neural_model = SimpleDimRedApproximator(input_dim=256, output_dim=output_dim)
    neural_model.load_state_dict(checkpoint['model_state_dict'])
    neural_model.eval()
    neural_model = neural_model.to(device)
    print(f"✅ Neural PCA model loaded")
    print(f"   Neural PCA output_dim: {output_dim}")
    
    # Process all images
    print("\n" + "="*80)
    print(f"Processing ALL {len(images)} images...")
    print("="*80)
    
    results = []
    
    for idx in tqdm(range(len(resized_images)), desc="Processing"):
        img_resized = resized_images[idx]
        key = keys[idx]
        bin_num, shot_num = key
        
        # Preprocess with dataset-level normalization
        img_tensor = preprocess_with_dataset_norm(img_resized, data_min, data_max)
        
        # Get latent features (same for both methods)
        latent = get_latent_features(img_tensor, vae_model, device)
        
        # Get PCA coordinates
        coords_pca = get_pca_coords(latent, pca_model)
        
        # Get Neural PCA coordinates
        coords_neural = get_neural_pca_coords(latent, scaler, neural_model, device)
        
        # Debug: Print shapes for first iteration
        if idx == 0:
            print(f"\nDebug - First sample shapes:")
            print(f"  Latent shape: {latent.shape}")
            print(f"  PCA coords shape: {coords_pca.shape}")
            print(f"  Neural PCA coords shape: {coords_neural.shape}")
        
        results.append({
            'idx': idx,
            'key': key,
            'bin_num': bin_num,
            'shot_num': shot_num,
            'latent': latent[0],
            'coords_pca': coords_pca[0],
            'coords_neural': coords_neural[0],
        })
    
    print(f"\n✅ Processed {len(results)} images")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_comparison(results, output_dir=output_dir)
    
    # Calculate final statistics
    coords_pca = np.array([r['coords_pca'] for r in results])
    coords_neural = np.array([r['coords_neural'] for r in results])
    
    corr = np.corrcoef(coords_pca.flatten(), coords_neural.flatten())[0, 1]
    mse = np.mean((coords_pca - coords_neural)**2)
    euclidean_mean = np.mean(np.linalg.norm(coords_pca - coords_neural, axis=1))
    
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Overall Correlation: {corr:.8f}")
    print(f"Overall MSE: {mse:.8f}")
    print(f"Mean Euclidean Distance: {euclidean_mean:.8f}")
    
    if corr > 0.9999:
        print("\n✅ IDENTICAL: PCA and Neural PCA produce nearly identical results!")
    elif corr > 0.99:
        print("\n✅ VERY SIMILAR: Minor differences only")
    elif corr > 0.95:
        print("\n⚠️  MODERATELY SIMILAR: Some notable differences")
    else:
        print("\n❌ DIFFERENT: Significant differences detected")
    
    print("="*80)

if __name__ == "__main__":
    main()