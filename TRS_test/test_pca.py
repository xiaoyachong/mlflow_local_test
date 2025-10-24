"""
Simple test: Compare latent features AND PCA results from NPZ+joblib vs MLflow models
WITH VISUALIZATION
"""
import numpy as np
import torch
import mlflow
import os
import sys
from scipy.ndimage import zoom
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
import joblib
import matplotlib.pyplot as plt

# Load environment
load_dotenv(dotenv_path="../../.env")

# MLflow setup
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

# Now you can import from src
from src.utils.mlflow_utils import MLflowClient
from models.vae.vae import ConvVAE

def resize_image(img, target_size=(128, 128)):
    h, w = img.shape
    zoom_h, zoom_w = target_size[0] / h, target_size[1] / w
    return zoom(img, (zoom_h, zoom_w), order=1)

def preprocess_like_wrapper(img_array, image_size=(128, 128)):
    """
    Preprocess image EXACTLY like vae_wrapper.py does
    """
    img_array = np.squeeze(img_array)
    
    if img_array.dtype == np.uint8:
        img_array_uint8 = img_array
    else:
        array_min = img_array.min()
        array_max = img_array.max()
        if array_max > array_min:
            img_array_uint8 = (
                ((img_array.astype(np.float32) - array_min)
                / (array_max - array_min))
                * 255
            ).astype(np.uint8)
        else:
            img_array_uint8 = np.zeros_like(img_array, dtype=np.uint8)
    
    pil_image = Image.fromarray(img_array_uint8)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ])
    
    tensor = transform(pil_image)
    return tensor

def get_results_from_npz_and_joblib(img, vae_weights_path, pca_path, scaler_path=None, is_neural=False):
    """
    Get both latent features and PCA/Neural PCA coords using NPZ VAE + joblib model
    
    Args:
        img: Input image
        vae_weights_path: Path to VAE weights
        pca_path: Path to PCA or Neural PCA model
        scaler_path: Path to StandardScaler (only used for neural PCA)
        is_neural: True for neural PCA, False for traditional PCA
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VAE model
    vae_model = ConvVAE(latent_dim=256, image_size=(128, 128))
    weights_npz = np.load(vae_weights_path)
    state_dict = {key: torch.tensor(weights_npz[key]) for key in weights_npz.files}
    vae_model.load_state_dict(state_dict, strict=True)
    vae_model.eval()
    vae_model = vae_model.to(device)
    
    # Preprocess
    tensor = preprocess_like_wrapper(img, image_size=(128, 128))
    tensor = tensor.unsqueeze(0).to(device)
    
    # Get latent features
    with torch.no_grad():
        _, mu, _ = vae_model(tensor)
        latent = mu.cpu().numpy()
    
    # Load dimensionality reduction model
    if is_neural:
        # Neural PCA: Apply StandardScaler then use neural network
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            latent_scaled = scaler.transform(latent)
        else:
            latent_scaled = latent
        
        # Load neural PCA model (PyTorch)
        import torch.nn as nn
        
        class SimpleDimRedApproximator(nn.Module):
            def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=2):
                super().__init__()
                layers = []
                for h in hidden_dims:
                    layers.append(nn.Linear(input_dim, h))
                    layers.append(nn.ReLU())
                    input_dim = h
                layers.append(nn.Linear(input_dim, output_dim))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x.view(x.size(0), -1))
        
        checkpoint = torch.load(pca_path, map_location=device)
        neural_model = SimpleDimRedApproximator(input_dim=256)
        neural_model.load_state_dict(checkpoint['model_state_dict'])
        neural_model.eval()
        neural_model = neural_model.to(device)
        
        with torch.no_grad():
            coords = neural_model(torch.tensor(latent_scaled, dtype=torch.float32).to(device)).cpu().numpy()
    else:
        # Traditional PCA: Direct transformation (no preprocessing)
        pca_model = joblib.load(pca_path)
        coords = pca_model.transform(latent)
    
    return latent, coords

def get_results_from_mlflow(img, mlflow_client, vae_model_name, dimred_model_name):
    """
    Get both latent features and dimensionality reduction coords using MLflow models
    """
    vae_model = mlflow_client.load_model(vae_model_name)
    dimred_model = mlflow_client.load_model(dimred_model_name)
    
    # Get latent features from VAE
    vae_result = vae_model.predict(img)
    latent = vae_result["latent_features"]
    
    # Get dimensionality reduction coords
    dimred_result = dimred_model.predict(latent)
    coords = dimred_result["coords"]
    
    return latent, coords

def plot_comparison(results, output_dir='comparison_results', model_type='PCA'):
    """
    Create comprehensive visualizations comparing NPZ+joblib vs MLflow
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    coords_npz = np.array([r['coords_npz'] for r in results])
    coords_mlflow = np.array([r['coords_mlflow'] for r in results])
    bin_nums = np.array([r['bin_num'] for r in results])
    shot_nums = np.array([r['shot_num'] for r in results])
    
    # Get unique bins and colors
    unique_bins = sorted(set(bin_nums))
    from matplotlib import cm
    colors_tab10 = cm.tab10(np.linspace(0, 1, 10))
    bin_colors = {bin_num: colors_tab10[i % 10] for i, bin_num in enumerate(unique_bins)}
    
    # Figure 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot NPZ+joblib results
    for bin_num in unique_bins:
        mask = bin_nums == bin_num
        axes[0].scatter(coords_npz[mask, 0], coords_npz[mask, 1], 
                       c=[bin_colors[bin_num]], s=30, alpha=0.6, label=f'Bin {bin_num}')
    axes[0].set_title(f'NPZ VAE + joblib {model_type}\n(Local Models)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot MLflow results
    for bin_num in unique_bins:
        mask = bin_nums == bin_num
        axes[1].scatter(coords_mlflow[mask, 0], coords_mlflow[mask, 1], 
                       c=[bin_colors[bin_num]], s=30, alpha=0.6, label=f'Bin {bin_num}')
    axes[1].set_title(f'MLflow VAE + MLflow {model_type}\n(MLflow Models)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'comparison_side_by_side.png')
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig1_path}")
    
    # Figure 2: Overlay comparison
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for bin_num in unique_bins:
        mask = bin_nums == bin_num
        ax.scatter(coords_npz[mask, 0], coords_npz[mask, 1], 
                  c=[bin_colors[bin_num]], s=50, alpha=0.5, marker='o', 
                  label=f'Bin {bin_num} (NPZ+joblib)')
        ax.scatter(coords_mlflow[mask, 0], coords_mlflow[mask, 1], 
                  c=[bin_colors[bin_num]], s=50, alpha=0.5, marker='x')
    
    # Draw lines connecting corresponding points (sample a few)
    sample_indices = np.random.choice(len(results), min(50, len(results)), replace=False)
    for i in sample_indices:
        ax.plot([coords_npz[i, 0], coords_mlflow[i, 0]], 
               [coords_npz[i, 1], coords_mlflow[i, 1]], 
               'gray', alpha=0.2, linewidth=0.5)
    
    ax.set_title(f'Overlay Comparison\n(o = NPZ+joblib, x = MLflow)', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'comparison_overlay.png')
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig2_path}")
    
    # Figure 3: Difference analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Coordinate differences
    coord_diff = np.abs(coords_npz - coords_mlflow)
    euclidean_dists = np.linalg.norm(coords_npz - coords_mlflow, axis=1)
    
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
    fig3_path = os.path.join(output_dir, 'difference_analysis.png')
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
    
    # Choose which model to test: 'pca' or 'neural_pca'
    test_mode = 'neural_pca'  # Change to 'pca' to test traditional PCA
    
    if test_mode == 'pca':
        # Traditional PCA (no scaler needed)
        local_model_path = "../../models/trs_vae/pca_model.joblib"
        scaler_path = None
        is_neural = False
        vae_model_name = "trs_auto_vae2"
        dimred_model_name = "trs_dr_pca_vae2"
        model_type = "PCA"
        output_dir = "comparison_results_pca"
    else:  # neural_pca
        # Neural PCA (needs StandardScaler)
        local_model_path = "../../models/trs_vae/neural_pca_model.pth"
        scaler_path = "../../models/trs_vae/scaler.pkl"
        is_neural = True
        vae_model_name = "trs_auto_vae2"
        dimred_model_name = "trs_dr_neural_pca_vae2"
        model_type = "Neural PCA"
        output_dir = "comparison_results_neural_pca"
    
    print("="*80)
    print(f"FULL PIPELINE COMPARISON WITH VISUALIZATION - {model_type.upper()}")
    print("="*80)
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"VAE Model: {vae_model_name}")
    print(f"DimRed Model: {dimred_model_name}")
    print(f"Mode: {test_mode}")
    print(f"Using Scaler: {scaler_path is not None}")
    
    # Initialize MLflow client
    print(f"\nInitializing MLflowClient...")
    mlflow_client = MLflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print(f"✅ MLflowClient initialized")
    
    # Load data
    print(f"\nLoading data from {data_path}")
    combined_dict = np.load(data_path, allow_pickle=True).item()
    keys = list(combined_dict.keys())
    images = list(combined_dict.values())
    print(f"Total images: {len(images)}")
    
    # Process ALL images
    print("\n" + "="*80)
    print(f"Processing ALL {len(images)} images...")
    print("="*80)
    
    results = []
    
    from tqdm import tqdm
    for idx in tqdm(range(len(images)), desc="Processing images"):
        img = images[idx]
        key = keys[idx]
        bin_num, shot_num = key
        
        # Resize
        img_resized = resize_image(img, target_size=(128, 128))
        
        # Get results from NPZ + joblib
        latent_npz, coords_npz = get_results_from_npz_and_joblib(
            img_resized.copy(), 
            vae_weights_path, 
            local_model_path,
            scaler_path,
            is_neural
        )
        
        # Get results from MLflow
        latent_mlflow, coords_mlflow = get_results_from_mlflow(
            img_resized.copy(), 
            mlflow_client, 
            vae_model_name, 
            dimred_model_name
        )
        
        results.append({
            'idx': idx,
            'key': key,
            'bin_num': bin_num,
            'shot_num': shot_num,
            'latent_npz': latent_npz[0],
            'latent_mlflow': latent_mlflow[0],
            'coords_npz': coords_npz[0],
            'coords_mlflow': coords_mlflow[0],
        })
    
    print(f"\n✅ Processed {len(results)} images")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_comparison(results, output_dir=output_dir, model_type=model_type)
    
    # Calculate final statistics
    coords_npz = np.array([r['coords_npz'] for r in results])
    coords_mlflow = np.array([r['coords_mlflow'] for r in results])
    
    corr = np.corrcoef(coords_npz.flatten(), coords_mlflow.flatten())[0, 1]
    mse = np.mean((coords_npz - coords_mlflow)**2)
    euclidean_mean = np.mean(np.linalg.norm(coords_npz - coords_mlflow, axis=1))
    
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Overall Correlation: {corr:.8f}")
    print(f"Overall MSE: {mse:.8f}")
    print(f"Mean Euclidean Distance: {euclidean_mean:.8f}")
    
    if corr > 0.9999:
        print("\n✅ IDENTICAL: NPZ+joblib and MLflow produce the same results!")
    elif corr > 0.99:
        print("\n✅ VERY SIMILAR: Minor numerical differences only")
    else:
        print("\n❌ DIFFERENT: Significant differences detected")
    
    print("="*80)

if __name__ == "__main__":
    main()