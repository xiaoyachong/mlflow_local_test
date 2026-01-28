"""
Simple test: Compare latent features AND PCA results from NPZ+joblib vs MLflow models
SMI Dataset Version - Tests with PNG files
WITH VISUALIZATION
"""
import numpy as np
import torch
import mlflow
import os
import sys
import glob
from PIL import Image
from torchvision import transforms
import joblib
import matplotlib.pyplot as plt

# Add authentication import
from als_computing import enable_auth

# Enable authentication
enable_auth()

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mlex_latent_explorer/'))
sys.path.insert(0, project_root)

# Now you can import from src
from src.utils.mlflow_utils import MLflowClient
from models.vae_202602.vae import ConvVAE

# MLflow setup
MLFLOW_TRACKING_URI = "https://mlflow.computing.als.lbl.gov"

def load_png_image(filepath, target_size=(512, 512)):
    """
    Load a PNG image and convert to grayscale numpy array
    """
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img)
    return img_array

def preprocess_like_wrapper(img_array, image_size=(512, 512)):
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

def get_results_from_npz_and_joblib(img, vae_weights_path, pca_path, scaler_path=None, 
                                     is_neural=False, latent_dim=512, image_size=(512, 512)):
    """
    Get both latent features and PCA/Neural PCA coords using NPZ VAE + joblib model
    
    Args:
        img: Input image
        vae_weights_path: Path to VAE weights
        pca_path: Path to PCA or Neural PCA model
        scaler_path: Path to StandardScaler (only used for neural PCA)
        is_neural: True for neural PCA, False for traditional PCA
        latent_dim: Latent dimension
        image_size: Image size tuple
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VAE model
    vae_model = ConvVAE(latent_dim=latent_dim, image_size=image_size)
    weights_npz = np.load(vae_weights_path)
    state_dict = {key: torch.tensor(weights_npz[key]) for key in weights_npz.files}
    vae_model.load_state_dict(state_dict, strict=True)
    vae_model.eval()
    vae_model = vae_model.to(device)
    
    # Preprocess
    tensor = preprocess_like_wrapper(img, image_size=image_size)
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
        neural_model = SimpleDimRedApproximator(input_dim=latent_dim)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            neural_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            neural_model.load_state_dict(checkpoint)
        
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

def plot_comparison(results, output_dir='comparison_results_smi', model_type='PCA'):
    """
    Create comprehensive visualizations comparing NPZ+joblib vs MLflow
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    coords_npz = np.array([r['coords_npz'] for r in results])
    coords_mlflow = np.array([r['coords_mlflow'] for r in results])
    filenames = [r['filename'] for r in results]
    
    # Figure 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot NPZ+joblib results
    axes[0].scatter(coords_npz[:, 0], coords_npz[:, 1], 
                   c='blue', s=50, alpha=0.6, label='All samples')
    axes[0].set_title(f'NPZ VAE + joblib {model_type}\n(Local Models)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MLflow results
    axes[1].scatter(coords_mlflow[:, 0], coords_mlflow[:, 1], 
                   c='red', s=50, alpha=0.6, label='All samples')
    axes[1].set_title(f'MLflow VAE + MLflow {model_type}\n(MLflow Models)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'comparison_side_by_side.png')
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig1_path}")
    
    # Figure 2: Overlay comparison
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.scatter(coords_npz[:, 0], coords_npz[:, 1], 
              c='blue', s=80, alpha=0.6, marker='o', label='NPZ+joblib')
    ax.scatter(coords_mlflow[:, 0], coords_mlflow[:, 1], 
              c='red', s=80, alpha=0.6, marker='x', label='MLflow')
    
    # Draw lines connecting corresponding points
    for i in range(len(results)):
        ax.plot([coords_npz[i, 0], coords_mlflow[i, 0]], 
               [coords_npz[i, 1], coords_mlflow[i, 1]], 
               'gray', alpha=0.3, linewidth=1)
    
    ax.set_title(f'Overlay Comparison\n(o = NPZ+joblib, x = MLflow)', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.legend(fontsize=11)
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
    axes[0, 0].hist(coord_diff[:, 0], bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(coord_diff[:, 0]), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(coord_diff[:, 0]):.6f}')
    axes[0, 0].set_title('PC1 Absolute Difference', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Absolute Difference', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of PC2 differences
    axes[0, 1].hist(coord_diff[:, 1], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(coord_diff[:, 1]), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(coord_diff[:, 1]):.6f}')
    axes[0, 1].set_title('PC2 Absolute Difference', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Absolute Difference', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of Euclidean distances
    axes[1, 0].hist(euclidean_dists, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(euclidean_dists), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(euclidean_dists):.6f}')
    axes[1, 0].set_title('Euclidean Distance', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Distance', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bar plot: Distance per sample
    indices = list(range(len(results)))
    axes[1, 1].bar(indices, euclidean_dists, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Euclidean Distance per Sample', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel('Euclidean Distance', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
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
    print(f"\nCoordinate Differences:")
    print(f"  PC1 - Mean: {np.mean(coord_diff[:, 0]):.6f}, Max: {np.max(coord_diff[:, 0]):.6f}")
    print(f"  PC2 - Mean: {np.mean(coord_diff[:, 1]):.6f}, Max: {np.max(coord_diff[:, 1]):.6f}")
    print(f"  Euclidean - Mean: {np.mean(euclidean_dists):.6f}, Max: {np.max(euclidean_dists):.6f}")
    print(f"\nFiles saved in: {output_dir}/")

def main():
    # SMI Configuration
    png_directory = "./733_april24_2025_png/Nafion_p25_30_run_1"
    vae_weights_path = "../../mlex_latent_explorer/models/vae_202602/vae_model_512_weights.npz"
    
    # Choose which model to test: 'umap' or 'neural_umap'
    test_mode = 'umap'  # Change to 'neural_umap' to test neural UMAP
    
    if test_mode == 'umap':
        # Traditional UMAP (no scaler needed)
        local_model_path = "../../mlex_latent_explorer/models/vae_202602/vae_joblib_test.joblib"
        scaler_path = None
        is_neural = False
        vae_model_name = "smi_auto_vae"
        dimred_model_name = "smi_dr_umap_vae"
        model_type = "UMAP"
        output_dir = "comparison_results_smi_umap"
    else:  # neural_umap
        # Neural UMAP (needs StandardScaler)
        local_model_path = "../../mlex_latent_explorer/models/vae_202602/umap_approximator.pth"
        scaler_path = "../../mlex_latent_explorer/models/vae_202602/scaler.pkl"
        is_neural = True
        vae_model_name = "smi_auto_vae"
        dimred_model_name = "smi_dr_neural_umap_vae"
        model_type = "Neural UMAP"
        output_dir = "comparison_results_smi_neural_umap"
    
    latent_dim = 512
    image_size = (512, 512)
    
    print("="*80)
    print(f"FULL PIPELINE COMPARISON WITH VISUALIZATION - SMI {model_type.upper()}")
    print("="*80)
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"VAE Model: {vae_model_name}")
    print(f"DimRed Model: {dimred_model_name}")
    print(f"Mode: {test_mode}")
    print(f"Using Scaler: {scaler_path is not None}")
    print(f"Latent dim: {latent_dim}")
    print(f"Image size: {image_size}")
    print(f"PNG directory: {png_directory}")
    
    # Initialize MLflow client
    print(f"\nInitializing MLflowClient...")
    mlflow_client = MLflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print(f"✅ MLflowClient initialized")
    
    # Load PNG files
    print(f"\nLoading PNG files from {png_directory}")
    png_files = sorted(glob.glob(os.path.join(png_directory, "*.png")))
    
    if len(png_files) == 0:
        print(f"❌ No PNG files found in {png_directory}")
        return
    
    print(f"Total PNG files found: {len(png_files)}")
    
    # Process images (limit to reasonable number for testing)
    max_samples = len(png_files)
    print(f"\nProcessing {max_samples} samples...")
    
    print("\n" + "="*80)
    print(f"Processing {max_samples} images...")
    print("="*80)
    
    results = []
    
    from tqdm import tqdm
    for idx in tqdm(range(max_samples), desc="Processing images"):
        png_path = png_files[idx]
        png_filename = os.path.basename(png_path)
        
        # Load PNG image
        img = load_png_image(png_path, target_size=image_size)
        
        # Get results from NPZ + joblib
        latent_npz, coords_npz = get_results_from_npz_and_joblib(
            img.copy(), 
            vae_weights_path, 
            local_model_path,
            scaler_path,
            is_neural,
            latent_dim,
            image_size
        )
        
        # Get results from MLflow
        latent_mlflow, coords_mlflow = get_results_from_mlflow(
            img.copy(), 
            mlflow_client, 
            vae_model_name, 
            dimred_model_name
        )
        
        results.append({
            'idx': idx,
            'filename': png_filename,
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