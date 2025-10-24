"""
Simple test: Compare latent features from NPZ model vs MLflow model
SMI Dataset Version - Tests with PNG files
Uses SAME preprocessing as vae_wrapper.py for fair comparison
"""
import numpy as np
import torch
import mlflow
import os
import sys
import glob
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

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
    This mimics the wrapper's preprocessing pipeline
    """
    # Squeeze
    img_array = np.squeeze(img_array)
    
    # Check and handle input dtype (EXACTLY like wrapper)
    if img_array.dtype == np.uint8:
        img_array_uint8 = img_array
    else:
        # Convert using per-image min-max scaling (WRAPPER BEHAVIOR)
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
    
    # Convert to PIL Image (EXACTLY like wrapper)
    pil_image = Image.fromarray(img_array_uint8)
    
    # Apply transforms (EXACTLY like wrapper)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),  # Does nothing but wrapper has it
    ])
    
    tensor = transform(pil_image)
    
    return tensor

def get_latent_from_npz(img, weights_path, latent_dim=512, image_size=(512, 512)):
    """
    Get latent features using NPZ weights with WRAPPER-STYLE preprocessing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size)
    weights_npz = np.load(weights_path)
    state_dict = {key: torch.tensor(weights_npz[key]) for key in weights_npz.files}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    
    # Preprocess EXACTLY like wrapper
    tensor = preprocess_like_wrapper(img, image_size=image_size)
    
    print(f"  Local preprocessing output:")
    print(f"    Tensor shape: {tensor.shape}")
    print(f"    Tensor range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print(f"    Tensor mean: {tensor.mean():.6f}")
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(device)
    
    # Get latent
    with torch.no_grad():
        _, mu, _ = model(tensor)
        latent = mu.cpu().numpy()
    
    return latent

def get_latent_from_mlflow(img, mlflow_client, model_name):
    """Get latent features using MLflow model via MLflowClient"""
    print(f"  Loading model via MLflowClient: {model_name}")
    model = mlflow_client.load_model(model_name)
    
    print(f"  MLflow input:")
    print(f"    Input shape: {img.shape}")
    print(f"    Input dtype: {img.dtype}")
    print(f"    Input range: [{img.min():.6f}, {img.max():.6f}]")
    
    # Call MLflow model (wrapper will do its own preprocessing)
    result = model.predict(img)
    latent = result["latent_features"]
    
    return latent

def main():
    # SMI Configuration
    png_directory = "./733_april24_2025_png/Nafion_p25_30_run_1"
    weights_path = "../../models/vae/vae_model_512_weights.npz"
    model_name = "smi_auto_vae"
    latent_dim = 512
    image_size = (512, 512)
    
    print("="*80)
    print("SIMPLE COMPARISON: NPZ (with wrapper preprocessing) vs MLflow - SMI Dataset")
    print("="*80)
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"Model name: {model_name}")
    print(f"Latent dim: {latent_dim}")
    print(f"Image size: {image_size}")
    print(f"PNG directory: {png_directory}")
    print(f"Project root: {project_root}")
    print(f"\nNote: Both use WRAPPER-STYLE preprocessing (per-image min/max)")
    
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
    
    # Test first 5 samples (or fewer if less than 5 files)
    test_indices = list(range(min(5, len(png_files))))
    
    print("\n" + "="*80)
    print("Testing samples...")
    print("="*80)
    
    results = []
    
    for idx in test_indices:
        png_path = png_files[idx]
        png_filename = os.path.basename(png_path)
        
        print(f"\n{'='*80}")
        print(f"Sample {idx} - {png_filename}")
        print(f"{'='*80}")
        
        # Load PNG image
        img = load_png_image(png_path, target_size=image_size)
        print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image range: [{img.min():.6f}, {img.max():.6f}]")
        
        # Get latents from NPZ (with wrapper-style preprocessing)
        print("\n--- NPZ Model (with wrapper preprocessing) ---")
        latent_npz = get_latent_from_npz(img.copy(), weights_path, latent_dim, image_size)
        
        # Get latents from MLflow
        print("\n--- MLflow Model (via MLflowClient) ---")
        latent_mlflow = get_latent_from_mlflow(img.copy(), mlflow_client, model_name)
        
        # Compare
        corr = np.corrcoef(latent_npz.flatten(), latent_mlflow.flatten())[0, 1]
        mse = np.mean((latent_npz - latent_mlflow)**2)
        max_diff = np.max(np.abs(latent_npz - latent_mlflow))
        
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"NPZ latent (first 10):    {latent_npz[0,:10]}")
        print(f"MLflow latent (first 10): {latent_mlflow[0,:10]}")
        print(f"Difference (first 10):    {latent_mlflow[0,:10] - latent_npz[0,:10]}")
        print(f"\nStatistics:")
        print(f"  Correlation: {corr:.8f}")
        print(f"  MSE:         {mse:.8f}")
        print(f"  Max diff:    {max_diff:.8f}")
        
        if corr > 0.9999:
            print(f"  ✅ IDENTICAL! Models produce the same results.")
            status = "✅ IDENTICAL"
        elif corr > 0.99:
            print(f"  ✅ VERY SIMILAR! Minor numerical differences only.")
            status = "✅ SIMILAR"
        else:
            print(f"  ❌ DIFFERENT! Models produce different results.")
            status = "❌ DIFFERENT"
        
        results.append({
            'idx': idx,
            'filename': png_filename,
            'corr': corr,
            'mse': mse,
            'max_diff': max_diff,
            'status': status
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Index':<10} {'Filename':<40} {'Correlation':<15} {'MSE':<15} {'Status'}")
    print("-" * 100)
    for r in results:
        print(f"{r['idx']:<10} {r['filename']:<40} {r['corr']:<15.8f} {r['mse']:<15.8f} {r['status']}")
    
    avg_corr = np.mean([r['corr'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    
    print("-" * 100)
    print(f"{'Average':<10} {'':<40} {avg_corr:<15.8f} {avg_mse:<15.8f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if avg_corr > 0.9999:
        print("✅ SUCCESS: NPZ and MLflow models are IDENTICAL!")
        print("   Both use the same preprocessing and produce identical results.")
        print("\n   This means:")
        print("   - MLflow wrapper is working correctly")
        print("   - Model is loaded properly in MLflow")
        print("   - The issue is that WRAPPER preprocessing ≠ TRAINING preprocessing")
        print("\n   SOLUTION:")
        print("   1. Save normalization params (data_min, data_max) during training")
        print("   2. Update vae_wrapper.py to load and use these params")
        print("   3. Re-upload model to MLflow with normalization params")
    elif avg_corr > 0.99:
        print("✅ GOOD: NPZ and MLflow models are VERY SIMILAR!")
        print("   Minor numerical differences likely due to:")
        print("   - Float precision differences")
        print("   - CPU vs GPU")
        print("   These differences are negligible.")
    else:
        print("❌ PROBLEM: NPZ and MLflow models are DIFFERENT!")
        print("   Possible causes:")
        print("   1. Model weights not loaded correctly in MLflow")
        print("   2. Model not in eval() mode in MLflow wrapper")
        print("   3. Different torch/numpy versions")
        print("   4. MLflow wrapper has additional preprocessing we missed")
        print("\n   Next steps:")
        print("   - Check vae_wrapper.py load_context() method")
        print("   - Verify model.eval() is called")
        print("   - Check state_dict loading (strict=True or False)")
    
    print("="*80)

if __name__ == "__main__":
    main()