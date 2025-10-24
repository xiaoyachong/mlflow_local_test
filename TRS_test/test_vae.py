"""
Simple test: Compare latent features from NPZ model vs MLflow model
Uses SAME preprocessing as vae_wrapper.py for fair comparison
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

def get_latent_from_npz(img, weights_path):
    """
    Get latent features using NPZ weights with WRAPPER-STYLE preprocessing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ConvVAE(latent_dim=256, image_size=(128, 128))
    weights_npz = np.load(weights_path)
    state_dict = {key: torch.tensor(weights_npz[key]) for key in weights_npz.files}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    
    # Preprocess EXACTLY like wrapper
    tensor = preprocess_like_wrapper(img, image_size=(128, 128))
    
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
    # Configuration
    data_path = "shot_averages/combined_all_bin_averages.npy"
    weights_path = "../../models/trs_vae/vae_model_128_weights.npz"
    model_name = "trs_auto_vae2"
    
    print("="*80)
    print("SIMPLE COMPARISON: NPZ (with wrapper preprocessing) vs MLflow")
    print("="*80)
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"Model name: {model_name}")
    print(f"Project root: {project_root}")
    print(f"\nNote: Both use WRAPPER-STYLE preprocessing (per-image min/max)")
    
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
    
    # Test 5 samples
    test_indices = [0, 50, 100, 200, 400]
    
    print("\n" + "="*80)
    print("Testing samples...")
    print("="*80)
    
    results = []
    
    for idx in test_indices:
        img = images[idx]
        key = keys[idx]
        
        print(f"\n{'='*80}")
        print(f"Sample {idx} (Key={key})")
        print(f"{'='*80}")
        
        # Resize
        img_resized = resize_image(img, target_size=(128, 128))
        print(f"Original image shape: {img.shape}")
        print(f"Resized image shape: {img_resized.shape}, range: [{img_resized.min():.6f}, {img_resized.max():.6f}]")
        
        # Get latents from NPZ (with wrapper-style preprocessing)
        print("\n--- NPZ Model (with wrapper preprocessing) ---")
        latent_npz = get_latent_from_npz(img_resized.copy(), weights_path)
        
        # Get latents from MLflow
        print("\n--- MLflow Model (via MLflowClient) ---")
        latent_mlflow = get_latent_from_mlflow(img_resized.copy(), mlflow_client, model_name)
        
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
            'key': key,
            'corr': corr,
            'mse': mse,
            'max_diff': max_diff,
            'status': status
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Index':<10} {'Key':<15} {'Correlation':<15} {'MSE':<15} {'Status'}")
    print("-" * 80)
    for r in results:
        print(f"{r['idx']:<10} {str(r['key']):<15} {r['corr']:<15.8f} {r['mse']:<15.8f} {r['status']}")
    
    avg_corr = np.mean([r['corr'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    
    print("-" * 80)
    print(f"{'Average':<10} {'':<15} {avg_corr:<15.8f} {avg_mse:<15.8f}")
    
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