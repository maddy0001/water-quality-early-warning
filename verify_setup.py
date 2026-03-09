"""
============================================================================
Setup Verification Script
Run this to check if your environment is properly configured
============================================================================
"""

import sys
from pathlib import Path

print("=" * 70)
print("ENVIRONMENT VERIFICATION".center(70))
print("=" * 70)
print()

# Check Python version
print("1. Checking Python version...")
python_version = sys.version_info
if python_version >= (3, 8):
    print(f"   ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ✗ Python {python_version.major}.{python_version.minor} (Need 3.8+)")
    sys.exit(1)

# Check required packages
print("\n2. Checking required packages...")
required_packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'scikit-learn',
    'tensorflow': 'TensorFlow',
    'xgboost': 'XGBoost',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'shap': 'SHAP',
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} (missing)")
        missing_packages.append(name)

if missing_packages:
    print(f"\n   Missing packages: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Check GPU availability
print("\n3. Checking GPU availability...")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✓ GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"      - GPU {i}: {gpu.name}")
    else:
        print("   ℹ No GPU detected (will use CPU)")
except Exception as e:
    print(f"   ⚠ Could not check GPU: {e}")

# Check datasets
print("\n4. Checking datasets...")
data_dir = Path(__file__).parent / "data" / "raw"

datasets = {
    'water_potability.csv': 'Classification dataset',
    'water_dataset.mat': 'Forecasting dataset'
}

for filename, description in datasets.items():
    filepath = data_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"   ✓ {description}: {size_mb:.2f} MB")
    else:
        print(f"   ✗ {description} (missing)")
        print(f"      Expected location: {filepath}")

# Check directory structure
print("\n5. Checking directory structure...")
required_dirs = [
    'data/raw',
    'data/processed',
    'models/classification',
    'models/forecasting',
    'models/anomaly',
    'outputs/logs',
    'outputs/plots',
    'outputs/metrics',
    'src/data',
    'src/models',
    'pipelines'
]

base_dir = Path(__file__).parent
missing_dirs = []
for dir_path in required_dirs:
    full_path = base_dir / dir_path
    if full_path.exists():
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ✗ {dir_path} (missing)")
        missing_dirs.append(dir_path)

if missing_dirs:
    print(f"\n   Creating missing directories...")
    for dir_path in missing_dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    print("   ✓ Directories created")

# Final summary
print("\n" + "=" * 70)
if not missing_packages and all((data_dir / f).exists() for f in datasets.keys()):
    print("✓ SETUP COMPLETE - Ready to run main.py".center(70))
else:
    print("⚠ SETUP INCOMPLETE - Please fix the issues above".center(70))
print("=" * 70)
print()

# Test import of config
print("6. Testing configuration file...")
try:
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from config import print_config
    print("   ✓ Configuration loaded successfully")
    print()
    print_config()
except Exception as e:
    print(f"   ✗ Configuration error: {e}")

print("\n" + "=" * 70)
print("Verification complete!".center(70))
print("=" * 70)
