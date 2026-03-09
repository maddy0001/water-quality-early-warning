# 🔍 JSON DIAGNOSTICS - RUN THIS FIRST!

import json
from pathlib import Path

print("="*70)
print("  CHECKING JSON FILES STRUCTURE")
print("="*70)

# Check anomaly_metrics.json
print("\n1️⃣ ANOMALY METRICS:")
print("-" * 70)
anomaly_path = Path('outputs/metrics/anomaly_metrics.json')
if anomaly_path.exists():
    with open(anomaly_path, 'r') as f:
        data = json.load(f)
    print(f"✓ File exists")
    print(f"Keys: {list(data.keys())}")
    print(f"Content: {json.dumps(data, indent=2)[:500]}...")
else:
    print("✗ FILE NOT FOUND!")

# Check feature_importance_summary.json
print("\n2️⃣ FEATURE IMPORTANCE:")
print("-" * 70)
feature_path = Path('outputs/metrics/feature_importance_summary.json')
if feature_path.exists():
    with open(feature_path, 'r') as f:
        data = json.load(f)
    print(f"✓ File exists")
    print(f"Keys: {list(data.keys())}")
    if 'random_forest' in data:
        print(f"✓ Has 'random_forest' key")
        print(f"Top 3 features: {list(data['random_forest'].items())[:3]}")
    else:
        print(f"✗ NO 'random_forest' KEY")
        print(f"Content: {json.dumps(data, indent=2)[:300]}")
else:
    print("✗ FILE NOT FOUND!")

# Check evaluation_summary.json
print("\n3️⃣ EVALUATION SUMMARY:")
print("-" * 70)
eval_path = Path('outputs/metrics/evaluation_summary.json')
if eval_path.exists():
    with open(eval_path, 'r') as f:
        data = json.load(f)
    print(f"✓ File exists")
    print(f"Keys: {list(data.keys())}")
    if 'metrics_summary' in data:
        print(f"✓ Has 'metrics_summary'")
        print(f"Models: {list(data['metrics_summary'].keys())}")
        if 'Random Forest' in data['metrics_summary']:
            rf = data['metrics_summary']['Random Forest']
            print(f"RF Accuracy: {rf.get('accuracy', 'N/A')}")
    else:
        print(f"Content: {json.dumps(data, indent=2)[:300]}")
else:
    print("✗ FILE NOT FOUND!")

print("\n" + "="*70)
print("  COPY THIS OUTPUT AND SEND TO ME!")
print("="*70)
