import os
import random
import shutil
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Define paths
source_dir = "MATH/train"
target_dir = "MATH_subsample_uniform"
target_per_category = 72  # 1000 / 7 ≈ 142.857, rounded up to 143

# Remove target directory if it exists and create a new one
shutil.rmtree(target_dir, ignore_errors=True)

# Find all JSON files recursively and organize by category
category_files = defaultdict(list)
for root, dirs, files in os.walk(source_dir):
    category = os.path.basename(root)
    if category:  # Skip root directory
        for file in files:
            if file.endswith('.json') and not file.startswith('.'):
                category_files[category].append(os.path.join(root, file))

# Print original distribution
print("Original distribution:")
for category, files in sorted(category_files.items()):
    print(f"{category}: {len(files)} files")

# Sample files uniformly from each category
selected_files = []
for category, files in sorted(category_files.items()):
    num_to_sample = min(target_per_category, len(files))
    if num_to_sample < target_per_category:
        print(f"\nWarning: Category {category} only has {len(files)} files, sampling all of them")
    sampled = random.sample(files, num_to_sample)
    selected_files.extend(sampled)

# Copy selected files to target directory while preserving structure
for source_path in selected_files:
    # Get the relative path from source_dir
    rel_path = os.path.relpath(source_path, source_dir)
    # Construct target path
    target_path = os.path.join(target_dir, rel_path)
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    # Copy the file
    shutil.copy2(source_path, target_path)

print(f"\nSuccessfully copied {len(selected_files)} files to {target_dir}/ maintaining directory structure")

# Print final distribution
print("\nFinal distribution:")
final_distribution = defaultdict(int)
for root, _, files in os.walk(target_dir):
    category = os.path.basename(root)
    if category:  # Skip root directory
        final_distribution[category] = len([f for f in files if f.endswith('.json')])

for category, count in sorted(final_distribution.items()):
    print(f"{category}: {count} files")
