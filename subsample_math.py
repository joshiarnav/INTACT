import os
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

# Define paths
source_dir = "MATH/train"
target_dir = "MATH_subsample"

# Remove target directory if it exists and create a new one
shutil.rmtree(target_dir, ignore_errors=True)

# Find all JSON files recursively
json_files = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.json') and not file.startswith('.'):
            json_files.append(os.path.join(root, file))

print(f"Found {len(json_files)} total JSON files")

# Randomly sample 1000 files
if len(json_files) > 1000:
    selected_files = random.sample(json_files, 1000)
else:
    selected_files = json_files
    print(f"Warning: Only found {len(json_files)} JSON files")

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

print(f"Successfully copied {len(selected_files)} files to {target_dir}/ maintaining directory structure")

# Verify the number of files
total_files = sum(len(files) for _, _, files in os.walk(target_dir))
print(f"Final number of files in {target_dir}: {total_files}")

# Print directory structure statistics
categories = {}
for root, _, files in os.walk(target_dir):
    category = os.path.basename(root)
    if category and files:  # Skip empty directories and root
        categories[category] = len(files)

print("\nFiles per category:")
for category, count in sorted(categories.items()):
    print(f"{category}: {count} files")
