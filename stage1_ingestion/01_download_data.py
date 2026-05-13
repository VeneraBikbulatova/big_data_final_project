import os
import shutil
import sys

try:
    import kagglehub
except ImportError:
    print("[ERROR] Library 'kagglehub' not found. Install it with: pip3 install kagglehub")
    sys.exit(1)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest_dir = os.path.join(project_root, "data", "raw")
    target_file = os.path.join(dest_dir, "2019-Oct.csv")

    if os.path.exists(target_file):
        size_mb = os.path.getsize(target_file) / (1024 * 1024)
        print(f"[INFO] Dataset already exists: {target_file} ({size_mb:.2f} MB). Skipping download.")
        return

    print("[INFO] Starting dataset download from Kaggle...")
    try:
        tmp_path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")
        
        os.makedirs(dest_dir, exist_ok=True)
        
        found = False
        for root, dirs, files in os.walk(tmp_path):
            if "2019-Oct.csv" in files:
                src_file = os.path.join(root, "2019-Oct.csv")
                print(f"[INFO] Moving file to {dest_dir}...")
                shutil.move(src_file, target_file)
                found = True
                break
        
        if found:
            print("[INFO] Success! Data is ready in data/raw/")
        else:
            print("[ERROR] File 2019-Oct.csv not found in downloaded archive.")
            
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
