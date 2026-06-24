import os

def count_images(dataset_path):
    classes = ['Normal', 'ASD', 'Alzheimers', 'Tumors', 'Schizophrenia']
    total_images = 0
    
    print(f"Scanning dataset in: {dataset_path}\n")
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_path):
            print(f"Directory not found: {cls_path}")
            continue
            
        # Count files (ignoring hidden files like .DS_Store)
        files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f)) and not f.startswith('.')]
        count = len(files)
        total_images += count
        
        print(f"Class '{cls}': {count} images")
        
    print(f"\nTotal images found: {total_images}")
    if total_images > 0:
        print("Dataset loaded successfully.")
    else:
        print("No images found. Please download your dataset and place the files into the Normal/ and ASD/ folders.")

if __name__ == "__main__":
    # Get the project root folder (one level up from 'preprocessing')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'datasets')
    
    count_images(dataset_dir)
