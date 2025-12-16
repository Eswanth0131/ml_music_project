import os
import torch

SEARCH_DIR = "." 

print(f"{'MODEL':<20} | {'PARAMS (N)':<15} | {'BEST VAL LOSS (L)':<20}")
print("-" * 65)

for root, dirs, files in os.walk(SEARCH_DIR):
    for file in files:
        if file == 'ckpt.pt':
            full_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            
            try:
                checkpoint = torch.load(full_path, map_location='cpu')
                
                val_loss = checkpoint.get('best_val_loss', None)
                
                if val_loss is None:
                    val_loss = checkpoint.get('val_loss', float('nan'))

                if 'model' in checkpoint:
                    n_params = sum(p.numel() for p in checkpoint['model'].values())
                else:
                    n_params = 0

                print(f"{folder_name:<20} | {n_params:<15,} | {val_loss:.4f}")
                
            except Exception as e:
                print(f"{folder_name:<20} | {'ERROR READING FILE':<15} | {e}")

print("-" * 65)
