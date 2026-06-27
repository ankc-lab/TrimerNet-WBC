import os
import csv

# Veri setlerinin olduğu ana klasör yolu
root_dir = 'data'
output_dir = 'manifests'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_csv(dataset_name):
    path = os.path.join(root_dir, dataset_name)
    csv_file = os.path.join(output_dir, f'{dataset_name}_manifest.csv')
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'subset'])
        
        # Klasör yapısını tara (örn: data/Raabin/Train/Neutrophil/...)
        for subset in ['train','validation','test']:
            subset_path = os.path.join(path, subset)
            if os.path.exists(subset_path):
                for label in os.listdir(subset_path):
                    label_path = os.path.join(subset_path, label)
                    if os.path.isdir(label_path):
                        for file in os.listdir(label_path):
                            writer.writerow([file, label, subset])
    print(f"{dataset_name} için liste oluşturuldu: {csv_file}")

# Dataset isimlerini listele
datasets = ['Raabin', 'Raabin_Diffusion', 'LISC', 'PBC'] 
for ds in datasets:
    generate_csv(ds)