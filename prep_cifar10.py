import os
import shutil
import numpy as np
import torchvision
import csv
import argparse

# Configuration
DEFAULT_NUM_CLIENTS = 20       
DEFAULT_ALPHA = 0.1
fedscale_home = os.environ.get("FEDSCALE_HOME")
if not fedscale_home:
    raise ValueError("The environment variable $FEDSCALE_HOME is not set!")
MAPPING_DIR = os.path.join(fedscale_home, "benchmark/dataset/cifar10/client_data_mapping")
# Temporary location just to read labels. 
TEMP_DOWNLOAD_DIR = "./temp_cifar10_raw" 

def get_cifar10_targets():
    """
    Download CIFAR-10 to a temp folder strictly to read labels (targets).
    """
    print(f"Downloading CIFAR-10 to temporary location: {TEMP_DOWNLOAD_DIR}...")
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    
    # Download to temp dir. 
    train_data = torchvision.datasets.CIFAR10(root=TEMP_DOWNLOAD_DIR, train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(root=TEMP_DOWNLOAD_DIR, train=False, download=True)
    
    return train_data, test_data

def partition_data(train_data, num_clients, alpha):
    """
    Partition the dataset indices among clients using Dirichlet distribution.
    """
    print(f"Partitioning data (Clients={num_clients}, Dirichlet alpha={alpha})...")
    np.random.seed(0)
    
    targets = np.array(train_data.targets)
    num_classes = 10
    
    client_idcs = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < len(targets) / num_clients) for p, idx_j in zip(proportions, client_idcs)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        class_splits = np.split(idx_k, proportions)
        for i, split in enumerate(class_splits):
            client_idcs[i] += split.tolist()

    return client_idcs

def save_mappings(client_idcs, train_data, test_data):
    """
    Save the mapping files required by FedScale.
    """
    print(f"Saving mapping files to {MAPPING_DIR}...")
    
    os.makedirs(MAPPING_DIR, exist_ok=True)
    
    # Map sample_index -> client_id
    sample_to_client = [-1] * len(train_data)
    for client_id, sample_indices in enumerate(client_idcs):
        for sample_idx in sample_indices:
            sample_to_client[sample_idx] = client_id
            
    # Write train.csv
    with open(f"{MAPPING_DIR}/train.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "sample_path", "label_name"])
        
        for sample_idx, client_id in enumerate(sample_to_client):
            if client_id != -1:
                label = train_data.targets[sample_idx] 
                writer.writerow([client_id, sample_idx, label])

    # Write test.csv
    with open(f"{MAPPING_DIR}/test.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "sample_path", "label_name"])
        for idx in range(len(test_data)):
            writer.writerow([-1, idx, test_data.targets[idx]])

def cleanup():
    """Remove the temp data to save space."""
    if os.path.exists(TEMP_DOWNLOAD_DIR):
        print(f"Cleaning up temporary data in {TEMP_DOWNLOAD_DIR}...")
        shutil.rmtree(TEMP_DOWNLOAD_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Non-IID partitions for CIFAR-10")
    parser.add_argument('--num_clients', type=int, default=DEFAULT_NUM_CLIENTS, 
                        help='Number of clients to simulate')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, 
                        help='Dirichlet distribution alpha (smaller = more non-IID)')
    
    args = parser.parse_args()

    try:
        train_set, test_set = get_cifar10_targets()
        
        client_indices = partition_data(train_set, args.num_clients, args.alpha)
        
        save_mappings(client_indices, train_set, test_set)
        
        print(f"Client mappings generated for {args.num_clients} clients with alpha={args.alpha} in {MAPPING_DIR}")
        
    finally:
        cleanup()