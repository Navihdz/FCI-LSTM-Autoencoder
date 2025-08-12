from modules.config import Config

def main():
    print("Testing access to configuration file inside Docker...")
    print(f"Working Directory: {Config.working_directory}")
    print(f"EZFIO Path: {Config.ezfio_path}")
    print(f"QP binary path: {Config.qpsh_path}")
    print(f"Max Iterations: {Config.max_iterations}")
    print(f"Num Epochs: {Config.num_epochs}")
    print(f"Learning Rate: {Config.learning_rate}")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Embedding Dim: {Config.embedding_dim}")
    print(f"Reference Energy: {Config.FCI_energy}")

if __name__ == "__main__":
    with open("salida_testing.out", "w") as f:
        import sys
        sys.stdout = f
        main()

