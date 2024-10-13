# run_multiple_experiments.py
from run_multimodal_3_losses_experiment import train_and_test_model


def main():
    # List of configuration file paths for multiple experiments
    experiment_configs = [
        # # 5000 descriptions, 10 classes
        # "./config/brandbank_5000_examples_10_groups/flava_multimodal_encoder_3_losses.yaml",
        # # 20000 descriptions, 10 classes
        # "./config/brandbank_20000_examples_10_groups/flava_multimodal_encoder_3_losses.yaml",
        # # 50000 descriptions, 100 classes
        # "./config/brandbank_50000_examples_100_groups/flava_multimodal_encoder_3_losses.yaml",
        # 262000 descriptions, 260 classes
        "./config/brandbank_all_examples_all_groups/flava_multimodal_encoder_3_losses.yaml",
    ]

    for config_path in experiment_configs:
        print(f"Running experiment with config: {config_path}")
        train_and_test_model(config_path)
        print(f"Experiment with config {config_path} completed.\n")


if __name__ == "__main__":
    main()
