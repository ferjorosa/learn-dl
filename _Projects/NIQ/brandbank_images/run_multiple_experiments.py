# run_multiple_experiments.py
from run_image_experiment import train_and_test_model


def main():
    # List of configuration file paths for multiple experiments
    experiment_configs = [
        # 5000 images, 10 classes
        "./config/brandbank_5000_examples_10_groups/flava_image_encoder.yaml",
        "./config/brandbank_5000_examples_10_groups/resnet_50.yaml",
        "./config/brandbank_5000_examples_10_groups/vision_transformer.yaml",
        # 20000 images, 10 classes
        "./config/brandbank_20000_examples_10_groups/flava_image_encoder.yaml",
        "./config/brandbank_20000_examples_10_groups/resnet_50.yaml",
        "./config/brandbank_20000_examples_10_groups/vision_transformer.yaml",
        # 50000 images, 100 classes
        "./config/brandbank_50000_examples_100_groups/flava_image_encoder.yaml",
        "./config/brandbank_50000_examples_100_groups/resnet_50.yaml",
        "./config/brandbank_50000_examples_100_groups/vision_transformer.yaml",
        # 262000 images, 260 classes
        "./config/brandbank_all_examples_all_groups/flava_image_encoder.yaml",
        "./config/brandbank_all_examples_all_groups/resnet_50.yaml",
        "./config/brandbank_all_examples_all_groups/vision_transformer.yaml",
    ]

    for config_path in experiment_configs:
        print(f"Running experiment with config: {config_path}")
        train_and_test_model(config_path)
        print(f"Experiment with config {config_path} completed.\n")


if __name__ == "__main__":
    main()
