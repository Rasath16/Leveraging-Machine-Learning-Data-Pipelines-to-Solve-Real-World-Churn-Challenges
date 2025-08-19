import yaml
import logging
from data_pipeline.data_pipeline import DataPipeline

if __name__ == "__main__":
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log"),
            logging.StreamHandler()
        ]
    )

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline = DataPipeline(config)
    pipeline.run()
    print("✅ Data pipeline executed successfully. Artifacts saved to 'artifacts_pipeline/'.")
