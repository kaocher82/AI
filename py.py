# main.py

from data_preprocessing import preprocess_data
from model_training import train_model
from model_inference import make_predictions
from evaluation import evaluate_model
from visualization import plot_results

class AIManager:
    def __init__(self, config_path="config.yaml"):
        # Load configuration settings
        self.config = self._load_config(config_path)
        self.data = None
        self.model = None
        self.results = None

    def _load_config(self, config_path):
        # Example: Load configuration from a YAML file
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def run_pipeline(self):
        print("Starting AI pipeline...")

        # 1. Data Preprocessing
        print("Preprocessing data...")
        self.data = preprocess_data(self.config['data_path'], self.config['preprocessing_params'])

        # 2. Model Training
        print("Training model...")
        self.model = train_model(self.data, self.config['model_params'])

        # 3. Model Inference (if applicable)
        if self.config.get('run_inference', False):
            print("Making predictions...")
            self.results = make_predictions(self.model, self.data['test_set'])

        # 4. Model Evaluation
        print("Evaluating model...")
        metrics = evaluate_model(self.model, self.data['validation_set'])
        print(f"Evaluation Metrics: {metrics}")

        # 5. Visualization
        print("Generating visualizations...")
        plot_results(self.results, self.config['visualization_params'])

        print("AI pipeline completed.")

if __name__ == "__main__":
    # Create an instance of the AIManager
    ai_manager = AIManager(config_path="config.yaml")

    # Run the entire AI pipeline
    ai_manager.run_pipeline()
