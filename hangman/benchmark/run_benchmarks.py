import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import List, Type
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from ..models.transformer_model import TransformerModel
from ..models.cnn_lstm_model import CNNLSTMModel
from ..models.gnn_model import GNNModel
from ..models.mlp_attention_model import MLPAttentionModel
from ..utils.benchmarking import ModelBenchmark
from ..data.word_processor import WordProcessor
from ..data.simulator import HangmanSimulator

class ModelBenchmarker:
    def __init__(
        self,
        model_classes: List[Type],
        train_words: List[str],
        val_words: List[str],
        device: str = None,
        batch_size: int = 64,
        max_epochs: int = 10
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                     else "cuda" if torch.cuda.is_available() 
                                     else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_classes = model_classes
        self.train_words = train_words
        self.val_words = val_words
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        # Initialize data processing
        self.word_processor = WordProcessor(train_words)
        self.simulator = HangmanSimulator(self.word_processor)
        
        # Results storage
        self.benchmark_results = {}
        self.training_results = {}
        
    def run_performance_benchmarks(self):
        """Run performance benchmarks for all models"""
        print("\nRunning Performance Benchmarks...")
        
        for model_class in self.model_classes:
            print(f"\nBenchmarking {model_class.__name__}...")
            benchmark = ModelBenchmark(model_class, self.device)
            results = benchmark.run_benchmark(batch_size=self.batch_size)
            self.benchmark_results[model_class.__name__] = results
            
            print(f"Results for {model_class.__name__}:")
            print(f"- Inference time (mean): {results['inference_mean_ms']:.2f}ms")
            print(f"- Memory impact: {results['memory_impact_cpu_memory_mb']:.2f}MB")
            if 'memory_impact_gpu_memory_mb' in results:
                print(f"- GPU Memory impact: {results['memory_impact_gpu_memory_mb']:.2f}MB")
                
    def train_and_evaluate(self, project_name: str = "hangman-benchmarks"):
        """Train and evaluate all models"""
        print("\nTraining and Evaluating Models...")
        
        for model_class in self.model_classes:
            model_name = model_class.__name__
            print(f"\nTraining {model_name}...")
            
            # Initialize wandb logger
            wandb_logger = WandbLogger(
                project=project_name,
                name=model_name,
                log_model=True
            )
            
            # Initialize callbacks
            callbacks = [
                ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=f'checkpoints/{model_name}',
                    filename='{epoch}-{val_loss:.2f}',
                    save_top_k=1,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    mode='min'
                )
            ]
            
            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator='auto',
                devices=1,
                logger=wandb_logger,
                callbacks=callbacks,
                enable_progress_bar=True
            )
            
            # Initialize model with data
            model = model_class()
            model.train_words = self.train_words  # Add training data
            model.val_words = self.val_words      # Add validation data
            
            # Train model
            trainer.fit(model)
            
            # Store results
            self.training_results[model_name] = {
                'best_val_loss': trainer.callback_metrics.get('val_loss', float('inf')).item(),
                'best_val_accuracy': trainer.callback_metrics.get('val_accuracy', 0.0).item(),
                'epochs_trained': trainer.current_epoch + 1
            }
            
            wandb.finish()
            
    def plot_results(self, save_dir: str = 'benchmark_results'):
        """Plot benchmark results"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Performance metrics plot
        perf_data = pd.DataFrame(self.benchmark_results).T
        plt.figure(figsize=(15, 10))
        
        # Inference time
        plt.subplot(2, 2, 1)
        sns.barplot(data=perf_data, y=perf_data.index, x='inference_mean_ms')
        plt.title('Inference Time (ms)')
        
        # Memory usage
        plt.subplot(2, 2, 2)
        sns.barplot(data=perf_data, y=perf_data.index, x='memory_impact_cpu_memory_mb')
        plt.title('CPU Memory Impact (MB)')
        
        # Training results plot
        train_data = pd.DataFrame(self.training_results).T
        
        # Validation loss
        plt.subplot(2, 2, 3)
        sns.barplot(data=train_data, y=train_data.index, x='best_val_loss')
        plt.title('Best Validation Loss')
        
        # Validation accuracy
        plt.subplot(2, 2, 4)
        sns.barplot(data=train_data, y=train_data.index, x='best_val_accuracy')
        plt.title('Best Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/benchmark_results.png')
        plt.close()
        
        # Save numerical results
        results_df = pd.DataFrame({
            'Model': list(self.benchmark_results.keys()),
            'Inference Time (ms)': [r['inference_mean_ms'] for r in self.benchmark_results.values()],
            'CPU Memory (MB)': [r['memory_impact_cpu_memory_mb'] for r in self.benchmark_results.values()],
            'Best Val Loss': [self.training_results[m]['best_val_loss'] for m in self.benchmark_results.keys()],
            'Best Val Accuracy': [self.training_results[m]['best_val_accuracy'] for m in self.benchmark_results.keys()],
            'Epochs Trained': [self.training_results[m]['epochs_trained'] for m in self.benchmark_results.keys()]
        })
        results_df.to_csv(f'{save_dir}/benchmark_results.csv', index=False)
        
        return results_df

def main():
    # Update path to look in parent directories
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            'words_250000_train.txt')
    
    # Load words and take a small subset
    with open(file_path, 'r') as f:
        all_words = [word.strip().lower() for word in f.readlines()]
        # Take only 1000 words for quick testing
        words = all_words[:1000]  
    
    # Split into train and validation
    train_size = int(0.8 * len(words))
    train_words = words[:train_size]
    val_words = words[train_size:]
    
    # Define models to benchmark (ordered by speed based on initial results)
    models = [
        MLPAttentionModel,    # Fastest (7.46ms)
        CNNLSTMModel,         # Second (13.65ms)
        GNNModel,             # Third (14.12ms)
        TransformerModel      # Slowest (28.73ms)
    ]
    
    # Run benchmarks with smaller epochs and batch size
    benchmarker = ModelBenchmarker(
        model_classes=models,
        train_words=train_words,
        val_words=val_words,
        batch_size=32,        # Reduced from 64
        max_epochs=3          # Reduced from 10
    )
    
    # Run performance benchmarks
    benchmarker.run_performance_benchmarks()
    
    # Train and evaluate models
    benchmarker.train_and_evaluate()
    
    # Plot and save results
    results = benchmarker.plot_results()
    print("\nFinal Results:")
    print(results)

if __name__ == "__main__":
    main() 