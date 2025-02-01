import time
import psutil
import torch
import numpy as np
from typing import Dict, Type
from ..models.base_model import BaseHangmanModel
import GPUtil

class ModelBenchmark:
    """Benchmarking utility for model performance"""
    
    def __init__(self, model_class: Type[BaseHangmanModel], device: torch.device):
        self.model_class = model_class
        self.device = device
        self.metrics = {}
        
    def measure_memory(self) -> Dict[str, float]:
        """Measure memory usage"""
        memory_stats = {}
        
        # CPU Memory
        process = psutil.Process()
        memory_stats['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        
        # GPU Memory if available
        if torch.cuda.is_available():
            memory_stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For Apple Silicon, just record 0 since we can't measure MPS memory
            memory_stats['gpu_memory_mb'] = 0
            
        return memory_stats
    
    def measure_inference_time(self, model: BaseHangmanModel, input_batch: Dict[str, torch.Tensor], num_runs: int = 100) -> Dict[str, float]:
        """Measure inference time statistics"""
        times = []
        model.eval()
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_batch)
            
            # Actual measurements
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_batch)
                end = time.perf_counter()
                times.append(end - start)
        
        return {
            'inference_mean_ms': np.mean(times) * 1000,
            'inference_std_ms': np.std(times) * 1000,
            'inference_median_ms': np.median(times) * 1000
        }
    
    def run_benchmark(self, batch_size: int = 32, sequence_length: int = 20) -> Dict[str, float]:
        """Run comprehensive benchmark"""
        # Initialize model
        model = self.model_class().to(self.device)
        
        # Create dummy batch
        dummy_batch = {
            'word_state': torch.randn(batch_size, sequence_length, 28).to(self.device),
            'alphabet_state': torch.zeros(batch_size, 26).to(self.device),
            'word_length': torch.ones(batch_size).long().to(self.device) * sequence_length,
            'target_distribution': torch.randn(batch_size, 26).softmax(dim=1).to(self.device)
        }
        
        # Measure initial memory
        initial_memory = self.measure_memory()
        
        # Measure inference time
        inference_metrics = self.measure_inference_time(model, dummy_batch)
        
        # Measure memory after inference
        final_memory = self.measure_memory()
        
        # Calculate memory impact
        memory_impact = {
            f'memory_impact_{k}': final_memory[k] - initial_memory[k]
            for k in initial_memory.keys()
        }
        
        # Combine all metrics
        self.metrics = {
            'model_class': self.model_class.__name__,
            'device': str(self.device),
            **inference_metrics,
            **memory_impact
        }
        
        return self.metrics 