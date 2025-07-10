import time
from typing import Callable, Tuple, Dict, Any, Optional

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_WARMUP_ITERATIONS: int = 100


def cuda_timer(func: Callable) -> Callable:
    """
    Decorator for measuring the execution time of a function on the GPU.
    :param func: Function to measure time
    :return: Wrapped function returning (result, time in ms)
    """
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record(stream=torch.cuda.current_stream())
        result = func(*args, **kwargs)
        end_time.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return result, start_time.elapsed_time(end_time)
    return wrapper


def cpu_timer(func: Callable) -> Callable:
    """
    Decorator for measuring the execution time of a function on the CPU.
    :param func: Function to measure time
    :return: Wrapped function returning (result, time in ms)
    """
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() - start_time
        return result, end_time * 1000
    return wrapper


def gpu_mem_usage(func: Callable) -> Callable:
    """
    Decorator for measuring GPU memory consumption of a function.
    :param func: Function for measuring memory
    :return: Wrapped function returning (result, memory in MB)
    """
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.max_memory_allocated()
        result = func(*args, **kwargs)
        return result, (torch.cuda.max_memory_allocated() - allocated_memory) / 2 ** 20
    return wrapper


def cpu_mem_usage(func: Callable) -> Callable:
    """
    Decorator for measuring CPU memory consumption of a function.
    :param func: Function for measuring memory
    :return: Wrapped function returning (result, memory in MB)
    """
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        allocated_memory = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        return result, (psutil.Process().memory_info().rss - allocated_memory) / 2 ** 20
    return wrapper


def run_test(
    model_wrapper: Callable[[torch.Tensor], Any],
    data_preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    input_shape: Tuple[int, int, int] = (3, 512, 512),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataloader: Optional[DataLoader] = None,
    timer_type: str = 'cuda'
) -> Dict[Tuple[int, Any], float]:
    """
    Runs a model performance test for different batch sizes.
    :param model_wrapper: Model wrapper for making predictions
    :param data_preprocess: Input data preprocessing function
    :param input_shape: Input data dimensions (channels, height, width)
    :param num_runs: Number of runs for each batch size
    :param min_batch_size: Minimum batch size for testing
    :param max_batch_size: Maximum batch size for testing
    :param batch_step: Step to increase batch size
    :param dataloader: Dataloader for real data (if used)
    :param timer_type: Timer type ('cuda' for GPU, otherwise CPU)
    :return: Dictionary with results: {(batch_size, c, h, w): average_time}
    """
    shapes = [(size, *input_shape) for size in range(min_batch_size, max_batch_size + 1, batch_step)]
    results = {}
    timer = cuda_timer if timer_type == 'cuda' else cpu_timer
    for shape in shapes:
        with torch.no_grad():
            for _ in tqdm(range(NUM_WARMUP_ITERATIONS), desc=f'Warmup for shape {shape}'):
                dummy_input = torch.randn(shape, device=timer_type)
                if data_preprocess:
                    dummy_input = data_preprocess(dummy_input)
                model_wrapper(dummy_input)
            times = []
            for _ in range(num_runs):
                for batch in tqdm(dataloader, desc=f'Testing for shape {shape}, iter {_}'):
                    image = batch[0].to(timer_type)
                    if data_preprocess:
                        image = data_preprocess(image)
                    result, time = timer(model_wrapper)(image)
                    times.append(time)
        times = np.array(times)
        times = times[~np.isnan(times)]
        times = times[times < np.percentile(times, 90)]
        times = times[times > np.percentile(times, 10)]
        results[shape] = np.mean(times).item()
    return results
