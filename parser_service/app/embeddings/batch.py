from typing import List, Optional, Dict, Any, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import heapq
import threading
import time
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchPriority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class DynamicBatchConfig:
    """Configuration for dynamic batch processing"""

    min_batch_size: int = 8
    max_batch_size: int = 64
    target_tokens_per_batch: int = 8192  # Target number of tokens per batch
    max_concurrent_batches: int = 4
    memory_threshold: float = 0.85  # Maximum memory usage threshold (85%)
    adjustment_factor: float = 0.1  # Batch size adjustment factor


class BatchItem(Generic[T]):
    """A single item in the batch queue"""

    def __init__(self, data: T, priority: BatchPriority, timestamp: float):
        self.data = data
        self.priority = priority
        self.timestamp = timestamp

    def __lt__(self, other: "BatchItem[T]") -> bool:
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


class BatchProgress:
    """Track progress of batch processing"""

    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def update(self, items_processed: int = 1) -> None:
        with self._lock:
            self.processed_items += items_processed

    @property
    def progress(self) -> float:
        return (
            (self.processed_items / self.total_items) * 100
            if self.total_items > 0
            else 0
        )

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class DynamicBatchProcessor(Generic[T]):
    """Process items in dynamically sized batches with priority and memory optimization"""

    def __init__(self, config: Optional[DynamicBatchConfig] = None):
        self.config = config or DynamicBatchConfig()
        self.queue: PriorityQueue[BatchItem[T]] = PriorityQueue()
        self.current_batch_size = self.config.min_batch_size
        self._lock = threading.Lock()
        self._memory_monitor = psutil.Process()

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return self._memory_monitor.memory_percent() / 100.0

    def _adjust_batch_size(self, processing_time: float, memory_usage: float) -> None:
        """Dynamically adjust batch size based on performance metrics"""
        with self._lock:
            if memory_usage > self.config.memory_threshold:
                # Reduce batch size if memory usage is too high
                reduction = max(
                    self.config.adjustment_factor * 2,
                    (memory_usage - self.config.memory_threshold),
                )
                self.current_batch_size = max(
                    self.config.min_batch_size,
                    int(self.current_batch_size * (1 - reduction)),
                )
            elif (
                processing_time < 0.1
                and memory_usage < self.config.memory_threshold * 0.8
            ):
                # Increase batch size if processing is fast and memory usage is low
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    int(self.current_batch_size * (1 + self.config.adjustment_factor)),
                )

    def add_item(self, item: T, priority: BatchPriority = BatchPriority.MEDIUM) -> None:
        """Add an item to the processing queue"""
        batch_item = BatchItem(item, priority, time.time())
        self.queue.put(batch_item)

    def _process_batch(
        self,
        batch: List[T],
        processor: Callable[[List[T]], Any],
        progress: BatchProgress,
    ) -> Any:
        """Process a single batch of items"""
        try:
            start_time = time.time()
            result = processor(batch)
            processing_time = time.time() - start_time

            # Update progress
            progress.update(len(batch))

            # Monitor and adjust batch size
            memory_usage = self._get_memory_usage()
            self._adjust_batch_size(processing_time, memory_usage)

            return result
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise

    def process_queue(self, processor: Callable[[List[T]], Any]) -> List[Any]:
        """Process all items in the queue using parallel processing"""
        if self.queue.empty():
            return []

        results = []
        current_batch: List[T] = []
        total_items = self.queue.qsize()
        progress = BatchProgress(total_items)

        with ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_batches
        ) as executor:
            futures = []

            while not self.queue.empty() or current_batch:
                # Fill current batch
                while (
                    len(current_batch) < self.current_batch_size
                    and not self.queue.empty()
                ):
                    item = self.queue.get()
                    current_batch.append(item.data)

                # Process batch if it's full or queue is empty
                if current_batch and (
                    len(current_batch) >= self.current_batch_size or self.queue.empty()
                ):
                    batch_to_process = current_batch
                    current_batch = []

                    future = executor.submit(
                        self._process_batch, batch_to_process, processor, progress
                    )
                    futures.append(future)

                # Collect results from completed futures
                for completed in as_completed(futures):
                    try:
                        result = completed.result()
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        raise

        logger.info(
            f"Processed {progress.processed_items} items in {progress.elapsed_time:.2f} seconds"
        )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            "queue_size": self.queue.qsize(),
            "current_batch_size": self.current_batch_size,
            "memory_usage": self._get_memory_usage(),
            "max_batch_size": self.config.max_batch_size,
            "min_batch_size": self.config.min_batch_size,
        }
