# Observability Module

This module provides comprehensive observability infrastructure for the CHEAP RAG system, including tracing, logging, error tracking, and performance monitoring.

## Components

### 1. Tracing (`tracing.py`)

OpenTelemetry-based distributed tracing for end-to-end visibility.

**Basic Usage:**
```python
from src.observability import init_tracing, trace_operation, trace_function

# Initialize tracing
init_tracing(
    service_name="cheap-rag",
    environment="production",
    enable_console=True,
    enable_otlp=True,
    otlp_endpoint="http://localhost:4317"
)

# Use context manager
with trace_operation("embed_query", attributes={"query_length": len(query)}):
    embedding = model.encode(query)

# Use decorator
@trace_function("retrieve_artifacts", include_args=True)
def retrieve(query: str, top_k: int = 5):
    # Function automatically traced
    return search(query, top_k)
```

**Features:**
- Automatic duration tracking
- Exception recording in spans
- Support for both sync and async functions
- Multiple exporters (Console, OTLP)
- Configurable span attributes

### 2. Logging (`logging.py`)

Structured logging with correlation ID support.

**Basic Usage:**
```python
from src.observability import init_logging, StructuredLogger, set_correlation_id

# Initialize logging
init_logging(
    level="INFO",
    format_type="json",
    log_file="logs/cheap-rag.log",
    enable_console=True
)

# Set correlation ID for request tracking
set_correlation_id("request-123")

# Create logger
logger = StructuredLogger("my_module")

# Log with structured fields
logger.info("Processing query", query_length=42, user_id="user-456")
logger.error("Failed to retrieve", error_type="TimeoutError", retry_count=3)
```

**Features:**
- Correlation ID propagation via context variables
- JSON and text output formats
- File rotation (100MB) and compression
- Structured key-value logging
- Colored console output for development

### 3. Error Tracking (`error_tracking.py`)

Centralized error tracking with aggregation and rate calculation.

**Basic Usage:**
```python
from src.observability import get_error_tracker, record_error, get_correlation_id

# Record an error
try:
    risky_operation()
except Exception as e:
    record_error("retrieval", e, correlation_id=get_correlation_id())
    raise

# Query error statistics
tracker = get_error_tracker()
error_rate = tracker.get_error_rate("retrieval")  # errors per minute
errors_by_type = tracker.get_error_counts_by_type()
recent_errors = tracker.get_recent_errors(component="retrieval", limit=10)

# Get summary
summary = tracker.get_summary()
```

**Features:**
- Thread-safe error aggregation
- Error rate calculation (errors per minute)
- Time-windowed retention (default: 5 minutes)
- Error grouping by component and type
- Recent error queries with filtering

### 4. Performance Profiling (`performance.py`)

Detailed performance metrics including latency percentiles and memory usage.

**Basic Usage:**
```python
from src.observability import get_performance_monitor
import time

monitor = get_performance_monitor()

# Record operation latency
start = time.perf_counter()
result = expensive_operation()
duration_ms = (time.perf_counter() - start) * 1000
monitor.record_operation("expensive_op", duration_ms)

# Get statistics
stats = monitor.profiler.get_stats("expensive_op")
print(f"Mean: {stats.mean_ms:.2f}ms")
print(f"P95: {stats.p95_ms:.2f}ms")
print(f"P99: {stats.p99_ms:.2f}ms")
print(f"Throughput: {stats.throughput_per_sec:.2f} ops/sec")

# Get memory usage
memory = monitor.memory.get_memory_usage()
print(f"RSS: {memory['rss_mb']:.2f}MB")

# Get GPU memory (if CUDA available)
gpu_memory = monitor.memory.get_gpu_memory()
if gpu_memory:
    print(f"GPU Allocated: {gpu_memory['allocated_mb']:.2f}MB")

# Get comprehensive summary
summary = monitor.get_summary()
```

**Features:**
- Latency percentiles (p50, p95, p99)
- Mean, median, min, max, standard deviation
- Throughput calculation (ops/sec)
- CPU memory tracking (RSS, VMS)
- GPU memory tracking (CUDA)
- Time-windowed statistics

## Integration Examples

### API Endpoint with Full Observability

```python
from fastapi import FastAPI, Request
from src.observability import (
    init_tracing,
    init_logging,
    trace_operation,
    set_correlation_id,
    StructuredLogger,
    record_error,
    get_performance_monitor,
)
import time
import uuid

app = FastAPI()

# Initialize observability
init_tracing(service_name="cheap-rag-api", enable_console=True)
init_logging(level="INFO", format_type="json", enable_console=True)

logger = StructuredLogger("api")
monitor = get_performance_monitor()

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    # Set correlation ID
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)

    # Log request
    logger.info("Query received", correlation_id=correlation_id, query=request.query)

    # Trace the operation
    with trace_operation("query_endpoint", attributes={"query_length": len(request.query)}):
        start = time.perf_counter()

        try:
            # Process query
            result = await process_query(request)

            # Record performance
            duration_ms = (time.perf_counter() - start) * 1000
            monitor.record_operation("query_endpoint", duration_ms)

            logger.info("Query completed", correlation_id=correlation_id, duration_ms=duration_ms)
            return result

        except Exception as e:
            # Record error
            record_error("api.query", e, correlation_id=correlation_id)
            logger.error("Query failed", correlation_id=correlation_id, error=str(e))
            raise
```

### Embedding Service with Tracing

```python
from src.observability import trace_function, get_performance_monitor
import time

class EmbeddingService:
    @trace_function("embed_query", include_args=True)
    def embed_query(self, query: str) -> list[float]:
        start = time.perf_counter()

        # Generate embedding
        embedding = self.model.encode(query)

        # Record performance
        duration_ms = (time.perf_counter() - start) * 1000
        get_performance_monitor().record_operation(
            "embed_query",
            duration_ms,
            metadata={"query_length": len(query)}
        )

        return embedding

    @trace_function("embed_batch", include_result=True)
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()

        # Generate embeddings
        embeddings = self.model.encode(texts, batch_size=32)

        # Record performance
        duration_ms = (time.perf_counter() - start) * 1000
        get_performance_monitor().record_operation(
            "embed_batch",
            duration_ms,
            metadata={"batch_size": len(texts)}
        )

        return embeddings
```

## Configuration

### Tracing Configuration

```python
from src.observability import TracingConfig

config = TracingConfig(
    enabled=True,
    service_name="cheap-rag",
    environment="production",
    console_export=False,
    otlp_export=True,
    otlp_endpoint="http://jaeger:4317"
)

config.initialize()
```

### Logging Configuration

```python
from src.observability import LoggingConfig

config = LoggingConfig(
    level="INFO",
    format_type="json",
    log_file="logs/cheap-rag.log",
    enable_console=True
)

config.initialize()
```

## Best Practices

1. **Always set correlation IDs** for request tracking
2. **Use structured logging** with key-value pairs instead of string formatting
3. **Record errors** immediately when they occur
4. **Trace expensive operations** (embed, retrieve, generate)
5. **Monitor performance** of critical paths
6. **Use appropriate log levels** (DEBUG for development, INFO for production)
7. **Include context** in trace attributes (query length, batch size, etc.)
8. **Aggregate metrics** periodically to identify trends

## Monitoring Dashboard Queries

When using OTLP export to observability platforms (Jaeger, Prometheus, etc.):

**Query duration by operation:**
```promql
histogram_quantile(0.95, sum(rate(operation_duration_ms_bucket[5m])) by (operation, le))
```

**Error rate by component:**
```promql
rate(errors_total[5m]) by (component)
```

**Memory usage trend:**
```promql
avg(memory_rss_mb) by (instance)
```

## Dependencies

- `opentelemetry-api` - Tracing API
- `opentelemetry-sdk` - Tracing SDK
- `opentelemetry-exporter-otlp` - OTLP exporter
- `loguru` - Structured logging
- `psutil` - System monitoring

## See Also

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/python/)
- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Phase 2 Completion Report](../../../cheap-planning/PHASE2_COMPLETE.md)
