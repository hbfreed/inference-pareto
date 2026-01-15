# Inference Pareto: The Goal

## The Question

At the latency of a deep-narrow network (80 layers, 576 hidden, ~321M params like Baguettotron), how much wider and shallower can we go to get the **same latency** but with **many more parameters**?

## What We Found

**The result is even stronger than expected!** Shallow-wide models can't even be slowed down enough to match the deep model - even at maximum width (before OOM), they're still faster:

| Layers | Hidden | Total Time | Decode | Params | vs Baseline |
|--------|--------|------------|--------|--------|-------------|
| 80     | 576    | 5.51s      | 53.3ms | 0.32B  | 1.0x        |
| 60     | 4096   | 4.37s      | 41.7ms | 10.81B | **33.7x**   |
| 40     | 5120   | 3.65s      | 34.9ms | 11.33B | **35.3x**   |
| 20     | 7168   | 3.27s      | 30.8ms | 11.27B | **35.1x**   |
| 10     | 8192   | 2.15s      | 20.3ms | 7.58B  | **23.6x**   |

**Key insight:** A 40-layer model with **35x more parameters** is still **34% faster** than the 80-layer baseline!

## Real-World Comparison: Baguettotron vs Gemma 3 12B

To validate this isn't just a synthetic result, we compared Baguettotron's architecture against Gemma 3 12B:

| Model        | Layers | Hidden | Params  | Total Time | Decode   |
|--------------|--------|--------|---------|------------|----------|
| Baguettotron | 80     | 576    | 0.32B   | 5.65s      | 54.3ms   |
| Gemma 3 12B  | 48     | 3840   | 10.87B  | 3.64s      | 35.2ms   |

**Gemma 3 12B has 34x more parameters but is 1.55x faster!**

The extreme depth of Baguettotron (80 layers) creates a sequential bottleneck that dominates inference time, despite having far fewer parameters.

## Why This Matters

For on-device inference, latency matters more than raw parameter count. If you have a 6-second latency budget, you can fit WAY more model capacity with a shallow-wide architecture than a deep-narrow one.

## Metrics to Report

1. **Total response time** (prefill + 100 tokens) - the real UX metric
2. **Decode latency** (ms/token) - traditional benchmark metric
3. **Prefill latency** - matters for every conversation turn
4. **Parameter count** - model capacity

## The Experiment

1. Measure baseline: 80L @ 576h total response time
2. Binary search: For each depth [60, 40, 20, 10], find hidden size where total time ≈ baseline
3. Report all metrics for each time-matched configuration
