"""Computational-complexity profiling for Question 5: parameter count,
FLOPs (via forward-hook MAC counting on Conv/Linear layers), wall-clock
inference latency, and peak memory -- the profile Question 1 flags as
largely absent from the rPPG literature outside TDA-Phys.
"""

import time
import resource
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, example_inputs: tuple) -> int:
    """Approximate multiply-accumulate count via forward hooks on Conv3d
    and Linear layers (the two op types dominating PICA-Net's cost)."""
    model_device = next(model.parameters()).device
    example_inputs = tuple(x.to(model_device) for x in example_inputs)

    total_macs = [0]
    hooks = []

    def conv3d_hook(module, inp, out):
        out_shape = out.shape  # (B, Cout, T, H, W)
        b, cout, t, h, w = out_shape
        cin_per_group = module.in_channels // module.groups
        kt, kh, kw = module.kernel_size
        macs = b * cout * t * h * w * cin_per_group * kt * kh * kw
        total_macs[0] += macs

    def linear_hook(module, inp, out):
        b = inp[0].shape[0] if inp[0].dim() > 1 else 1
        n_elems = inp[0].numel() // inp[0].shape[-1]
        macs = n_elems * module.in_features * module.out_features
        total_macs[0] += macs

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            hooks.append(m.register_forward_hook(conv3d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(*example_inputs)

    for h in hooks:
        h.remove()

    return total_macs[0] * 2  # MACs -> FLOPs


def measure_inference_latency(model: nn.Module, example_inputs: tuple, device: str = "cpu",
                               n_warmup: int = 5, n_runs: int = 30) -> dict:
    model = model.to(device).eval()
    example_inputs = tuple(x.to(device) for x in example_inputs)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(*example_inputs)
        if device == "mps":
            torch.mps.synchronize()

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(*example_inputs)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)

    times = torch.tensor(times)
    return {
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "min_ms": float(times.min() * 1000),
        "max_ms": float(times.max() * 1000),
    }


def measure_peak_memory_mb() -> float:
    """Peak resident set size for the current process (macOS: bytes; Linux: KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def full_complexity_report(model: nn.Module, example_inputs: tuple, device: str = "cpu") -> dict:
    n_params = count_parameters(model)
    flops = estimate_flops(model, example_inputs)
    latency = measure_inference_latency(model, example_inputs, device=device)
    peak_mem_mb = measure_peak_memory_mb()
    return {
        "n_parameters": n_params,
        "flops_per_window": flops,
        "gflops_per_window": flops / 1e9,
        "latency_ms": latency,
        "peak_process_memory_mb": peak_mem_mb,
        "device": device,
    }
