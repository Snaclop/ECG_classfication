"""Test ROCm GPU execution of Conv1d, learned gating and backward updates."""
import copy
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/setup/windows_gpu_check.json"


def main():
    report = {"python": sys.version, "executable": sys.executable, "status": "failed"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from torch import nn
        report.update(torch=torch.__version__, hip=torch.version.hip, cuda_available=torch.cuda.is_available())
        report["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(json.dumps(report, indent=2), flush=True)
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("ROCm GPU is not available; CPU fallback is not a GPU pass")
        selected = next((i for i, name in enumerate(report["devices"]) if "7600" in name), None)
        if selected is None:
            raise RuntimeError("Target RX 7600M XT not enumerated")
        device = torch.device(f"cuda:{selected}")
        report["selected_gpu"] = report["devices"][selected]
        report["total_memory_bytes"] = torch.cuda.get_device_properties(selected).total_memory
        torch.manual_seed(42)
        torch.set_num_threads(4)

        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                def branch(channels):
                    return nn.Sequential(nn.Conv1d(channels, 16, 7, padding=3), nn.ReLU(), nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
                self.a, self.b = branch(12), branch(24)
                self.gate = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
                self.head = nn.Linear(32, 5)

            def forward(self, raw, encoded):
                a, b = self.a(raw), self.b(encoded)
                weight = self.gate(torch.cat([a, b], dim=1))
                return self.head(weight * a + (1 - weight) * b)

        cpu = Probe()
        gpu = copy.deepcopy(cpu).to(device)
        raw = torch.randn(8, 12, 5000)
        encoded = torch.randint(-1, 2, (8, 24, 5000)).float()
        labels = torch.randint(0, 2, (8, 5)).float()
        x, z, y = raw.to(device), encoded.to(device), labels.to(device)
        print("Checking GPU forward against CPU...", flush=True)
        with torch.no_grad():
            expected = cpu(raw, encoded)
            actual = gpu(x, z).cpu()
        report["cpu_gpu_max_abs_error"] = float((expected - actual).abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        optimizer = torch.optim.Adam(gpu.parameters(), lr=1e-3, foreach=False)
        original = [p.detach().clone() for p in gpu.parameters()]
        losses = []
        started = time.perf_counter()
        for step in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(gpu(x, z), y)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in gpu.parameters()):
                raise RuntimeError("Nonfinite loss or invalid gradient")
            optimizer.step()
            torch.cuda.synchronize(device)
            losses.append(float(loss.detach()))
            print(f"GPU step {step + 1}/20 loss={losses[-1]:.6f}", flush=True)
        if not any(not torch.equal(before, after) for before, after in zip(original, gpu.parameters())):
            raise RuntimeError("Parameters did not update")
        report.update(status="passed", steps=20, batch_size=8, sequence_length=5000,
                      losses=losses, elapsed_seconds=time.perf_counter()-started,
                      peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                      note="Synthetic operator and training smoke test; not project accuracy or long-term stability validation.")
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Result: {report['status']} -> {OUT}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
