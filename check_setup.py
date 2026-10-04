import sys, time, platform
import torch, numpy, chess, zstandard

print("Python :", sys.version.split()[0], "on", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS built    :", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

board = chess.Board()
print("python-chess :", board.legal_moves.count(), "legal opening moves (should be 20)")

if not torch.backends.mps.is_available():
    print("\n!! Metal GPU not available - stop here and fix this.")
    sys.exit(1)


def bench(device, n=30, size=2048):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    if device == "mps":
        torch.mps.synchronize()
    start = time.time()
    for _ in range(n):
        c = a @ b
    if device == "mps":
        torch.mps.synchronize()
    return time.time() - start


cpu = bench("cpu")
gpu = bench("mps")
print(f"\nCPU: {cpu:.2f}s   GPU: {gpu:.2f}s   speedup: {cpu/gpu:.1f}x")
