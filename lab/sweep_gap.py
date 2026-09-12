import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.harness import experiment

# Frozen reference for comparison: dev mean 0.676, worst 0.474.
for gap in (0., .01, .02, .03, .05, .10, 1.01):
    experiment(f'gap={gap:.2f}', {'gap': gap})
