# NYA-9: Not Your Address

**A Unique Physics-Inspired Cryptographic Hash Function**

Version 3.0 | May 2026

---

## Overview

**NYA-9** ("Not Your Address") is a 256-bit cryptographic hash function that uses a custom **hyperbolic 4D physics engine** as its core mixing mechanism.

Instead of traditional ARX (Add-Rotate-XOR) operations, NYA-9 derives its non-linearity and uniqueness from:
- Hyperbolic geometry and 4D pentagonal lattices
- Gyroscopic plane tuning (120° radial angles)
- Tetrahedral incidence dynamics (109.5°)
- Lynchpin hyperbolic multiplication (`1 ⊗ 1 ≈ 2`)
- Pressure field functionals (mean curvature + Gaussian curvature)

This creates a highly unique computational path that is extremely difficult to reverse-engineer or replicate without the complete **Lynchpin Physics** system.

---

## Philosophy

> "A hash should not merely scramble bits — it should make your address unknowable in the universe of possible messages."

NYA-9 treats input data as vibrational states within a hyperbolic lattice and transmutes them through gyroscopic pressure fields and incidence-based resonance.

---

## Features

- **Output**: 256-bit (64 hexadecimal characters)
- **Strong Avalanche Effect**: One-bit change produces completely different output
- **Deterministic**: Same input → always same output
- **High Uniqueness**: Deep integration with custom physics engine
- **Computational Bottleneck as Feature**: Makes brute-force and cryptanalysis more expensive
- **Pure Python + NumPy** (depends on Lynchpin Physics library)

---

## Installation & Usage

### Requirements
- Python 3.8+
- `numpy`
- `sympy`
- `scipy`
- `lynchpin_physics.py` (included)

### Basic Usage

```python
from nya9 import NYA9

nya = NYA9(rounds=9)

# Hash a message
print(nya.hash("hello world"))
print(nya.hexdigest("The quick brown fox jumps over the lazy dog"))
