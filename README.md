# Ciphertext-Bound Online/Offline One-Time Re-Encryption

Minimal reproducibility code for the experiments in our ICASSP 2027 submission on
ciphertext-bound online/offline one-time re-encryption for attribute-key-free CP-ABE.

## Scope

This repository contains **operation-level cryptographic benchmarks**. It is **not**
the official implementation of Shi et al. and is **not** a complete application-level
end-to-end prototype.

The benchmark code reproduces the cryptographic operation sequences and timing
boundaries used in the paper. In particular, the baseline online path samples a fresh
one-time factor \(z\) and performs the ciphertext-dependent re-encryption operations,
whereas the proposed method prepares independently randomized single-use tokens before
the access-critical online path and consumes one prepared token online.

Unless otherwise stated for a specific benchmark, authentication, network/IPC
transmission, serialization, token-pool lookup, candidate-set enumeration, LSSS
coefficient solving, and payload decryption are outside the timed cryptographic path.

## Repository Contents

| File | Paper result |
|---|---|
| `cpu_benchmark.py` | Fig. 2: CPU operation-level latency |
| `android_benchmark.py` | Fig. 3: Android one-time re-encryption latency |
| `android_controlled_benchmark.py` | Table 1: controlled Android online-path latency |
| `storage_benchmark.py` | Table 2: serialized token storage |

The benchmark scripts are kept as the validated experimental artifacts. The repository
does not intentionally refactor or alter their timing boundaries or cryptographic
semantics.

## Dependencies

The experiments use Python and Charm-Crypto.

Python packages used directly by the scripts include:

- `charm-crypto-framework`
- `numpy`
- `matplotlib`

Charm-Crypto may require native cryptographic/build dependencies depending on the
platform. Follow the current official Charm-Crypto installation documentation:

- Source repository: https://github.com/JHUISI/charm
- Installation instructions: https://github.com/JHUISI/charm/blob/dev/INSTALL

The scripts use the Charm pairing group `SS512` to approximate the 512-bit Type-A
evaluation setting used by the baseline paper. This parameter choice is for experimental
comparability and should not be interpreted as a recommendation for production security.

Install the Python-side dependencies with:

```bash
python -m pip install -r requirements.txt
```

If Charm-Crypto cannot be installed directly on the target platform, install its native
dependencies first according to the official documentation above.

## Experimental Mapping and Settings

### 1. CPU benchmark — Fig. 2

Run:

```bash
python cpu_benchmark.py
```

Main settings:

- pairing group: `SS512`
- policy sizes: \(l \in \{2,4,6,8,10,15,20\}\)
- 10 warm-up batches per metric
- 50 measured batches per metric
- fixed batch sizes:
  - re-encryption: 20
  - receiver-side path: 8
  - full online decryption path: 2
- execution order is counterbalanced within each comparison family
- reported CPU summary statistics trim 10% from each tail
- per-request latency is obtained from each fixed-size batch
- candidate-set enumeration and LSSS coefficient solving are excluded
- token-pool lookup is excluded
- the single-use token-state transition is modeled in memory

The CPU script writes raw measurements, summary statistics, execution-order records,
metadata, and a diagnostic plot.

### 2. Android benchmark — Fig. 3

Run in the Android/Termux environment used for the experiment:

```bash
python android_benchmark.py
```

Main settings:

- pairing group: `SS512`
- policy sizes: \(l \in \{2,4,6,8,10,15,20,30,40,50\}\)
- 10 warm-up runs
- 50 measured runs
- individual executions are timed (not batches)
- all measured Android samples are retained; no trimming
- Baseline / Offline / Ours execution order is rotated across six permutations
- token preparation for the proposed online path is completed before online timing

Outputs:

- `android_fig3_raw.csv`
- `android_fig3_summary.csv`

### 3. Controlled Android benchmark — Table 1

Run in the same Android/Termux environment:

```bash
python android_controlled_benchmark.py
```

Main settings:

- pairing group: `SS512`
- representative policy sizes: \(l \in \{10,30,50\}\)
- 10 warm-up runs
- 50 measured runs
- all measured samples are retained; no trimming
- Baseline/Ours execution order alternates `B->O` / `O->B`
- matched offline/baseline measurements alternate `B->F` / `F->B`
- cooldowns and short inter-measurement gaps are used to mitigate execution-order and
  device-heating effects
- token pools are generated before the corresponding proposed online measurement blocks

Outputs:

- `android_thermal_raw.csv`
- `android_thermal_summary.csv`
- `android_thermal_metadata.json`

Thermal readings are auxiliary metadata. Availability depends on Android sysfs access;
missing readings do not alter latency timing.

### 4. Storage benchmark — Table 2

Run:

```bash
python storage_benchmark.py
```

Main settings:

- pairing group: `SS512`
- policy sizes: \(l \in \{10,20,30,40,50\}\)
- token-pool size: `q = 5`
- 50 repetitions per policy size
- serialized size includes:
  - one scalar \(z_j\)
  - \(2l\) precomputed group elements
  - one-byte state flag
- Python object/list/dictionary overhead is excluded

Output:

- `storage_results.csv`

## Timing Boundaries

The paper reports cryptographic operation-level latency, not complete application
latency. The benchmark is designed so that request/input construction and precomputation
that belongs to the offline phase are outside the proposed online timing region.

The proposed online benchmark models the required single-use token state transition in
memory. A production implementation should enforce the same check-and-set atomically in
protected storage; this repository does not claim to benchmark a production database,
TEE, or persistent transaction layer.

## Reproducibility Notes

Absolute latency can vary across processors, Android devices, OS/kernel versions,
Python/Charm builds, power states, and background load. The benchmark scripts fix the
cryptographic operation sequence, policy sizes, measurement counts, and comparison order
used for the paper.

For strongest provenance, keep the benchmark files unchanged when reproducing the
reported experiment and record the environment together with generated raw/summary
outputs.

## Baseline

The benchmark operation sequence follows the receiver-side one-time re-encryption and
subsequent cryptographic processing described in:

X. Shi et al., “Toward Forward-Secure End-to-End Data Sharing: An Attribute-Key-Free
CP-ABE Scheme,” *Proc. IEEE ICASSP*, 2025.

The copyrighted baseline paper itself is not redistributed in this repository.

## License

The benchmark code in this repository is released under the MIT License. Third-party
software, including Charm-Crypto, remains subject to its own license.
