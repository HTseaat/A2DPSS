# A²DPSS

This repository contains the research prototype for *Asynchronous Dynamic Proactive Secret Sharing in the YOSO Model and Its Applications*. It implements the committee handover studied in Chapter 4 of Wenxuan Yu’s doctoral dissertation.

Each outgoing holder sends a publicly verifiable resharing contribution bound to its existing share commitment. A total-order broadcast protocol lets the incoming committee agree on a set of valid contributions. Each incoming holder then derives a fresh share of the same secret. The implementation builds on the [Continuum](https://github.com/HTseaat/Continuum) experimental framework.

## Run a local check

From the repository root, run the following with Python 3.10 or later.

```sh
python3 -m scripts.run_dag_tob_phase1_smoke
```

This one-wave DAG-TOB simulation uses four local nodes and a deterministic test coin. The JSON output should show `all_ledgers_equal` and `all_ordered_vertices_equal` set to `true`. This check uses the Python standard library.

With pytest installed, the following tests exercise the A²DPSS context and contribution formats, handover metrics, and computation metrics. These tests use synthetic inputs and can run independently of the experiment configuration.

```sh
python3 -m pytest --noconftest -q \
  tests/test_a2dpss_data_model.py \
  tests/test_a2dpss_computation_metrics.py \
  tests/test_a2dpss_handover_metrics.py
```

## Chapter 4 experiments

| Experiment | Comparison | Python entry point |
| --- | --- | --- |
| Single handover | DAG-TOB, ACS-TOB, and Asynchronous Algorand as consensus backends for A²DPSS | `scripts.a2dpss_benchmark` |
| Resharing | A²DPSS and the native RU22 resharing baseline | `scripts.run_yoso_adpss_dist` |
| Concurrent handovers | DAG-TOB and ACS-TOB with an initial batch and a later batch of requests | `scripts.a2dpss_pipeline_benchmark` |

The [experiment guide](docs/experiments.md) describes configuration and the commands for each comparison. For network runs, generate fresh node and threshold-coin keys and supply the endpoints of the participating nodes. The configuration generator uses a deterministic example secret and polynomial for the benchmark workload.

## Implementation

| Path | Contents |
| --- | --- |
| `beaver/a2dpss.py` | Holder contributions, public verification and finalization, and successor share reconstruction |
| `beaver/pvss.py` | Publicly verifiable secret sharing and verification against the previous commitment |
| `beaver/dag_tob.py` | DAG-based total-order broadcast |
| `beaver/async_yoso_tob.py`, `beaver/async_algorand/` | Alternative consensus backends |
| `beaver/baselines/` | Comparison protocols, including native RU22 |
| `native/` | Rust and Go source for the shared cryptographic libraries used by the experiment runners |

The distributed experiments use a Linux environment with PyCryptodome, Charm-Crypto (MNT224), pyzmq, and psutil. The original runtime loads two shared libraries from the repository root. With the Rust toolchain, Go 1.20 or later, a C compiler, and access to the dependencies listed in the Cargo and Go module files, the Linux build commands are

```sh
cargo build --release --locked --manifest-path native/bulletproofs-amcl/Cargo.toml
cp native/bulletproofs-amcl/target/release/libbulletproofs_amcl.so .
(cd native/gnark-crypto/kzg_ped_bls12-381 && \
  CGO_ENABLED=1 go build -buildmode=c-shared -o ../../../kzg_ped_out.so kzg_ped_out.go)
```

The local checks above cover the DAG state machine, public data formats, and measurement code. A distributed protocol run also exercises the native cryptography, node transport, and generated configuration. Third-party components retain their license files and copyright notices under `native/` and in the [third-party notices](THIRD_PARTY.md).
