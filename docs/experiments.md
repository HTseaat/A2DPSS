# Chapter 4 experiments

The examples below run from the repository root using the Python interpreter and two native libraries described in the README. Use a fresh trial identifier and output directory for each repetition. Running the logical nodes on one host exercises the protocol and reporting path. To measure the WAN configurations in the thesis, place the logical nodes on separate hosts as described below.

## Single handover with three agreement backends

Generate a five-node configuration with independent parameters for the resharing, agreement and common-coin committees. This small configuration satisfies the Algorand backend's AVSS thresholds. Generated files contain node-specific secrets and remain local to each trial.

```sh
python -m scripts.run_key_gen_atcaas \
  --M 5 --n-r 3 --t-r 1 --n-a 5 --t-a 1 --m-a 5 \
  --coin-c 5 --coin-t 1 --coin-k 2 --out-dir conf/exp-five

python -m scripts.a2dpss_benchmark prepare \
  --config-dir conf/exp-five --trial-id single-001 --view live \
  --output runs/single-001

for backend in dag acs algorand
do
  python -m scripts.a2dpss_benchmark run \
    --bundle runs/single-001 --protocol "$backend" --nodes 0 1 2 3 4 \
    --output runs/single-001-results || break
done

python -m scripts.a2dpss_benchmark compare \
  --bundle runs/single-001 --dag-results runs/single-001-results \
  --acs-results runs/single-001-results --algorand-results runs/single-001-results \
  --output runs/single-001-comparison.json
```

The live trial includes online generation of holder contributions. For a separate trial, `--view paired` prepares identical contribution payloads for all three backends before timing. The report includes maximum successor completion latency, ADPSS CPU time and transmitted bytes. Compare backends within the same view.

## Resharing with A²DPSS and native RU22

Prepare the same initial workload for the two protocols. The cluster example runs the nodes locally. The command uses the 30-node example parameters from the experiment sources. Choose a host with sufficient capacity or distribute these logical nodes across machines.

```sh
python -m scripts.run_yoso_adpss_dist prepare \
  --cluster conf/local-cluster.example.json --M 30 \
  --n-r 7 --t-r 3 --n-a 12 --t-a 3 \
  --coin-c 7 --coin-t 2 --coin-k 5 \
  --protocols a2dpss ru22 --ru22-mode native \
  --trial-id reshare-001 --out runs/reshare-001

python -m scripts.run_yoso_adpss_dist run --trial runs/reshare-001 --protocol a2dpss
python -m scripts.run_yoso_adpss_dist run --trial runs/reshare-001 --protocol ru22
python -m scripts.run_yoso_adpss_dist compare \
  --trial runs/reshare-001 --output runs/reshare-001-comparison.json
```

Preparation generates the keys and captures the code used for the trial. Each run produces node results for one protocol. Compare the reported maximum successor latency, `comparison_cpu_seconds` and `communication.total_network_bytes`. Native RU22 has its own leader/verifier flow. Both protocols reshare the same input and generate fresh output sharing randomness.

## Concurrent independent handovers

Reuse the five-node configuration from the first experiment. Queue one initial request and four later requests. This experiment precomputes the contributions before timing the online path.

```sh
python -m scripts.a2dpss_pipeline_benchmark prepare \
  --config-dir conf/exp-five --trial-id concurrent-001 \
  --initial-queries 1 --late-queries 4 --output runs/concurrent-001

for backend in dag acs
do
  python -m scripts.a2dpss_pipeline_benchmark run \
    --bundle runs/concurrent-001 --protocol "$backend" --nodes 0 1 2 3 4 \
    --output runs/concurrent-001-results || break
done

python -m scripts.a2dpss_pipeline_benchmark compare \
  --bundle runs/concurrent-001 --dag-results runs/concurrent-001-results \
  --acs-results runs/concurrent-001-results \
  --output runs/concurrent-001-comparison.json
```

The report records when the later requests become ready, enter the agreement path, and reach the successor committee. It also reports batch completion time, online ADPSS CPU time, communication and whether the late-arrival condition held.

## Running on multiple hosts

For the first and third experiments, pass `--peers peers.json` to `prepare`. Supply a JSON list of reachable `host:port` strings in logical node order. Use the corresponding experiment module's `export` command to create a separate bundle for each host. For example, `export --bundle runs/single-001 --nodes 0 1 --output runs/single-001-host-a` packages the public setup and those two nodes' private configurations. Use a distinct export for the other nodes. Run the same backend concurrently on every host with its own bundle and node IDs. Keep each backend's public result subdirectory when merging the node results for `compare`.

For the second experiment, supply a cluster file whose `hosts` array provides each host's `name`, `transport` (`ssh`), `address`, `ssh` target, `python` interpreter and `run_root`. Omit node assignments for round-robin placement or supply `nodes` for each host so the assignments cover all logical nodes exactly once. The runner stages per-host bundles. Use mutually trusted experiment hosts with compatible native libraries and a reachable TCP network.
