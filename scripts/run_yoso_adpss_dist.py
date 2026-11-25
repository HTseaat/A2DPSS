"""Prepare a shared fixture; run exactly one ADPSS protocol once on local/SSH hosts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("a2dpss", "ru22", "bhk22")
SCHEMA = "yoso-adpss-distributed/v1"
SSH_STAGE_ATTEMPTS = 4
SSH_STAGE_WORKERS = 10


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o600)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def checked_label(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("trial/host names must use letters, digits, dots, underscores or hyphens")
    return value


def load_cluster(path, M, base_port):
    raw = read(path)
    hosts = raw.get("hosts")
    if not isinstance(hosts, list) or not hosts:
        raise ValueError("cluster must contain a nonempty hosts list")
    explicit = ["nodes" in h for h in hosts]
    if any(explicit) and not all(explicit):
        raise ValueError("give nodes for every host, or omit nodes for automatic round-robin placement")
    normalized = []
    peers = [None] * M
    for index, original in enumerate(hosts):
        h = dict(original)
        h["name"] = checked_label(h.get("name", f"host{index}"))
        h.setdefault("transport", "ssh")
        if h["transport"] not in ("local", "ssh"):
            raise ValueError("host transport must be local or ssh")
        address = h.get("address", "127.0.0.1" if h["transport"] == "local" else "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", address):
            raise ValueError("address must be an IPv4 address or hostname")
        h["address"] = address
        if h["transport"] == "ssh":
            target = h.get("ssh", "")
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", target):
                raise ValueError("ssh must be a host or user@host (SSH config aliases supported)")
            if address in ("localhost", "127.0.0.1"):
                raise ValueError("SSH nodes need cluster-reachable advertised addresses")
            h.setdefault("python", "python3")
            h.setdefault("run_root", "/tmp/yoso-adpss-runs")
            if not Path(h["run_root"]).is_absolute() or h["run_root"] == "/":
                raise ValueError("remote run_root must be an absolute experiment directory")
            h.setdefault("ssh_port", 22)
            if not isinstance(h["ssh_port"], int) or not 1 <= h["ssh_port"] <= 65535:
                raise ValueError("invalid SSH port")
        else:
            h.setdefault("python", sys.executable)
        h["nodes"] = h.get("nodes", list(range(index, M, len(hosts))))
        for node in h["nodes"]:
            if isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < M:
                raise ValueError("logical node ids must be integers in [0,M)")
            if peers[node] is not None:
                raise ValueError("duplicate logical node placement")
            peers[node] = f"{address}:{base_port + node}"
        if h["nodes"]:
            normalized.append(h)
    if any(p is None for p in peers) or len(set(peers)) != M:
        raise ValueError("every logical node needs one distinct network endpoint")
    if base_port < 1024 or base_port + M - 1 > 65535:
        raise ValueError("node port range must be within 1024..65535")
    if len({h["name"] for h in normalized}) != len(normalized):
        raise ValueError("duplicate host names")
    return normalized, peers


def source_files():
    for folder in ("beaver", "scripts", "optimizedhbmpc"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                if path.suffix in (".py", ".so", ".yaml", ".yml"):
                    yield path
    yield from sorted(ROOT.glob("*.so"))


def prepare(args):
    from beaver.atcaas_params import ATCaaSProtocolParameters
    from beaver.protocol_trace import CommitteeRoleTraceArtifact
    from beaver.yoso_adpss_workload import build_yoso_adpss_fixture, write_yoso_adpss_fixture, normalize_protocols
    protocols = normalize_protocols(getattr(args, "protocols", PROTOCOLS))
    params = ATCaaSProtocolParameters(
        M=args.M, n_r=args.n_r, t_r=args.t_r, n_a=args.n_a, t_a=args.t_a,
        m_a=args.n_a, quorum_h=args.n_a - args.t_a, coin_c=args.coin_c,
        coin_t=args.coin_t, coin_k=args.coin_k)
    if args.n_r != 2 * args.t_r + 1:
        raise ValueError("the common RU22 fixture requires n_r = 2*t_r + 1")
    checked_label(args.trial_id)
    hosts, peers = load_cluster(args.cluster, args.M, args.base_port)
    reusable_config = getattr(args, "config_dir", None)
    if reusable_config is not None:
        from beaver.reusable_ru22_bundle import validate_base_config_directory
        reusable_config = Path(reusable_config).resolve()
        # Validate the entire input before creating output or changing any files.
        validate_base_config_directory(reusable_config, params)
    if args.dry_run:
        return dict(status="ready", protocols=list(protocols), parameters=params.to_config_obj(), hosts=hosts, peers=peers)
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    write(directory / "preparation.json", {"status": "preparing"})
    configs = directory / "configs"
    command = [sys.executable, "-m", "scripts.run_key_gen_atcaas", "--out-dir", str(configs)]
    for flag, value in (("M", args.M), ("n-r", args.n_r), ("t-r", args.t_r),
                        ("n-a", args.n_a), ("t-a", args.t_a), ("m-a", args.n_a),
                        ("quorum-h", params.quorum_h), ("coin-c", args.coin_c),
                        ("coin-t", args.coin_t), ("coin-k", args.coin_k)):
        command += ["--" + flag, str(value)]
    with (directory / "prepare.log").open("w") as log:
        if reusable_config is None:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            configs.mkdir(mode=0o700)
            for node in range(args.M):
                target = configs / f"local.{node}.json"
                shutil.copy2(reusable_config / target.name, target)
                target.chmod(0o600)
            # Lookup files are copied once here. Node packages may hardlink
            # these public files; mutable configs never alias the sealed base.
            shutil.copytree(reusable_config / "lookup_tables", configs / "lookup_tables")
            for path in (configs / "lookup_tables").rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
            (configs / "lookup_tables").chmod(0o700)
            if (reusable_config / "coin_manifest.json").is_file():
                shutil.copy2(reusable_config / "coin_manifest.json", configs / "coin_manifest.json")
                (configs / "coin_manifest.json").chmod(0o600)
            log.write("Reused validated Phase-6 configuration; keygen was not invoked.\n")
        for node in range(args.M):
            path = configs / f"local.{node}.json"
            config = read(path)
            config["peers"] = peers
            # Unrelated historical re-encryption fixture is not needed by this
            # experiment and must not be sent to every distributed node.
            config["extra"].pop("re_stage", None)
            write(path, config)
        # RU22's existing context uses an integer epoch; trial_id remains the
        # human-readable identity and all arms share this recorded numeric epoch.
        epoch = str(secrets.randbelow(2**31 - 1) + 1)
        public = directory / "public"
        public.mkdir()
        trace = public / "role-trace.json"
        subprocess.run([sys.executable, "-m", "scripts.generate_dag_role_trace",
                        "--config-dir", str(configs), "--M", str(args.M),
                        "--epoch-prefix", epoch, "--rounds", str(args.rounds),
                        "--indices", str(max(10, args.n_a)), "--selection-mode", "fixed-size",
                        "--fixed-committee-seed", epoch, "--output", str(trace)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    trace_digest = CommitteeRoleTraceArtifact.from_file(trace).digest()
    fixture = directory / "fixture"
    bundle = build_yoso_adpss_fixture(
        config_directory=configs, trial_id=args.trial_id, trial_epoch=epoch,
        seed=secrets.token_hex(32), role_trace_digest=trace_digest,
        network_profile={"peers": peers, "kind": "distributed"},
        bhk22_path_mode=args.bhk22_path, protocols=protocols)
    write_yoso_adpss_fixture(bundle, fixture)
    ru22_mode = getattr(args, "ru22_mode", "native")
    if ru22_mode == "native":
        from beaver.baselines.ru22_native_experiment import write_common_native_fixture
        write_common_native_fixture(bundle, fixture)
    for path in fixture.rglob("*.json"):
        path.chmod(0o600)
    code = directory / "code"
    hashes = {}
    for source in source_files():
        relative = source.relative_to(ROOT)
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha(destination)
    # Each node package can be shipped independently; no other node's config or
    # private shard is needed by its fixture loader.
    for node in range(args.M):
        target = directory / "nodes" / str(node)
        target.mkdir(parents=True)
        shutil.copy2(configs / f"local.{node}.json", target / "config.json")
        shutil.copytree(
            configs / "lookup_tables",
            target / "lookup_tables",
            copy_function=os.link,
        )
        node_fixture = target / "fixture"
        node_fixture.mkdir()
        shutil.copy2(fixture / "manifest.json", node_fixture / "manifest.json")
        shutil.copytree(fixture / "arms", node_fixture / "arms")
        shutil.copy2(fixture / f"node.{node}.private.json", node_fixture / f"node.{node}.private.json")
        if ru22_mode == "native":
            from beaver.baselines.ru22_native_experiment import PUBLIC_FILE, private_file
            for name in (PUBLIC_FILE, private_file(node)):
                shutil.copy2(fixture / name, node_fixture / name)
    settings = dict(schema=SCHEMA, trial_id=args.trial_id, epoch=epoch,
                    protocols=list(protocols),
                    parameters=params.to_config_obj(), peers=peers,
                    common_workload_digest=bundle.manifest.digest(),
                    role_trace_digest=trace_digest, rounds=args.rounds,
                    timeout=args.timeout, startup_timeout=args.startup_timeout,
                    coin_timeout=args.coin_timeout, source_hashes=hashes,
                    bhk22_path=args.bhk22_path,
                    ru22_mode=ru22_mode,
                    chunk_lookup_tables=read(configs / "local.0.json")["extra"]["chunk_lookup_tables"],
                    node_hashes={str(n): {str(p.relative_to(directory / "nodes" / str(n))): sha(p)
                                           for p in (directory / "nodes" / str(n)).rglob("*")
                                           if p.is_file()}
                                 for n in range(args.M)})
    settings["digest"] = digest(settings)
    write(public / "experiment.json", settings)
    write(directory / "cluster.json", {"hosts": hosts})
    write(directory / "preparation.json", {"status": "ready", "digest": settings["digest"]})
    return dict(status="ready", trial=str(directory), parameters=params.to_config_obj(),
                protocols=list(protocols), common_workload_digest=bundle.manifest.digest())


def load_settings(path):
    settings = read(path)
    if settings.get("schema") != SCHEMA or settings.get("digest") != digest(
            {k: v for k, v in settings.items() if k != "digest"}):
        raise ValueError("invalid prepared experiment manifest")
    return settings


def prepared_protocols(settings):
    from beaver.yoso_adpss_workload import normalize_protocols
    return normalize_protocols(settings.get("protocols", PROTOCOLS))


def require_prepared_protocol(settings, protocol):
    if protocol not in prepared_protocols(settings):
        raise ValueError(f"protocol {protocol!r} was not prepared for this trial")


def ssh_command(host, words):
    # The single remote command is shell-quoted; no values are interpolated as code.
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "-p", str(host["ssh_port"]), host["ssh"], shlex.join(words)]


def bundle_location(trial, host, settings, protocol):
    if host["transport"] == "local":
        return trial / "staged" / host["name"] / protocol
    return Path(host["run_root"]) / (settings["trial_id"] + "-" + settings["digest"][:12]) / protocol


def clone_staged_bundle(host, source, destination, experiment_digest):
    """Clone immutable input from a completed staging bundle on the same host."""
    script = ("import os,shutil,sys; from pathlib import Path; "
              "source,target=Path(sys.argv[1]),Path(sys.argv[2]); expected=sys.argv[3]; "
              "ready=lambda p: (p/'.stage-complete').is_file() and "
              "(p/'.stage-complete').read_text()==expected; "
              "sys.exit(0) if target.exists() and ready(target) else None; "
              "sys.exit(4) if target.exists() else None; "
              "sys.exit(3) if not ready(source) else None; "
              "temporary=Path(str(target)+'.stage-'+sys.argv[4]); "
              "ignored=lambda directory,names: {name for name in names if "
              "name in ('output','supervisor.json','.stage-complete') or name.endswith('.used')}; "
              "shutil.copytree(str(source),str(temporary),copy_function=os.link,ignore=ignored); "
              "marker=temporary/'.stage-complete'; marker.write_text(expected); marker.chmod(0o600); "
              "os.rename(str(temporary),str(target))")
    for attempt in range(1, SSH_STAGE_ATTEMPTS + 1):
        result = subprocess.run(
            ssh_command(host, [host.get("control_python", "python3"), "-c", script,
                               str(source), str(destination),
                               experiment_digest, secrets.token_hex(6)]),
            timeout=120, stdout=subprocess.DEVNULL)
        if result.returncode == 0:
            return True
        if result.returncode in (3, 4):
            return False
        if attempt < SSH_STAGE_ATTEMPTS:
            time.sleep(attempt)
    return False


def stage_from_cached_code(trial, host, settings, protocol, run_dir, destination, source_digest):
    """Hard-link immutable code from an earlier digest and upload only current inputs."""
    archive = run_dir / (host["name"] + ".tar.gz")
    overlays = []
    for raw in settings.get("deployment_code_overlays", []):
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("deployment code overlay must be a safe relative path")
        path = trial / "code" / relative
        if not path.is_file():
            raise ValueError("deployment code overlay is missing: " + str(relative))
        overlays.append(relative)
    with tarfile.open(archive, "w:gz", dereference=True) as tar:
        tar.add(trial / "public" / "role-trace.json", arcname="public/role-trace.json")
        tar.add(run_dir / "experiment.json", arcname="public/experiment.json")
        for relative in overlays:
            tar.add(trial / "code" / relative, arcname=str(Path("code") / relative))
        for node in host["nodes"]:
            tar.add(trial / "nodes" / str(node), arcname=f"nodes/{node}")
    archive.chmod(0o600)
    source = (Path(host["run_root"]) /
              (settings["trial_id"] + "-" + source_digest[:12]) / "a2dpss")
    control_python = host.get("control_python", "python3")
    setup = ("import os,shutil,sys; from pathlib import Path; os.umask(0o077); "
             "source,target=Path(sys.argv[1]),Path(sys.argv[2]); expected=sys.argv[3]; "
             "marker=source/'.stage-complete'; "
             "sys.exit(3) if not marker.is_file() or marker.read_text()!=expected else None; "
             "os.makedirs(target,mode=0o700,exist_ok=False); "
             "shutil.copytree(str(source/'code'),str(target/'code'),copy_function=os.link); "
             "[(target/'code'/raw).unlink() for raw in sys.argv[4:]]")
    promote = ("import os,sys; from pathlib import Path; "
               "source,target=Path(sys.argv[1]),Path(sys.argv[2]); "
               "marker=source/'.stage-complete'; marker.write_text(sys.argv[3]); "
               "marker.chmod(0o600); os.rename(str(source),str(target))")
    confirm = ("import sys; from pathlib import Path; "
               "marker=Path(sys.argv[1])/'.stage-complete'; "
               "raise SystemExit(0 if marker.is_file() and "
               "marker.read_text()==sys.argv[2] else 1)")
    cleanup = "import shutil,sys; shutil.rmtree(sys.argv[1],ignore_errors=True)"
    staged = False
    for attempt in range(1, SSH_STAGE_ATTEMPTS + 1):
        temporary = Path(str(destination) + ".stage-" + secrets.token_hex(6))
        try:
            result = subprocess.run(
                ssh_command(host, [control_python, "-c", setup, str(source),
                                   str(temporary), source_digest,
                                   *[str(path) for path in overlays]]),
                timeout=120, stdout=subprocess.DEVNULL)
            if result.returncode == 3:
                archive.unlink()
                return False
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, result.args)
            with archive.open("rb") as stream:
                subprocess.run(
                    ssh_command(host, ["tar", "-xzf", "-", "-C", str(temporary)]),
                    stdin=stream, check=True, timeout=180, stdout=subprocess.DEVNULL)
            try:
                subprocess.run(
                    ssh_command(host, [control_python, "-c", promote, str(temporary),
                                       str(destination), settings["digest"]]),
                    check=True, timeout=45, stdout=subprocess.DEVNULL)
                staged = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                for _ in range(SSH_STAGE_ATTEMPTS):
                    result = subprocess.run(
                        ssh_command(host, [control_python, "-c", confirm,
                                           str(destination), settings["digest"]]),
                        timeout=45, stdout=subprocess.DEVNULL)
                    if result.returncode == 0:
                        staged = True
                        break
            if staged:
                break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        try:
            subprocess.run(
                ssh_command(host, [control_python, "-c", cleanup, str(temporary)]),
                timeout=45, stdout=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
        if attempt < SSH_STAGE_ATTEMPTS:
            time.sleep(attempt)
    if not staged:
        raise RuntimeError(
            f"failed to stage cached inputs for {host['name']} after {SSH_STAGE_ATTEMPTS} attempts"
        )
    archive.unlink()
    return True


def stage_host(trial, host, settings, protocol, run_dir):
    destination = bundle_location(trial, host, settings, protocol)
    # A deployment orchestrator may have populated the immutable bundle through
    # a faster in-cluster fanout.  The digest marker is the same atomic contract
    # used by this function after its normal SSH upload, so accepting it here is
    # safe and keeps staging transport independent from the measured protocol.
    if host["transport"] == "ssh":
        confirm = ("import sys; from pathlib import Path; "
                   "marker=Path(sys.argv[1])/'.stage-complete'; "
                   "raise SystemExit(0 if marker.is_file() and "
                   "marker.read_text()==sys.argv[2] else 1)")
        for attempt in range(1, SSH_STAGE_ATTEMPTS + 1):
            try:
                result = subprocess.run(
                    ssh_command(host, [host.get("control_python", "python3"), "-c", confirm,
                                       str(destination), settings["digest"]]),
                    timeout=45, stdout=subprocess.DEVNULL)
                if result.returncode == 0:
                    return destination
                # Exit 1 is the helper's authoritative "not staged" result;
                # transport failures such as SSH 255 are transient and retry.
                if result.returncode == 1:
                    break
            except subprocess.TimeoutExpired:
                pass
            if attempt < SSH_STAGE_ATTEMPTS:
                time.sleep(attempt)
    source_digest = settings.get("deployment_source_digest")
    if host["transport"] == "ssh" and source_digest:
        if stage_from_cached_code(
            trial, host, settings, protocol, run_dir, destination, source_digest
        ):
            return destination
    if host["transport"] == "ssh" and protocol != "a2dpss":
        source = bundle_location(trial, host, settings, "a2dpss")
        if clone_staged_bundle(host, source, destination, settings["digest"]):
            return destination
    archive = run_dir / (host["name"] + ".tar.gz")
    # Archive membership is deliberately explicit: no controller configs/fixture.
    # Node lookup artifacts are hard-linked locally; serialize regular files
    # so the same archive is accepted by our link-rejecting safe extractor.
    with tarfile.open(archive, "w:gz", dereference=True) as tar:
        tar.add(trial / "code", arcname="code", filter=lambda info: None if "__pycache__" in info.name else info)
        tar.add(trial / "public" / "role-trace.json", arcname="public/role-trace.json")
        tar.add(run_dir / "experiment.json", arcname="public/experiment.json")
        for node in host["nodes"]:
            tar.add(trial / "nodes" / str(node), arcname=f"nodes/{node}")
    archive.chmod(0o600)
    if host["transport"] == "local":
        destination.mkdir(parents=True, exist_ok=False)
        safe_extract(archive, destination)
    else:
        control_python = host.get("control_python", "python3")
        setup = ("import os,sys; os.umask(0o077); "
                 "os.makedirs(sys.argv[1],mode=0o700,exist_ok=False)")
        promote = ("import os,sys; from pathlib import Path; "
                   "source,target=Path(sys.argv[1]),Path(sys.argv[2]); "
                   "marker=source/'.stage-complete'; marker.write_text(sys.argv[3]); "
                   "marker.chmod(0o600); os.rename(str(source),str(target))")
        confirm = ("import sys; from pathlib import Path; "
                   "marker=Path(sys.argv[1])/'.stage-complete'; "
                   "raise SystemExit(0 if marker.is_file() and "
                   "marker.read_text()==sys.argv[2] else 1)")
        cleanup = ("import shutil,sys; shutil.rmtree(sys.argv[1],ignore_errors=True)")
        staged = False
        for attempt in range(1, SSH_STAGE_ATTEMPTS + 1):
            temporary = Path(str(destination) + ".stage-" + secrets.token_hex(6))
            try:
                subprocess.run(ssh_command(host, [control_python, "-c", setup, str(temporary)]),
                               check=True, timeout=45, stdout=subprocess.DEVNULL)
                with archive.open("rb") as source:
                    subprocess.run(ssh_command(host, ["tar", "-xzf", "-", "-C", str(temporary)]),
                                   stdin=source, check=True, timeout=180,
                                   stdout=subprocess.DEVNULL)
                try:
                    subprocess.run(ssh_command(host, [control_python, "-c", promote,
                                                       str(temporary), str(destination),
                                                       settings["digest"]]),
                                   check=True, timeout=45, stdout=subprocess.DEVNULL)
                    staged = True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    # A disconnect can happen after the atomic rename completed.
                    # Confirm the digest marker before deciding that promotion failed.
                    for _ in range(SSH_STAGE_ATTEMPTS):
                        result = subprocess.run(
                            ssh_command(host, [control_python, "-c", confirm,
                                               str(destination), settings["digest"]]),
                            timeout=45, stdout=subprocess.DEVNULL)
                        if result.returncode == 0:
                            staged = True
                            break
                if staged:
                    break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            subprocess.run(ssh_command(host, [control_python, "-c", cleanup, str(temporary)]),
                           timeout=45, stdout=subprocess.DEVNULL)
            if attempt < SSH_STAGE_ATTEMPTS:
                time.sleep(attempt)
        if not staged:
            raise RuntimeError(f"failed to stage {host['name']} after {SSH_STAGE_ATTEMPTS} attempts")
    archive.unlink()  # node-local material remains only in the protected bundle.
    return destination


def safe_extract(archive, destination):
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for item in members:
            path = Path(item.name)
            if path.is_absolute() or ".." in path.parts or not (item.isfile() or item.isdir()):
                raise ValueError("unsafe archive member")
        tar.extractall(destination, members=members)


def clean_environment(settings):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DAG_TOB_", "ATCAAS_", "ASYNC_YOSO_", "YOSO_ADPSS_"))}
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               ATCAAS_COMMITTEE_SELECTION_MODE="fixed-size",
               ATCAAS_FIXED_COMMITTEE_SEED=settings["epoch"], ATCAAS_TRIAL_EPOCH=settings["epoch"],
               DAG_TOB_COIN_MODE="threshold-bls", DAG_TOB_MAX_ROUNDS=str(settings["rounds"]),
               DAG_TOB_MAX_RUNTIME=str(settings["timeout"]))
    if "dpss_pool_wait_seconds" in settings:
        env["DAG_TOB_DPSS_POOL_WAIT_SECONDS"] = str(settings["dpss_pool_wait_seconds"])
    return env


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def host_run(args):
    bundle = args.bundle.resolve()
    settings = load_settings(bundle / "public/experiment.json")
    require_prepared_protocol(settings, args.protocol)
    for name, expected in settings["source_hashes"].items():
        if sha(bundle / "code" / name) != expected:
            raise ValueError("deployed source checksum mismatch: " + name)
    node_ids = sorted(int(p.name) for p in (bundle / "nodes").iterdir())
    if not node_ids:
        raise ValueError("empty host bundle")
    for node in node_ids:
        for name, expected in settings["node_hashes"][str(node)].items():
            if sha(bundle / "nodes" / str(node) / name) != expected:
                raise ValueError(f"node {node} configuration checksum mismatch")
    output = bundle / "output"
    output.mkdir(exist_ok=False)
    write(bundle / "supervisor.json", {"pid": os.getpid(), "protocol": args.protocol})
    processes, handles = [], []
    error = None
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    env = clean_environment(settings)
    env["ATCAAS_ROLE_TRACE"] = str(bundle / "public/role-trace.json")
    env["PYTHONPATH"] = str(bundle / "code")
    try:
        for node in node_ids:
            node_root = bundle / "nodes" / str(node)
            marker = node_root / (args.protocol + ".used")
            with marker.open("x") as f:
                f.write(settings["digest"] + "\n")
            log = (output / f"node.{node}.log").open("w")
            handles.append(log)
            processes.append(subprocess.Popen([
                sys.executable, "-m", "scripts.run_yoso_adpss_dist", "_worker",
                "--bundle", str(bundle), "--protocol", args.protocol, "--node", str(node)],
                cwd=bundle / "code", env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True))
        deadline = time.monotonic() + settings["startup_timeout"] + settings["timeout"] + 60
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("a local node failed; see node logs")
            if time.monotonic() >= deadline:
                raise TimeoutError("host process group exceeded its deadline")
            time.sleep(0.2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("node process failed")
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_processes(processes)
        for log in handles:
            log.close()
        signal.signal(signal.SIGTERM, previous)
        (bundle / "supervisor.json").unlink(missing_ok=True)
    write(output / "host.json", {"status": "failure" if error else "success", "error": error,
                                 "returncodes": [p.returncode for p in processes]})
    return 1 if error else 0


def worker(args):
    import contextlib
    import logging
    bundle = args.bundle.resolve()
    node_root = bundle / "nodes" / str(args.node)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(bundle / "public/experiment.json")
    require_prepared_protocol(settings, args.protocol)
    if args.protocol == "ru22" and settings.get("ru22_mode") == "native":
        import asyncio
        from beaver.config import HbmpcConfig
        from beaver.baselines.ru22_native_runner import run_native_node
        sys.argv = [sys.argv[0], "-d", "-f", str(node_root / "config.json"), "-time", "0"]
        HbmpcConfig.load_config()
        if HbmpcConfig.my_id != args.node:
            raise ValueError("native worker config node mismatch")
        result = asyncio.run(run_native_node(fixture=node_root / "fixture", node=args.node,
            peers=settings["peers"], timeout=settings["timeout"], startup_timeout=settings["startup_timeout"]))
        if result["common_workload_digest"] != settings["common_workload_digest"]:
            raise ValueError("native worker workload mismatch")
        result["experiment_digest"] = settings["digest"]
        result["lookup_table_preload_seconds"] = HbmpcConfig.lookup_table_preload_seconds
        write(bundle / "output" / f"node.{args.node}.json", result)
        return 0
    from scripts.run_yoso_adpss_dag_arm_node import main as run_node
    sys.argv = [sys.argv[0], "--fixture", str(node_root / "fixture"), "--arm", args.protocol,
                "--experiment-settings", str(bundle / "public/experiment.json"),
                "-d", "-f", str(node_root / "config.json"), "-time", "0"]
    # Final JSON goes to a dedicated file; hot-path logs stay in the per-node log.
    class Capture:
        def write(self, text):
            sys.__stdout__.write(text)
            if text.startswith('{'):
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    return
                if "yoso_adpss_metrics" in value:
                    write(bundle / "output" / f"node.{args.node}.json", value["yoso_adpss_metrics"])
        def flush(self):
            sys.__stdout__.flush()
    with contextlib.redirect_stdout(Capture()):
        run_node()
    return 0


def remote_stop(host, destination):
    # Signal only this run's supervisor. It owns and reaps its node process groups.
    script = ("import json,os,signal,sys; from pathlib import Path; "
              "p=Path(sys.argv[1])/'supervisor.json'; "
              "pid=json.loads(p.read_text())['pid'] if p.exists() else None; "
              "cmd=Path('/proc')/str(pid)/'cmdline'; "
              "valid=pid and cmd.exists() and str(p.parent).encode() in cmd.read_bytes(); "
              "os.kill(pid,signal.SIGTERM) if valid else None")
    subprocess.run(ssh_command(host, [host["python"], "-c", script, str(destination)]),
                   check=True, timeout=30, stdout=subprocess.DEVNULL)


def collect_host(host, destination, target):
    if host["transport"] == "local":
        shutil.copytree(destination / "output", target, dirs_exist_ok=True)
    else:
        archive = target.parent / (host["name"] + "-results.tar.gz")
        with archive.open("wb") as stream:
            subprocess.run(ssh_command(host, ["tar", "-czf", "-", "-C", str(destination), "output"]),
                           stdout=stream, check=True, timeout=180)
        unpack = target.parent / (host["name"] + "-unpack")
        unpack.mkdir()
        safe_extract(archive, unpack)
        shutil.copytree(unpack / "output", target, dirs_exist_ok=True)
        archive.unlink()
    host_result = target / "host.json"
    if host_result.exists():
        host_result.rename(target / (host["name"] + ".host.json"))


def execute(args):
    trial = args.trial.resolve()
    if read(trial / "preparation.json").get("status") != "ready":
        raise ValueError("trial preparation is incomplete")
    settings = load_settings(trial / "public/experiment.json")
    require_prepared_protocol(settings, args.protocol)
    hosts = read(trial / "cluster.json")["hosts"]
    if args.timeout is not None:
        settings["timeout"] = args.timeout
        settings["digest"] = digest({k: v for k, v in settings.items() if k != "digest"})
    run_dir = trial / "results" / args.protocol
    plan = dict(status="ready", protocol=args.protocol, trial=str(trial), output=str(run_dir),
                nodes=settings["parameters"]["M"], hosts=[h["name"] for h in hosts],
                timeout=settings["timeout"], experiment_digest=settings["digest"])
    if args.dry_run:
        return plan
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / "experiment.json", settings)
    write(run_dir / "run.json", plan)
    children, streams, staged = [], [], []
    errors = []
    try:
        # Staging is outside the measured protocol timeline.  Independent hosts
        # can receive their immutable code/config bundles concurrently; keeping
        # this serial makes lookup-table-enabled trials unnecessarily slow to
        # deploy without changing any protocol work.
        staged_by_name = {}
        with ThreadPoolExecutor(
            max_workers=min(SSH_STAGE_WORKERS, len(hosts)),
            thread_name_prefix="yoso-adpss-stage",
        ) as executor:
            futures = {
                executor.submit(
                    stage_host,
                    trial,
                    host,
                    settings,
                    args.protocol,
                    run_dir,
                ): host
                for host in hosts
            }
            for future in as_completed(futures):
                host = futures[future]
                staged_by_name[host["name"]] = future.result()
        staged = [(host, staged_by_name[host["name"]]) for host in hosts]
        for host, destination in staged:
            invocation = [host["python"], "-m", "scripts.run_yoso_adpss_dist", "_host-run",
                          "--bundle", str(destination), "--protocol", args.protocol]
            stream = (run_dir / (host["name"] + ".log")).open("w")
            streams.append(stream)
            if host["transport"] == "ssh":
                command = "cd " + shlex.quote(str(destination / "code")) + " && exec " + shlex.join(invocation)
                invocation = ssh_command(host, ["sh", "-c", command])
                cwd = ROOT
            else:
                cwd = destination / "code"
            children.append(subprocess.Popen(invocation, cwd=cwd, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True,
                                             env=clean_environment(settings)))
        deadline = time.monotonic() + settings["startup_timeout"] + settings["timeout"] + 90
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("a host supervisor failed")
            if time.monotonic() >= deadline:
                raise TimeoutError("distributed run exceeded its deadline")
            time.sleep(0.2)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("a host supervisor failed")
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if errors:
            for host, destination in staged:
                if host["transport"] == "ssh":
                    try:
                        remote_stop(host, destination)
                    except Exception as exc:
                        errors.append(f"remote cleanup {host['name']}: {exc}")
        stop_processes(children)
        for stream in streams:
            stream.close()
        for host, destination in staged:
            try:
                collect_host(host, destination, run_dir / "nodes")
            except Exception as exc:
                errors.append(f"collect {host['name']}: {exc}")
    result = dict(schema="yoso-adpss-dag-arm-summary/v1", protocol=args.protocol,
                  status="failure", errors=errors, experiment_digest=settings["digest"])
    if not errors:
        try:
            from scripts.analyze_yoso_adpss_dag_arm import analyze_arm
            nodes = [read(run_dir / "nodes" / f"node.{i}.json") for i in range(settings["parameters"]["M"])]
            result = analyze_arm(trial / "fixture", args.protocol, nodes, experiment_digest=settings["digest"])
            result["communication_endpoint"] = "global-completion-excluding-control"
        except Exception as exc:
            errors.append(f"analyze: {exc}")
    write(run_dir / "summary.json", result)
    return result


def compare(args):
    """Reanalyze existing node results; never start or alter a protocol run."""
    from beaver.yoso_adpss_workload import (
        load_yoso_adpss_public_fixture, normalize_protocols,
        verify_same_initial_workload,
    )
    from scripts.analyze_yoso_adpss_dag_arm import analyze_arm

    trial = args.trial.resolve()
    settings = load_settings(trial / "public/experiment.json")
    selected = normalize_protocols(getattr(args, "protocols", None) or prepared_protocols(settings))
    if not set(selected).issubset(prepared_protocols(settings)):
        raise ValueError("comparison includes a protocol that was not prepared")
    if settings.get("ru22_mode") != "native":
        raise ValueError("this comparison requires native RU22; use the historical analyzer for legacy-dag")
    manifest, arms = load_yoso_adpss_public_fixture(trial / "fixture")
    if (tuple(manifest.arm_order) != prepared_protocols(settings)
            or not verify_same_initial_workload(manifest, arms)
            or manifest.digest() != settings["common_workload_digest"]
            or manifest.parameters.to_config_obj() != settings["parameters"]):
        raise ValueError("prepared fixture/workload/parameter mismatch")
    summaries, metrics, failures = {}, {}, []
    run_digests = set()
    # run permits an explicit timeout override. Both arms must use the same
    # override; all other code, inputs, placement and budgets remain prepared.
    prepared_material = {k: v for k, v in settings.items() if k not in ("digest", "timeout")}
    for protocol in selected:
        try:
            directory = trial / "results" / protocol
            run_settings = load_settings(directory / "experiment.json")
            if {k: v for k, v in run_settings.items() if k not in ("digest", "timeout")} != prepared_material:
                raise ValueError("run settings differ from prepared inputs, parameters or budgets")
            run_digests.add(run_settings["digest"])
            nodes = [read(directory / "nodes" / f"node.{i}.json")
                     for i in range(settings["parameters"]["M"])]
            result = analyze_arm(trial / "fixture", protocol, nodes,
                                 experiment_digest=run_settings["digest"])
            if (result.get("status") != "success"
                    or result.get("common_workload_digest") != manifest.digest()
                    or result.get("parameters") != settings["parameters"]):
                raise ValueError("analyzed result does not match prepared workload/parameters")
            if protocol == "ru22" and result.get("protocol_mode") != "native":
                raise ValueError("native comparison contains legacy RU22 results")
            row = {
                "successor_max_latency_sec": result.get("successor_max_latency_sec"),
                "comparison_cpu_seconds": result.get("comparison_cpu_seconds"),
                "total_network_bytes": (result.get("communication") or {}).get("total_network_bytes"),
            }
            for name, value in row.items():
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("missing or invalid comparison metric: " + name)
            summaries[protocol], metrics[protocol] = result, row
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{protocol}: {error}")
    if len(run_digests) > 1:
        failures.append("arms used different run budgets/settings")
    ratios = {}
    if not failures:
        baseline = metrics["a2dpss"]
        ratios = {protocol: {name: value / baseline[name] if baseline[name] else None
                             for name, value in row.items()}
                  for protocol, row in metrics.items() if protocol != "a2dpss"}
    return {
        "schema": "yoso-adpss-distributed-comparison/v1",
        "status": "failure" if failures else "success",
        "failures": failures, "protocols": list(selected),
        "trial_id": settings["trial_id"], "parameters": settings["parameters"],
        "common_workload_digest": manifest.digest(),
        "experiment_digest": next(iter(run_digests)) if len(run_digests) == 1 else None,
        "ru22_mode": "native", "arms": summaries, "metrics": metrics,
        "ratios_to_a2dpss": ratios,
        "comparison_basis": "same initial secret/sharing, parameters, placement, code and run budgets; independently refreshed outputs need not be equal",
        "cpu_accounting": "comparison_cpu_seconds: ADPSS computation including RU22 QVC; transport authentication, sortition, DAG, coin and experiment control excluded",
        "communication_endpoint": "global-completion-excluding-control",
    }


def positive_int(raw):
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="generate configuration and selected compatible fixtures; run nothing")
    p.add_argument("--protocols", choices=PROTOCOLS, nargs="+", default=list(PROTOCOLS),
                   help="a2dpss ru22 for the thesis pair; default preserves the historical triplet")
    p.add_argument("--cluster", type=Path, required=True)
    for name in (
        "M", "n-r", "t-r", "n-a", "t-a", "coin-c", "coin-t", "coin-k"
    ):
        p.add_argument("--" + name, type=int, required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config-dir", type=Path,
                   help="reuse a complete Phase-6 base config/lookup directory instead of keygen")
    p.add_argument("--base-port", type=int, default=16000)
    p.add_argument("--rounds", type=positive_int, default=64)
    p.add_argument("--timeout", type=positive_int, default=900)
    p.add_argument("--startup-timeout", type=positive_int, default=180)
    p.add_argument("--coin-timeout", type=positive_int, default=120)
    p.add_argument("--bhk22-path", choices=("optimistic", "one-accusation"), default="optimistic")
    p.add_argument("--ru22-mode", choices=("native", "legacy-dag"), default="native",
                   help="native uses one n-member verifier committee per leader; legacy-dag replays the historical adaptation")
    p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("compare", help="reanalyze existing results for prepared protocols; run nothing")
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--protocols", choices=PROTOCOLS, nargs="+",
                   help="defaults to prepared protocols; optionally select a2dpss ru22 from an old triplet")
    p.add_argument("--output", type=Path, help="optional comparison JSON; otherwise only print the report")
    p = commands.add_parser("run", help="deploy and run one chosen protocol exactly once")
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--protocol", choices=PROTOCOLS, required=True)
    p.add_argument("--timeout", type=positive_int)
    p.add_argument("--dry-run", action="store_true")
    for name in ("_host-run", "_worker"):
        if len(sys.argv) < 2 or sys.argv[1] != name:
            continue
        p = commands.add_parser(name, help=argparse.SUPPRESS)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--protocol", choices=PROTOCOLS, required=True)
        if name == "_worker":
            p.add_argument("--node", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "_host-run":
            return host_run(args)
        if args.command == "_worker":
            return worker(args)
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "compare":
            result = compare(args)
            if args.output is not None:
                write(args.output, result)
        else:
            import fcntl
            with (args.trial / ".run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("another protocol is already running for this trial")
                result = execute(args)
    except Exception as exc:
        result = {"status": "failure", "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in ("ready", "success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
