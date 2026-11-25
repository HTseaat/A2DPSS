"""Generate and validate a dedicated threshold-BLS common-coin key set.

Example:

    python -m scripts.generate_coin_keys generate \
        --config-dir conf/atcaas_4_4 --M 4 \
        --committee 0 1 2 3 --threshold 3

This is an offline trusted setup.  It updates only ``coin_*`` fields and
never replaces the repository's existing ``public_key``/``private_key``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from beaver.async_yoso_coin import (
    COIN_CONFIG_VERSION,
    CoinCommitteeConfig,
    coin_membership_digest,
    coin_public_key_digest,
    load_threshold_coin_key_material,
)
from beaver.broadcast.crypto.boldyreva import dealer


COIN_MANIFEST_FILENAME = "coin_manifest.json"
COIN_MANIFEST_VERSION = 1
COIN_SCHEME = "boldyreva-threshold-bls-mnt224"

COIN_CONFIG_FIELDS = frozenset(
    {
        "coin_config_version",
        "coin_committee",
        "coin_threshold",
        "coin_public_key",
        "coin_public_key_digest",
        "coin_membership_digest",
        "coin_private_key",
        "coin_private_share_index",
    }
)


def _plain_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _config_paths(config_dir: Path, network_size: int) -> List[Path]:
    return [config_dir / f"local.{node_id}.json" for node_id in range(network_size)]


def _load_base_configs(
    config_dir: Path,
    network_size: int,
) -> List[Tuple[Path, dict]]:
    if not config_dir.is_dir():
        raise FileNotFoundError(f"config directory does not exist: {config_dir}")
    result = []
    paths = _config_paths(config_dir, network_size)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing node config files: " + ", ".join(missing)
        )
    for node_id, path in enumerate(paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed JSON config: {path}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"node config must be a JSON object: {path}")
        if data.get("my_id") != node_id:
            raise ValueError(
                f"config {path} has my_id={data.get('my_id')!r}; "
                f"expected {node_id}"
            )
        configured_size = data.get("M", data.get("N"))
        if configured_size != network_size:
            raise ValueError(
                f"config {path} has network size {configured_size!r}; "
                f"expected {network_size}"
            )
        extras = data.get("extra")
        if not isinstance(extras, dict):
            raise ValueError(f"config {path} requires an extra JSON object")
        result.append((path, data))
    return result


def _validate_generation_parameters(
    network_size: int,
    committee: Iterable[int],
    threshold: int,
) -> Tuple[int, Tuple[int, ...], int]:
    network_size = _plain_int(network_size, name="M")
    threshold = _plain_int(threshold, name="threshold")
    # Reuse the Phase-0 canonical committee and threshold checks.  The zero
    # digest is only a placeholder until the real public key is generated.
    probe_node = 0
    members = tuple(committee)
    probe = CoinCommitteeConfig(
        network_size=network_size,
        my_id=probe_node,
        committee=members,
        threshold=threshold,
        public_key_digest=b"\x00" * 32,
        private_share_index=(
            members.index(probe_node) if probe_node in members else None
        ),
    )
    return network_size, probe.committee, probe.threshold


def _serialized_key(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _base64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _manifest(
    *,
    network_size: int,
    committee: Tuple[int, ...],
    threshold: int,
    public_key_digest: bytes,
    membership_digest: bytes,
    deterministic_test_setup: bool,
) -> dict:
    return {
        "version": COIN_MANIFEST_VERSION,
        "scheme": COIN_SCHEME,
        "trusted_setup": True,
        "deterministic_test_setup": deterministic_test_setup,
        "network_size": network_size,
        "committee": list(committee),
        "threshold": threshold,
        "public_key_digest": public_key_digest.hex(),
        "membership_digest": membership_digest.hex(),
        "share_index_by_node": {
            str(node_id): share_index
            for share_index, node_id in enumerate(committee)
        },
        "config_files": [
            f"local.{node_id}.json" for node_id in range(network_size)
        ],
    }


def _atomic_write_json_files(files: Mapping[Path, Mapping[str, Any]]) -> None:
    temporary = []
    try:
        for target, value in files.items():
            temp = target.with_name(
                f".{target.name}.coin-key-tmp-{os.getpid()}"
            )
            if temp.exists():
                raise FileExistsError(f"temporary output already exists: {temp}")
            payload = json.dumps(
                value,
                indent=4,
                ensure_ascii=False,
            ) + "\n"
            with temp.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append((temp, target))
        for temp, target in temporary:
            os.replace(str(temp), str(target))
    finally:
        for temp, _target in temporary:
            if temp.exists():
                temp.unlink()


def generate_and_install_coin_keys(
    *,
    config_dir: Path,
    network_size: int,
    committee: Iterable[int],
    threshold: int,
    seed: Optional[int] = None,
    force: bool = False,
) -> dict:
    """Generate one key set, install it, self-check it, and return manifest."""

    config_dir = Path(config_dir)
    network_size, committee, threshold = _validate_generation_parameters(
        network_size,
        committee,
        threshold,
    )
    base_configs = _load_base_configs(config_dir, network_size)
    manifest_path = config_dir / COIN_MANIFEST_FILENAME
    existing_fields = {
        path: sorted(COIN_CONFIG_FIELDS.intersection(data["extra"]))
        for path, data in base_configs
        if COIN_CONFIG_FIELDS.intersection(data["extra"])
    }
    if not force and (existing_fields or manifest_path.exists()):
        details = ", ".join(
            f"{path.name}:{fields}" for path, fields in existing_fields.items()
        )
        if manifest_path.exists():
            details = (details + ", " if details else "") + manifest_path.name
        raise FileExistsError(
            "coin key material already exists; use force=True/--force to replace it: "
            + details
        )

    public_key, private_keys = dealer(
        len(committee),
        threshold,
        seed=seed,
    )
    public_key_bytes = _serialized_key(public_key)
    public_digest = coin_public_key_digest(public_key_bytes)
    membership_digest = coin_membership_digest(
        network_size=network_size,
        committee=committee,
        threshold=threshold,
        public_key_digest=public_digest,
    )
    common = {
        "coin_config_version": COIN_CONFIG_VERSION,
        "coin_committee": list(committee),
        "coin_threshold": threshold,
        "coin_public_key": _base64(public_key_bytes),
        "coin_public_key_digest": public_digest.hex(),
        "coin_membership_digest": membership_digest.hex(),
    }

    outputs: Dict[Path, Mapping[str, Any]] = {}
    for node_id, (path, data) in enumerate(base_configs):
        extras = data["extra"]
        for field in COIN_CONFIG_FIELDS:
            extras.pop(field, None)
        extras.update(common)
        if node_id in committee:
            share_index = committee.index(node_id)
            extras["coin_private_share_index"] = share_index
            extras["coin_private_key"] = _base64(
                _serialized_key(private_keys[share_index])
            )
        outputs[path] = data

    manifest = _manifest(
        network_size=network_size,
        committee=committee,
        threshold=threshold,
        public_key_digest=public_digest,
        membership_digest=membership_digest,
        deterministic_test_setup=seed is not None,
    )
    outputs[manifest_path] = manifest
    _atomic_write_json_files(outputs)
    validate_coin_key_directory(
        config_dir=config_dir,
        network_size=network_size,
    )
    return manifest


def validate_coin_key_directory(
    *,
    config_dir: Path,
    network_size: int,
) -> dict:
    """Reload every node config and cryptographically self-check the key set."""

    config_dir = Path(config_dir)
    network_size = _plain_int(network_size, name="M")
    if network_size <= 0:
        raise ValueError("M must be positive")
    base_configs = _load_base_configs(config_dir, network_size)
    manifest_path = config_dir / COIN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing coin manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed coin manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("coin manifest must be a JSON object")
    if manifest.get("version") != COIN_MANIFEST_VERSION:
        raise ValueError("unsupported coin manifest version")
    if manifest.get("scheme") != COIN_SCHEME:
        raise ValueError("coin manifest scheme mismatch")
    if manifest.get("network_size") != network_size:
        raise ValueError("coin manifest network_size mismatch")

    materials = []
    for node_id, (_path, data) in enumerate(base_configs):
        materials.append(
            load_threshold_coin_key_material(
                data["extra"],
                network_size=network_size,
                my_id=node_id,
            )
        )
    first = materials[0]
    committee = first.config.committee
    threshold = first.config.threshold
    for material in materials[1:]:
        if material.config.committee != committee:
            raise ValueError("coin committee differs across node configs")
        if material.config.threshold != threshold:
            raise ValueError("coin threshold differs across node configs")
        if material.config.membership_digest != first.config.membership_digest:
            raise ValueError("coin membership digest differs across node configs")
        if material.public_key_bytes != first.public_key_bytes:
            raise ValueError("coin public key differs across node configs")
        if material.public_key != first.public_key:
            raise ValueError("decoded coin public key differs across node configs")

    expected_manifest = _manifest(
        network_size=network_size,
        committee=committee,
        threshold=threshold,
        public_key_digest=first.config.public_key_digest,
        membership_digest=first.config.membership_digest,
        deterministic_test_setup=bool(manifest.get("deterministic_test_setup")),
    )
    if manifest != expected_manifest:
        raise ValueError("coin manifest does not match installed node configs")

    private_by_index = {
        material.config.private_share_index: material.private_key
        for material in materials
        if material.config.is_member
    }
    if set(private_by_index) != set(range(len(committee))):
        raise ValueError("coin private-key share set is incomplete")
    message_hash = first.public_key.hash_message(
        "ATCAAS-COMMON-COIN-KEY-SELF-CHECK-V1"
    )
    signed = {
        index: private_by_index[index].sign(message_hash)
        for index in range(threshold)
    }
    for index, share in signed.items():
        first.public_key.verify_share(share, index, message_hash)
    signature = first.public_key.combine_shares(signed)
    first.public_key.verify_signature(signature, message_hash)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or validate ATCaaS threshold common-coin keys."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="generate and install a new offline trusted key set",
    )
    generate.add_argument("--config-dir", required=True, type=Path)
    generate.add_argument("--M", required=True, type=int, dest="network_size")
    generate.add_argument("--committee", required=True, type=int, nargs="+")
    generate.add_argument("--threshold", required=True, type=int)
    generate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="TEST ONLY: deterministic dealer seed; never use in experiments",
    )
    generate.add_argument(
        "--force",
        action="store_true",
        help="replace existing coin_* fields and coin manifest",
    )

    validate = subparsers.add_parser(
        "validate",
        help="reload and cryptographically validate an installed key set",
    )
    validate.add_argument("--config-dir", required=True, type=Path)
    validate.add_argument("--M", required=True, type=int, dest="network_size")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        if args.seed is not None:
            print(
                "WARNING: --seed creates reconstructible TEST keys; do not use "
                "them in experiments.",
                file=sys.stderr,
            )
        manifest = generate_and_install_coin_keys(
            config_dir=args.config_dir,
            network_size=args.network_size,
            committee=args.committee,
            threshold=args.threshold,
            seed=args.seed,
            force=args.force,
        )
        print(
            f"Installed coin keys for M={manifest['network_size']} "
            f"committee={manifest['committee']} k={manifest['threshold']} "
            f"in {args.config_dir}"
        )
        return 0
    validate_coin_key_directory(
        config_dir=args.config_dir,
        network_size=args.network_size,
    )
    print(f"Validated coin keys in {args.config_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
