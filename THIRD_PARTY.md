# Third-party sources and licenses

## Python support from HoneyBadgerMPC and AD-MPC

The following files incorporate support code from the `honeybadgermpc` package in the archived HoneyBadgerMPC source. Corresponding implementations also appear under `adkg/` in the archived AD-MPC source. Both source trees include the GNU General Public License, version 3. This release includes a [verbatim copy of the HoneyBadgerMPC license](licenses/HoneyBadgerMPC-GPL-3.0.txt).

| File in this repository | Corresponding HoneyBadgerMPC source |
| --- | --- |
| `beaver/broadcast/crypto/boldyreva.py` | `honeybadgermpc/broadcast/crypto/boldyreva.py` |
| `beaver/config.py` | `honeybadgermpc/config.py` |
| `beaver/ipc.py` | `honeybadgermpc/ipc.py` |
| `beaver/utils/misc.py` | `honeybadgermpc/utils/misc.py` |
| `beaver/utils/typecheck.py` | `honeybadgermpc/utils/typecheck.py` |

`typecheck.py` matches the archived source byte for byte. The other four files contain adaptations. Source comments and bundled copyright and license notices retain their original attributions.

## Native components

| Directory | Source and licensing information |
| --- | --- |
| `native/gnark-crypto/` | Includes code from [ConsenSys gnark-crypto](https://github.com/consensys/gnark-crypto). The upstream [Apache License 2.0](native/gnark-crypto/LICENSE) and applicable source-file headers accompany the code. |
| `native/amcl_rust_wrapper/` | The package manifest credits Lovesh Harchandani and identifies the [upstream Rust wrapper](https://github.com/lovesh/amcl_rust_wrapper). The [Apache License 2.0](native/amcl_rust_wrapper/LICENSE) accompanies the code. |
| `native/bulletproofs-amcl/` | The package manifest credits Lovesh Harchandani. The [component README](native/bulletproofs-amcl/README.md) credits [Dalek Bulletproofs](https://github.com/dalek-cryptography/bulletproofs) and offers the choice of the [MIT License](native/bulletproofs-amcl/LICENSE-MIT) or the [Apache License 2.0](native/bulletproofs-amcl/LICENSE-APACHE). |
