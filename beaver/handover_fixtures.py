"""Neutral genesis/config helpers shared by handover protocol tests.

These helpers create initial sharing material and adapt the repository's
historical configuration schema.  They are setup utilities, not a wire
protocol or qid namespace.  The returned local config still contains the
legacy context shape so callers must project it into ``A2DPSSContext`` or
``OCDPSSContext`` before entering a formal protocol path.
"""

from beaver.legacy_yoso_dpss import (
    DPSSGenesisShares,
    DPSSLocalConfig,
    build_dpss_local_config,
    make_genesis_shares,
)


GenesisShares = DPSSGenesisShares
LocalHandoverConfig = DPSSLocalConfig
build_handover_local_config = build_dpss_local_config


__all__ = [
    "GenesisShares",
    "LocalHandoverConfig",
    "build_handover_local_config",
    "make_genesis_shares",
]
