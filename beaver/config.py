"""
Module for ``honeybadgermpc``'s configuration.

This module can be used to:

* define default configuration settings
* load a configuration
* validate a comfiguration

Sample config can be found at: conf/sample.ini
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import time

from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.chunk_elgamal import install_lookup_table_artifacts
# from honeybadgermpc.reed_solomon import Algorithm as RSAlgorithm


class NodeDetails(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port


# class ConfigVars(object):
#     Reconstruction = "reconstruction"


# class ReconstructionConfig(object):
#     def __init__(self, induce_faults, decoding_algorithm):
#         self.induce_faults = induce_faults
#         self.decoding_algorithm = decoding_algorithm

#     @classmethod
#     def default(cls):
#         return cls(induce_faults=False, decoding_algorithm=RSAlgorithm.GAO)

#     @classmethod
#     def from_json(cls, json_config):
#         res = cls.default()
#         if "induce_faults" in json_config:
#             res.induce_faults = json_config["induce_faults"]

#         decoding_algorithms = [RSAlgorithm.WELCH_BERLEKAMP, RSAlgorithm.GAO]
#         if "decoding_algorithm" in json_config:
#             decoding_algorithm = json_config["decoding_algorithm"]
#             assert (
#                 decoding_algorithm in decoding_algorithms
#             ), f"decoding_algorithm must be in {decoding_algorithms}"
#             res.decoding_algorithm = json_config["decoding_algorithm"]

#         return res


class HbmpcConfig(object):
    N = None
    M = None
    committee_n = None
    expected_committee_size = None
    committee_m = None
    quorum_h = None
    n_r = None
    t_r = None
    n_a = None
    t_a = None
    m_a = None
    agreement_t_impl = None
    coin_c = None
    coin_t = None
    coin_k = None
    atcaas_parameters = None
    t = None
    my_id = None
    peers = None
    skip_preprocessing = False
    extras = None
    reconstruction = None
    lookup_table_preload_seconds = 0.0

    @staticmethod
    def load_config():
        parser = ArgumentParser(description="Runs an HBMPC program.")

        parser.add_argument(
            "-d",
            "--distributed",
            dest="is_dist",
            action="store_true",
            help="Indicates that the program is being run in a distributed setting. \
                This will validate all `default` and `required` parameters.",
        )

        parser.add_argument(
            "-f",
            "--config-file",
            type=str,
            dest="config_file_path",
            #dest='\conf\avss\local.0.json',
            help="Path from where to load the HBMPC config file.",
        )
        
        parser.add_argument(
            "-time",
            type=int,
            dest="time",
            help="Stores the start time of the experiment.",
        )

        args = parser.parse_args()
        config = json.load(open(args.config_file_path))

        HbmpcConfig.N = config["N"]
        HbmpcConfig.M = config.get("M", HbmpcConfig.N)
        HbmpcConfig.committee_n = config.get("n", HbmpcConfig.N)
        HbmpcConfig.expected_committee_size = config.get(
            "expected_committee_size",
            HbmpcConfig.committee_n,
        )
        HbmpcConfig.committee_m = config.get(
            "committee_m",
            HbmpcConfig.expected_committee_size,
        )
        HbmpcConfig.t = config["t"]
        HbmpcConfig.quorum_h = config.get("quorum_h", 2 * HbmpcConfig.t + 1)
        if "atcaas_parameters" in config:
            params = ATCaaSProtocolParameters.from_mapping(config)
            aliases = {
                "expected_committee_size": params.n_a,
                "committee_m": params.m_a,
                "quorum_h": params.quorum_h,
                "t": params.agreement_t_impl,
                "n_r": params.n_r,
                "t_r": params.t_r,
                "n_a": params.n_a,
                "t_a": params.t_a,
                "m_a": params.m_a,
                "coin_c": params.coin_c,
                "coin_t": params.coin_t,
                "coin_k": params.coin_k,
            }
            for field, expected in aliases.items():
                if config.get(field) != expected:
                    raise ValueError(
                        f"top-level {field} does not match atcaas_parameters"
                    )
            HbmpcConfig.atcaas_parameters = params
            HbmpcConfig.n_r = params.n_r
            HbmpcConfig.t_r = params.t_r
            HbmpcConfig.n_a = params.n_a
            HbmpcConfig.t_a = params.t_a
            HbmpcConfig.m_a = params.m_a
            HbmpcConfig.agreement_t_impl = params.agreement_t_impl
            HbmpcConfig.coin_c = params.coin_c
            HbmpcConfig.coin_t = params.coin_t
            HbmpcConfig.coin_k = params.coin_k
            HbmpcConfig.expected_committee_size = params.n_a
            HbmpcConfig.committee_m = params.m_a
            HbmpcConfig.quorum_h = params.quorum_h
            HbmpcConfig.t = params.agreement_t_impl
        else:
            HbmpcConfig.atcaas_parameters = None
            HbmpcConfig.n_r = config.get("n_r")
            HbmpcConfig.t_r = config.get("t_r", HbmpcConfig.t)
            HbmpcConfig.n_a = config.get(
                "n_a", HbmpcConfig.expected_committee_size
            )
            HbmpcConfig.t_a = config.get(
                "t_a", HbmpcConfig.n_a - HbmpcConfig.quorum_h
            )
            HbmpcConfig.m_a = config.get("m_a", HbmpcConfig.committee_m)
            HbmpcConfig.agreement_t_impl = (
                HbmpcConfig.m_a - HbmpcConfig.quorum_h
            )
            HbmpcConfig.coin_c = config.get("coin_c")
            HbmpcConfig.coin_t = config.get("coin_t")
            HbmpcConfig.coin_k = config.get("coin_k")
        HbmpcConfig.my_id = config["my_id"]
        HbmpcConfig.peers = {
            peerid: NodeDetails(addrinfo.split(":")[0], int(addrinfo.split(":")[1]))
            for peerid, addrinfo in enumerate(config["peers"])
        }
        
        HbmpcConfig.time = args.time
        # if args.is_dist:
        #     config = json.load(open(args.config_file_path))

        #     HbmpcConfig.N = config["N"]
        #     HbmpcConfig.t = config["t"]
        #     HbmpcConfig.my_id = config["my_id"]
        #     HbmpcConfig.peers = {
        #         peerid: NodeDetails(addrinfo.split(":")[0], int(addrinfo.split(":")[1]))
        #         for peerid, addrinfo in enumerate(config["peers"])
        #     }

        if "skip_preprocessing" in config:
            HbmpcConfig.skip_preprocessing = config["skip_preprocessing"]
        # Do not retain another node/configuration's extras when this process
        # loads more than one config (notably in tests and local orchestration).
        HbmpcConfig.extras = config.get("extra")
        HbmpcConfig.lookup_table_preload_seconds = 0.0
        lookup_tables = (config.get("extra") or {}).get(
            "chunk_lookup_tables"
        )
        if lookup_tables is not None:
            lookup_started = time.perf_counter()
            install_lookup_table_artifacts(
                lookup_tables,
                base_directory=Path(args.config_file_path).resolve().parent,
            )
            HbmpcConfig.lookup_table_preload_seconds = (
                time.perf_counter() - lookup_started
            )
        if HbmpcConfig.atcaas_parameters is not None:
            extras = HbmpcConfig.extras or {}
            coin_committee = extras.get("coin_committee")
            if not isinstance(coin_committee, list) or len(
                coin_committee
            ) != HbmpcConfig.coin_c:
                raise ValueError("coin_committee size does not match coin_c")
            if extras.get("coin_threshold") != HbmpcConfig.coin_k:
                raise ValueError("coin_threshold does not match coin_k")
            dpss = extras.get("dpss")
            if not isinstance(dpss, dict):
                raise ValueError("Phase-6 config requires extra.dpss")
            if dpss.get("private_material_scope") != "node-local":
                raise ValueError("Phase-6 DPSS config must be node-local")
            if dpss.get("n_r") != HbmpcConfig.n_r:
                raise ValueError("DPSS n_r does not match protocol parameters")
            if dpss.get("t_r") != HbmpcConfig.t_r:
                raise ValueError("DPSS t_r does not match protocol parameters")

        reconstruction_data = {}
        # if "reconstruction" in config:
        #     reconstruction_data = config["reconstruction"]

        # HbmpcConfig.reconstruction = ReconstructionConfig.from_json(
        #     reconstruction_data
        # )

        # Ensure the required values are set before this method terminates
        assert HbmpcConfig.my_id is not None, "Node Id: missing"
        assert HbmpcConfig.N is not None, "N: missing"
        assert HbmpcConfig.M is not None, "M: missing"
        assert HbmpcConfig.committee_n is not None, "n: missing"
        assert HbmpcConfig.expected_committee_size is not None, "expected_committee_size: missing"
        assert HbmpcConfig.committee_m is not None, "committee_m: missing"
        assert HbmpcConfig.quorum_h is not None, "quorum_h: missing"
        assert HbmpcConfig.t is not None, "t: missing"
        assert HbmpcConfig.peers is not None, "peers: missing"
