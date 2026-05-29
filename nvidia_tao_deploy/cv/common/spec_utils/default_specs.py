# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This script is responsible for generating default experiment.yaml files from dataclasses."""

from __future__ import annotations

import os
import logging
import importlib
from os import makedirs, listdir
from os.path import abspath, dirname, exists, join

from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass

import nvidia_tao_deploy
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

# Usage example:
# ==============
"""
python default_specs \
    results_dir=/results/classification_tf2/ \
    module_name=classification_tf2
"""


# Resolve the config root from the in-tree nvidia_tao_deploy.config package.
TAO_DEPLOY_PKG_ROOT = dirname(abspath(nvidia_tao_deploy.__file__))
CONFIG_ROOT = join(TAO_DEPLOY_PKG_ROOT, "config")

# Container directories under config/ and the deploy implementation tree that
# group networks by family (e.g. config/multimodal/clip). The walk recurses one
# level into these instead of treating them as networks themselves.
_CONTAINER_DIRS = {"multimodal"}
_SKIP_DIRS = {"utils", "__pycache__", "common"}


def get_supported_module_paths():
    """Discover supported networks and their dotted subpaths under config/.

    A network is "supported" when it has both:
      1. A config package at nvidia_tao_deploy/config/<...>/
      2. A deploy implementation with an entrypoint/ dir at
         nvidia_tao_deploy/{cv,multimodal}/<network>/

    Returns:
        Dict[str, str]: Map from short network name (e.g. "clip") to its
        dotted subpath under nvidia_tao_deploy.config (e.g. "multimodal.clip").
    """
    if not exists(CONFIG_ROOT):
        logging.warning("Config root not found at %s", CONFIG_ROOT)
        return {}

    # Walk config/ — flat entries become {name: name}; entries inside a known
    # container dir become {name: "<container>.<name>"}.
    config_paths = {}
    for entry in listdir(CONFIG_ROOT):
        entry_path = join(CONFIG_ROOT, entry)
        if not os.path.isdir(entry_path) or entry in _SKIP_DIRS:
            continue
        if entry in _CONTAINER_DIRS:
            for sub in listdir(entry_path):
                sub_path = join(entry_path, sub)
                if os.path.isdir(sub_path) and sub not in _SKIP_DIRS:
                    config_paths[sub] = f"{entry}.{sub}"
        else:
            config_paths[entry] = entry

    # Find deploy implementations with an entrypoint/ — search both cv/ and
    # the same container dirs (e.g. multimodal/) used on the config side.
    nvidia_tao_deploy_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
    deploy_modules = set()
    for impl_root in ("cv", *_CONTAINER_DIRS):
        impl_dir = join(nvidia_tao_deploy_dir, impl_root)
        if not exists(impl_dir):
            continue
        for item in listdir(impl_dir):
            item_path = join(impl_dir, item)
            if not os.path.isdir(item_path) or item in _SKIP_DIRS:
                continue
            if exists(join(item_path, "entrypoint")):
                deploy_modules.add(item)

    supported = {name: path for name, path in config_paths.items() if name in deploy_modules}

    if not supported:
        logging.warning(
            "No matching modules found between config (%d modules) "
            "and deploy implementation (%d modules)",
            len(config_paths), len(deploy_modules)
        )

    return supported


def get_supported_modules():
    """Return the sorted list of supported network short names."""
    return sorted(get_supported_module_paths())


def import_module_from_path(module_name):
    """
    Import a module from its full path.

    Args:
        module_name (str): Full module path (e.g., 'nvidia_tao_deploy.config.classification_tf2.default_config')

    Returns:
        module: The imported module
    """
    return importlib.import_module(module_name)


def dataclass_to_yaml(dataclass_obj, yaml_file_path):
    """
    Convert a dataclass object to a YAML file using omegaconf.

    Args:
        dataclass_obj (object): The dataclass object to convert.
        yaml_file_path (str): The path to the output YAML file.

    Returns:
        None
    """
    if not hasattr(dataclass_obj, "__dataclass_fields__"):
        raise ValueError("Provided object is not a dataclass instance.")

    # Convert dataclass to OmegaConf structured object
    conf = OmegaConf.structured(dataclass_obj)

    # Save as YAML
    output_dir = dirname(yaml_file_path)
    if output_dir and not exists(output_dir):
        makedirs(output_dir, exist_ok=True)
    with open(yaml_file_path, 'w', encoding='utf-8') as yaml_file:
        yaml_file.write(OmegaConf.to_yaml(conf))
        logging.info("Generated default spec: %s", yaml_file_path)


@dataclass
class DefaultConfig:
    """This is a structured config for generating default specs."""

    # Minimalistic experiment manager.
    results_dir: str = MISSING
    module_name: str = MISSING


spec_path = dirname(abspath(__file__))


@hydra_runner(config_path=spec_path, config_name="default_specs", schema=DefaultConfig)
def main(cfg: DefaultConfig) -> None:
    """Script to generate default experiment YAML from dataclasses.

    Args:
        cfg (OmegaConf.DictConf): Hydra parsed config object.
    """
    logging.info("Generating default spec for module: %s", cfg.module_name)

    # Validate module name
    supported_module_paths = get_supported_module_paths()
    if cfg.module_name not in supported_module_paths:
        error_msg = (f"Module '{cfg.module_name}' is not supported.\n"
                     f"Supported modules: {', '.join(sorted(supported_module_paths))}")
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Create results directory if it doesn't exist
    if not exists(cfg.results_dir):
        makedirs(cfg.results_dir, exist_ok=True)
        logging.info("Created results directory: %s", cfg.results_dir)

    # Set output file path
    output_filename = "experiment.yaml"
    output_path = join(cfg.results_dir, output_filename)
    if exists(output_path):
        logging.warning("Output file already exists and will be overwritten: %s", output_path)

    # Import the module and get the ExperimentConfig dataclass
    module_path = f"nvidia_tao_deploy.config.{supported_module_paths[cfg.module_name]}.default_config"
    try:
        imported_module = import_module_from_path(module_path)
        if not hasattr(imported_module, 'ExperimentConfig'):
            raise AttributeError(f"Module '{module_path}' does not have 'ExperimentConfig' dataclass")

        # Generate YAML from dataclass
        dataclass_to_yaml(imported_module.ExperimentConfig, output_path)

        # Success logging
        logging.info("Default specification file for %s generated at '%s'", cfg.module_name, output_path)

    except ImportError as e:
        error_msg = f"Failed to import module '{module_path}': {str(e)}"
        logging.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Failed to generate spec for {cfg.module_name}: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
