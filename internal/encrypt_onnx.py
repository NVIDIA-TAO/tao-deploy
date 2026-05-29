# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal script to encrypt an onnx file to .etlt."""

import argparse
import os
from nvidia_tao_deploy.utils.decoding import encode_etlt

def main(args=None):
    """Encrypt an onnx file."""
    args = parse_command_line_arguments(args)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"{args.model_path} does not exist")
    
    dir_name = os.path.dirname(args.output_path)
    os.makedirs(dir_name, exist_ok=True)

    encode_etlt(args.model_path, args.output_path, "", args.key)
    print(f"Model encrypted at {args.output_path}")

def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='encrypt_onnx', description='Encrypt an onnx file.')

    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to an onnx model file.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        required=True,
        help='Key to save a .etlt model.'
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        required=True,
        help="Output .etlt file path."
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
