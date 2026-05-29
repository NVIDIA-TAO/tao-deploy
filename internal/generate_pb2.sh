
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Generate _pb2.py file for respective model type

apt-get install -y protobuf-compiler

protoc nvidia_tao_deploy/cv/$1/proto/*.proto --python_out=.
