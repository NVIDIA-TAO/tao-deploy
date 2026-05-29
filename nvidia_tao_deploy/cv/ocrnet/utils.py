# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper function to decode OCRNet model's output."""

import numpy as np


def decode_ctc(output_id, output_prob, character_list, blank_id=0):
    """Decode the raw CTC output to string."""
    prob = np.cumprod(output_prob)[-1]
    seq = np.squeeze(output_id)
    prev = seq[0]
    tmp_seq = [prev]
    for idx in range(1, len(seq)):
        if seq[idx] != prev:
            tmp_seq.append(seq[idx])
            prev = seq[idx]
    text = ""
    for idx in tmp_seq:
        if idx != blank_id:
            text += character_list[idx]
    return text, prob


def decode_attn(output_id, output_prob, character_list):
    """Decode the raw attn output to string."""
    seq = np.squeeze(output_id)
    pred = ''.join([character_list[i] for i in seq])
    pred_EOS = pred.find('[s]')
    text = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
    prob = np.cumprod(output_prob[:pred_EOS])[-1]
    return text, prob
