# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper function to decode ctc trained model's output."""


def decode_ctc_conf(pred,
                    classes,
                    blank_id):
    """Decode ctc trained model's output.

    Return decoded license plate and confidence.
    """
    pred_id = []
    pred_conf = []
    for pred_item in pred:
        # Looking at first element of first output batch
        if isinstance(pred_item[0][0], float):
            pred_conf = pred_item
        elif isinstance(pred_item[0][0], int):
            pred_id = pred_item
        else:
            raise ValueError("Unsupported output data type.")
    decoded_lp = []
    decoded_conf = []

    for idx_in_batch, seq in enumerate(pred_id):
        seq_conf = pred_conf[idx_in_batch]
        prev = seq[0]
        tmp_seq = [prev]
        tmp_conf = [seq_conf[0]]
        for idx in range(1, len(seq)):
            if seq[idx] != prev:
                tmp_seq.append(seq[idx])
                tmp_conf.append(seq_conf[idx])
                prev = seq[idx]
        lp = ""
        output_conf = []
        for index, i in enumerate(tmp_seq):
            if i != blank_id:
                lp += classes[i]
                output_conf.append(tmp_conf[index])
        decoded_lp.append(lp)
        decoded_conf.append(output_conf)

    return decoded_lp, decoded_conf
