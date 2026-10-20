// SPDX-FileCopyrightText: Copyright (c) 2024 Horizon Robotics
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

template<typename feature_t, typename index_t, bool bilinear>
void deformable_aggregation(
    feature_t* output,
    const feature_t* mc_ms_feat,
    const index_t* spatial_shape,
    const index_t* scale_start_index,
    const feature_t* sample_location,
    const feature_t* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups,
    void* stream
);
