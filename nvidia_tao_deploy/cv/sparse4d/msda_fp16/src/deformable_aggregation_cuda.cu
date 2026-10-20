// SPDX-FileCopyrightText: Copyright (c) 2024 Horizon Robotics
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#include "deformable_aggregation_cuda.h"
#include <cuda_fp16.h>
#include <stdio.h>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#  define CP_ASYNC_SM80_ENABLED
#endif

static __device__ void cpasync_global_to_shared_4byte(void *dst, const void *src) {
  #ifdef CP_ASYNC_SM80_ENABLED
    unsigned int shmem_ptr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 4;" ::"r"(shmem_ptr), "l"(src));
  #else
    *(int*)dst = *(const int*)src;
  #endif
}

static __device__ void cpasync_global_to_shared_8byte(void *dst, const void *src) {
  #ifdef CP_ASYNC_SM80_ENABLED
    unsigned int shmem_ptr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;" ::"r"(shmem_ptr), "l"(src));
  #else
    *(int2*)dst = *(const int2*)src;
  #endif
}

static __device__ void cpasync_wait(){
  #ifdef CP_ASYNC_SM80_ENABLED
    asm volatile("cp.async.wait_all;\n" ::);
  #endif
}

template<typename value_t, bool bilinear, int num_embeds>
struct sampling{};

template<int num_embeds> struct sampling<float, true, num_embeds>{
  static float __forceinline__ __device__ call(int width, int height, int scale_start_index, int ibatch_by_num_feat, int ic, const float* mc_ms_feat, const float2& loc){
    const float h_im = loc.y * height - 0.5f;
    const float w_im = loc.x * width  - 0.5f;
    mc_ms_feat += (ibatch_by_num_feat + scale_start_index) * num_embeds;

    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
      v1 = mc_ms_feat[(h_low * width + w_low) * num_embeds + ic];

    float v2 = 0;
    if (h_low >= 0 && w_high < width)
      v2 = mc_ms_feat[(h_low * width + w_high) * num_embeds + ic];

    float v3 = 0;
    if (h_high < height && w_low >= 0)
      v3 = mc_ms_feat[(h_high * width + w_low) * num_embeds + ic];

    float v4 = 0;
    if (h_high < height && w_high < width)
      v4 = mc_ms_feat[(h_high * width + w_high) * num_embeds + ic];

    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
  }
};

template<int num_embeds> struct sampling<float, false, num_embeds>{
  static float __forceinline__ __device__ call(int width, int height, int scale_start_index, int ibatch_by_num_feat, int ic, const float* mc_ms_feat, const float2& loc){
    const int w_im = roundf(loc.x * width  - 0.5f);
    const int h_im = roundf(loc.y * height - 0.5f);
    const int value_offset  = (ibatch_by_num_feat + scale_start_index + h_im * width + w_im) * num_embeds;
    return mc_ms_feat[value_offset + ic];
  }
};

template<int num_embeds> struct sampling<half2, true, num_embeds>{
  static half2 __forceinline__ __device__ call(int width, int height, int scale_start_index, int ibatch_by_num_feat, int ic, const half* mc_ms_feat, const half2& loc){
    const half2 size = make_half2(width, height);
    const half2 pos  = __hfma2(loc, size, make_half2(-0.5f, -0.5f));
    mc_ms_feat += (ibatch_by_num_feat + scale_start_index) * num_embeds;

    const half2 one2(1, 1);
    const half2 low = h2floor(pos);
    const int w_low = low.x;
    const int h_low = low.y;
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    half2 ll = pos  - low;
    half2 hh = one2 - ll;

    half2 v1(0, 0);
    if (h_low >= 0 && w_low >= 0)
      v1 = *((half2*)(mc_ms_feat + (h_low * width + w_low) * num_embeds) + ic);

    half2 v2(0, 0);
    if (h_low >= 0 && w_high < width)
      v2 = *((half2*)(mc_ms_feat + (h_low * width + w_high) * num_embeds) + ic);

    half2 v3(0, 0);
    if (h_high < height && w_low >= 0)
      v3 = *((half2*)(mc_ms_feat + (h_high * width + w_low) * num_embeds) + ic);

    half2 v4(0, 0);
    if (h_high < height && w_high < width)
      v4 = *((half2*)(mc_ms_feat + (h_high * width + w_high) * num_embeds) + ic);

    const half w1 = hh.y * hh.x, w2 = hh.y * ll.x, w3 = ll.y * hh.x, w4 = ll.y * ll.x;
    return make_half2(w1, w1) * v1 + make_half2(w2, w2) * v2 + make_half2(w3, w3) * v3 + make_half2(w4, w4) * v4;
  }
};

template<int num_embeds> struct sampling<half2, false, num_embeds>{
  static half2 __forceinline__ __device__ call(int width, int height, int scale_start_index, int ibatch_by_num_feat, int ic, const half* mc_ms_feat, const half2& loc){
    const int posx = roundf(width  * float(loc.x) - 0.5f);
    const int posy = roundf(height * float(loc.y) - 0.5f);
    const int value_offset  = (ibatch_by_num_feat + scale_start_index + posy * width + posx) * num_embeds;
    return *(reinterpret_cast<const half2*>(mc_ms_feat + value_offset) + ic);
  }
};

template<typename index_t, bool bilinear, int num_anchors, int group_size, int num_scale, int num_groups, int num_embeds, int num_warps, int chunk_size>
__global__ void deformable_aggregation_nearest_kernel_fp32(
    float* output,
    const float* mc_ms_feat,
    const index_t* spatial_shape,
    const index_t* scale_start_index,
    const float2* sample_location,
    const float* weights,
    const int batch_size,
    const int num_feat,
    const int num_pts,
    const int num_cams,
    const int points_by_cams
) {
    const int anchor_index = blockIdx.y;
    const int iwarp = threadIdx.y;
    const int ilane = threadIdx.x;
    const int ichunk = blockIdx.x;

    constexpr int64_t pb = 1.0 / num_anchors * (1ll << 32) + 1;
    const int32_t ibatch = (anchor_index * pb) >> 32;
    __shared__ float2 cached_locations[chunk_size];

    constexpr int points_per_warp = (chunk_size + num_warps - 1) / num_warps;
    const int warp_base_points_index = iwarp * points_per_warp + ichunk * chunk_size;

    // pre load locations
    for(int ipoints_by_cams = warp_base_points_index + ilane; ipoints_by_cams < points_by_cams && ipoints_by_cams < warp_base_points_index + points_per_warp; ipoints_by_cams += warpSize){
      const int loc_offset = anchor_index * points_by_cams + ipoints_by_cams;
      cpasync_global_to_shared_8byte(cached_locations + ipoints_by_cams - ichunk * chunk_size, sample_location + loc_offset);
    }
    cpasync_wait();
    __syncwarp();

    for(int ipoints_by_cams = warp_base_points_index; ipoints_by_cams < points_by_cams && ipoints_by_cams < warp_base_points_index + points_per_warp; ++ipoints_by_cams){
      const int loc_offset = anchor_index * points_by_cams + ipoints_by_cams;
      const float2 loc = cached_locations[ipoints_by_cams - ichunk * chunk_size];
      // Require a positively valid location rather than detecting an invalid one.
      // Written as a rejected-unless-proven-valid test because every comparison
      // against NaN is false: the previous form (loc.x <= 0 || ... ) tripped on
      // nothing for a NaN, so the location was sampled and accumulated, leaving
      // this anchor's whole embedding non-finite. A NaN projection is exactly as
      // invalid as the +/-inf one it is produced alongside - both come from a
      // keypoint behind the camera - and must be skipped the same way.
      if(!(loc.x > 0 && loc.x < 1 && loc.y > 0 && loc.y < 1)) continue;

      const int32_t ipoints = ipoints_by_cams / num_cams;
      const int32_t icamera = ipoints_by_cams % num_cams;
      float* offsetd_output = output + anchor_index * num_embeds;

      for(int ic = threadIdx.x; ic < num_embeds; ic += warpSize){
        float feature_sum = 0.0f;
        int weight_offset = (loc_offset * num_scale) * num_groups + ic / group_size;
        for(int iscale = 0; iscale < num_scale; ++iscale, weight_offset += num_groups){
          int cam_scale_index = icamera * num_scale + iscale;
          const int h = spatial_shape[cam_scale_index * 2 + 0];
          const int w = spatial_shape[cam_scale_index * 2 + 1];
          float value = sampling<float, bilinear, num_embeds>::call(w, h, scale_start_index[cam_scale_index], ibatch * num_feat, ic, mc_ms_feat, loc);
          feature_sum += value * weights[weight_offset];
        }
        atomicAdd(offsetd_output + ic, feature_sum);
      }
    }
}

template<typename index_t, bool bilinear, int num_anchors, int group_size, int num_scale, int num_groups, int num_embeds, int num_warps, int chunk_size>
__global__ void deformable_aggregation_nearest_kernel_fp16(
    half* output,
    const half* mc_ms_feat,
    const index_t* spatial_shape,
    const index_t* scale_start_index,
    const half2* sample_location,
    const half* weights,
    const int batch_size,
    const int num_feat,
    const int num_pts,
    const int num_cams,
    const int points_by_cams
) {
    const int anchor_index = blockIdx.y;
    const int iwarp = threadIdx.y;
    const int ilane = threadIdx.x;
    const int ichunk = blockIdx.x;

    constexpr int64_t pb = 1.0 / num_anchors * (1ll << 32) + 1;
    const int64_t ibatch = (anchor_index * pb) >> 32;
    __shared__ half2 cached_locations[chunk_size];

    constexpr int points_per_warp = (chunk_size + num_warps - 1) / num_warps;
    const int warp_base_points_index = iwarp * points_per_warp + ichunk * chunk_size;

    // pre load locations
    for(int ipoints_by_cams = warp_base_points_index + ilane; ipoints_by_cams < points_by_cams && ipoints_by_cams < warp_base_points_index + points_per_warp; ipoints_by_cams += warpSize){
      const int loc_offset = anchor_index * points_by_cams + ipoints_by_cams;
      cpasync_global_to_shared_4byte(cached_locations + ipoints_by_cams - ichunk * chunk_size, sample_location + loc_offset);
    }
    cpasync_wait();
    __syncwarp();

    for(int ipoints_by_cams = warp_base_points_index; ipoints_by_cams < points_by_cams && ipoints_by_cams < warp_base_points_index + points_per_warp; ++ipoints_by_cams){
      const int loc_offset = anchor_index * points_by_cams + ipoints_by_cams;
      const half2 loc = cached_locations[ipoints_by_cams - ichunk * chunk_size];
      // See the fp32 kernel above: rejected unless proven valid, so that a NaN -
      // which satisfies no comparison - is skipped rather than accumulated.
      if(!(__hgt(loc.x, 0) && __hlt(loc.x, 1) && __hgt(loc.y, 0) && __hlt(loc.y, 1))) continue;

      const int32_t ipoints = ipoints_by_cams / num_cams;
      const int32_t icamera = ipoints_by_cams % num_cams;
      half2* offsetd_output = reinterpret_cast<half2*>(output + anchor_index * num_embeds);

      for(int ic = ilane; ic < num_embeds / 2; ic += warpSize){
        half2 csum = make_half2(0, 0);
        const half* offsetd_weight = weights + (loc_offset * num_scale) * num_groups + (ic * 2) / group_size;
        for(int iscale = 0; iscale < num_scale; ++iscale, offsetd_weight += num_groups){
          int cam_scale_index = icamera * num_scale + iscale;
          const int h = spatial_shape[cam_scale_index * 2];
          const int w = spatial_shape[cam_scale_index * 2 + 1];
          half2 t2 = sampling<half2, bilinear, num_embeds>::call(w, h, scale_start_index[cam_scale_index], ibatch * num_feat, ic, mc_ms_feat, loc);
          csum = __hfma2(t2, make_half2(*offsetd_weight, *offsetd_weight), csum);
        }
        atomicAdd(offsetd_output + ic, csum);
      }
    }
}

template<typename feature_t, typename index_t, bool bilinear, int num_anchors, int num_embeds, int num_scale, int num_groups, int num_pts>
struct deformable_aggregation_impl{};

template<typename index_t, bool bilinear, int num_anchors, int num_embeds, int num_scale, int num_groups, int num_pts>
struct deformable_aggregation_impl<float, index_t, bilinear, num_anchors, num_embeds, num_scale, num_groups, num_pts>{

  template<int num_warps, int chunk_size>
  static void call_implement(
      float* output,
      const float* mc_ms_feat,
      const index_t* spatial_shape,
      const index_t* scale_start_index,
      const float* sample_location,
      const float* weights,
      int batch_size,
      int num_cams,
      int num_feat,
      cudaStream_t stream
  ){
    const int points_by_cams = num_pts * num_cams;
    const dim3 blocks(32, num_warps);
    const dim3 grids((points_by_cams + chunk_size - 1) / chunk_size, batch_size * num_anchors);
    constexpr int group_size = num_embeds / num_groups;
    static_assert(num_embeds % num_groups == 0);
    deformable_aggregation_nearest_kernel_fp32<index_t, bilinear, num_anchors, group_size, num_scale, num_groups, num_embeds, num_warps, chunk_size> <<<grids, blocks, 0, stream>>>(
        output, mc_ms_feat, spatial_shape, scale_start_index, (const float2*)sample_location, weights,
        batch_size, num_feat, num_pts, num_cams, points_by_cams
    );
  }

  static void call(
      float* output,
      const float* mc_ms_feat,
      const index_t* spatial_shape,
      const index_t* scale_start_index,
      const float* sample_location,
      const float* weights,
      int batch_size,
      int num_cams,
      int num_feat,
      cudaStream_t stream
  ){
    if constexpr(bilinear){
      if constexpr(num_pts < 32){
        call_implement<8, 32>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream);
      }else{
        call_implement<16, 256>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream);
      }
    }else{
        if constexpr(num_pts < 32){
            switch(num_cams){
                case 1: call_implement<8, num_pts * 1>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream); break;
                case 6: call_implement<8, num_pts * 6>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream); break;
            }
        }else{
            call_implement<16, 256>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream);
        }
    }
  }
};

template<typename index_t, bool bilinear, int num_anchors, int num_embeds, int num_scale, int num_groups, int num_pts>
struct deformable_aggregation_impl<half, index_t, bilinear, num_anchors, num_embeds, num_scale, num_groups, num_pts>{

  template<int num_warps, int chunk_size>
  static void call_implement(
      half* output,
      const half* mc_ms_feat,
      const index_t* spatial_shape,
      const index_t* scale_start_index,
      const half* sample_location,
      const half* weights,
      int batch_size,
      int num_cams,
      int num_feat,
      cudaStream_t stream
  ){
    const int points_by_cams = num_pts * num_cams;
    const dim3 blocks(32, num_warps);
    const dim3 grids((points_by_cams + chunk_size - 1) / chunk_size, batch_size * num_anchors);
    constexpr int group_size = num_embeds / num_groups;
    static_assert(num_embeds % num_groups == 0);
    deformable_aggregation_nearest_kernel_fp16<index_t, bilinear, num_anchors, group_size, num_scale, num_groups, num_embeds, num_warps, chunk_size> <<<grids, blocks, 0, stream>>>(
        output, mc_ms_feat, spatial_shape, scale_start_index, (const half2*)sample_location, weights,
        batch_size, num_feat, num_pts, num_cams, points_by_cams
    );
  }

  static void call(
      half* output,
      const half* mc_ms_feat,
      const index_t* spatial_shape,
      const index_t* scale_start_index,
      const half* sample_location,
      const half* weights,
      int batch_size,
      int num_cams,
      int num_feat,
      cudaStream_t stream
  ){
    if constexpr(bilinear){
      if constexpr(num_pts < 32){
        call_implement<8, 64>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream);
      }else{
        call_implement<8, 128>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream);
      }
    }else{
        if constexpr(num_pts < 32){
            switch(num_cams){
                case 1: call_implement<4, num_pts * 1>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream); break;
                case 6: call_implement<4, num_pts * 6>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream); break;
            }
        }else{
            call_implement<16, 256>(output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size, num_cams, num_feat, stream);
        }
    }
  }
};

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
){
    if(num_anchors == 900 && num_pts == 13 && num_embeds == 256 && num_scale == 4 && num_groups == 8){
        deformable_aggregation_impl<feature_t, index_t, bilinear, 900, 256, 4, 8, 13>::call(
            output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
            batch_size, num_cams, num_feat,
            static_cast<cudaStream_t>(stream)
        );
    }else if(num_anchors == 100 && num_pts == 300 && num_embeds == 256 && num_scale == 4 && num_groups == 8){
        deformable_aggregation_impl<feature_t, index_t, bilinear, 100, 256, 4, 8, 300>::call(
            output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
            batch_size, num_cams, num_feat,
            static_cast<cudaStream_t>(stream)
        );
    }else{
        fprintf(stderr, "[deformable_aggregation] Unsupported format with num_anchors[%d], num_cams[%d], num_embeds[%d], num_scale[%d], num_groups[%d], num_pts[%d]\n",
            num_anchors, num_cams, num_embeds, num_scale, num_groups, num_pts
        );
    }
}

template void deformable_aggregation<float, int64_t, true>(float*, const float*, const int64_t*, const int64_t*, const float*, const float*, int, int, int, int, int, int, int, int, void*);
template void deformable_aggregation<half, int64_t, true>(half*, const half*, const int64_t*, const int64_t*, const half*, const half*, int, int, int, int, int, int, int, int, void*);
template void deformable_aggregation<float, int32_t, true>(float*, const float*, const int32_t*, const int32_t*, const float*, const float*, int, int, int, int, int, int, int, int, void*);
template void deformable_aggregation<half, int32_t, true>(half*, const half*, const int32_t*, const int32_t*, const half*, const half*, int, int, int, int, int, int, int, int, void*);

template void deformable_aggregation<float, int64_t, false>(float*, const float*, const int64_t*, const int64_t*, const float*, const float*, int, int, int, int, int, int, int, int, void*);
template void deformable_aggregation<half, int64_t, false>(half*, const half*, const int64_t*, const int64_t*, const half*, const half*, int, int, int, int, int, int, int, int, void*);
template void deformable_aggregation<float, int32_t, false>(float*, const float*, const int32_t*, const int32_t*, const float*, const float*, int, int, int, int, int, int, int, int, void*);
template void deformable_aggregation<half, int32_t, false>(half*, const half*, const int32_t*, const int32_t*, const half*, const half*, int, int, int, int, int, int, int, int, void*);
