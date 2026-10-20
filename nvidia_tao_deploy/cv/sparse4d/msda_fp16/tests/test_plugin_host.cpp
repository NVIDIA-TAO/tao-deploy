// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
// Host-only ABI regressions. No CUDA context or GPU is required.
#include "deformable_aggregation_plugin.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "Failed at line " << __LINE__ << ": " << #condition << std::endl; \
    std::exit(1); } } while (false)

int main()
{
    Deformableaggregationplugin_TrtIPluginV2DynamicExtCreator creator;
    CHECK(std::strcmp(creator.getPluginName(), "MSDA") == 0);
    CHECK(std::strcmp(creator.getPluginVersion(), "1") == 0);
    CHECK(getPluginRegistry()->getPluginCreator("MSDA", "1", "") != nullptr);
    creator.setPluginNamespace("test");
    int cameras = 4;
    PluginField field{"num_cams", &cameras, PluginFieldType::kINT32, 1};
    PluginFieldCollection fields{1, &field};
    auto* plugin = creator.createPlugin("probe", &fields);
    CHECK(plugin != nullptr);
    CHECK(std::strcmp(plugin->getPluginNamespace(), "test") == 0);
    std::vector<char> data(plugin->getSerializationSize());
    plugin->serialize(data.data());
    int restored = 0;
    std::memcpy(&restored, data.data(), sizeof(restored));
    CHECK(restored == cameras);
    auto* copy = creator.deserializePlugin("probe", data.data(), data.size());
    CHECK(copy != nullptr);
    auto* clone = copy->clone();
    CHECK(std::strcmp(clone->getPluginNamespace(), "test") == 0);
    std::vector<char> cloned(data.size());
    clone->serialize(cloned.data());
    CHECK(cloned == data);
    CHECK(creator.deserializePlugin("bad", nullptr, data.size()) == nullptr);
    CHECK(creator.deserializePlugin("bad", data.data(), data.size() - 1) == nullptr);
    CHECK(creator.deserializePlugin("bad", data.data(), data.size() + 1) == nullptr);
    cameras = 0;
    CHECK(creator.createPlugin("bad", &fields) == nullptr);
    CHECK(creator.deserializePlugin("bad", &cameras, sizeof(cameras)) == nullptr);
    cameras = 4;
    field.length = 2;
    CHECK(creator.createPlugin("bad", &fields) == nullptr);
    field.length = 1;
    field.data = nullptr;
    CHECK(creator.createPlugin("bad", &fields) == nullptr);

    PluginTensorDesc desc[6]{};
    for (auto precision : {DataType::kFLOAT, DataType::kHALF})
    {
        for (auto index : {DataType::kINT32, DataType::kINT64})
        {
            for (int pos = 0; pos < 6; ++pos)
            {
                desc[pos].type = pos == 1 || pos == 2 ? index : precision;
                desc[pos].format = TensorFormat::kLINEAR;
                CHECK(plugin->supportsFormatCombination(pos, desc, 5, 1));
            }
            desc[2].type = DataType::kFLOAT;
            CHECK(!plugin->supportsFormatCombination(2, desc, 5, 1));
            desc[2].type = index;
            desc[4].type = DataType::kINT32;
            CHECK(!plugin->supportsFormatCombination(4, desc, 5, 1));
        }
    }
    CHECK(!plugin->supportsFormatCombination(-1, desc, 5, 1));
    CHECK(!plugin->supportsFormatCombination(6, desc, 5, 1));
    CHECK(!plugin->supportsFormatCombination(0, desc, 4, 1));
    const Dims shapes[] = {{3, {1, 96, 256}}, {3, {6, 4, 2}}, {2, {6, 4}},
                          {5, {1, 900, 13, 6, 2}}, {6, {1, 900, 13, 6, 4, 8}},
                          {3, {1, 900, 256}}};
    for (int i = 0; i < 6; ++i)
    {
        desc[i].dims = shapes[i];
        desc[i].type = i == 1 || i == 2 ? DataType::kINT32 : DataType::kFLOAT;
    }
    // Invalid shapes must return before any CUDA call or buffer access.
    for (int tensor = 0; tensor < 6; ++tensor)
    {
        --desc[tensor].dims.nbDims;
        CHECK(plugin->enqueue(desc, desc + 5, nullptr, nullptr, nullptr, nullptr) != 0);
        desc[tensor].dims = shapes[tensor];
        for (int dim = 0; dim < shapes[tensor].nbDims; ++dim)
        {
            if (tensor == 0 && dim == 1) // Feature length is intentionally variable.
                desc[tensor].dims.d[dim] = 0;
            else
                ++desc[tensor].dims.d[dim];
            CHECK(plugin->enqueue(desc, desc + 5, nullptr, nullptr, nullptr, nullptr) != 0);
            desc[tensor].dims = shapes[tensor];
        }
    }
    clone->destroy();
    copy->destroy();
    plugin->destroy();
    std::cout << "MSDA registration, serialization, format, and layout checks passed" << std::endl;
}
