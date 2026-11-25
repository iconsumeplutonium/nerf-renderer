// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// vecadd — vortex2.h-native regression test.
//
// Async pattern: uploads are fire-and-forget; the launch produces an event;
// the dst readback gates on that event; the host waits once at the end.
// The per-queue worker serializes ops in FIFO order, so no inter-step host sync is needed.

#include <vortex2.h>
#include "common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <fstream>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t    size        = 16;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n words] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}

bool float_eq(float a, float b) {
    union fi { float f; int32_t i; };
    fi fa{a}, fb{b};
    return std::abs(fa.i - fb.i) <= 6;
}
} // namespace

void readFromFile(std::string fileName, char* destination, std::streamsize amountToRead) {
	std::ifstream file(fileName, std::ios::binary);
	file.read(destination, amountToRead);
	file.close();
}

const int WIDTH = 800;
const int HEIGHT = 800;
const int NUM_ROWS = WIDTH * HEIGHT;
const int FLOATS_PER_ROW = 9;

int main(int argc, char** argv) {
    parse_args(argc, argv);
    std::srand(50);

	// testing area for reading the file and stuff
	std::ifstream testingData("../igp/testing_data.bin", std::ios::binary);
	if (!testingData) {
		std::cout << "file doesnt exist" << std::endl;
		return 0;
	}

	int testPositionToRead = 0;
	
	// each camera pose is 800 by 800, so 640000 rows for each ray
	// each row is 9 values (3 camera pos, 3 ray dir, 3 rgb)
	// so each new camera position is this
	testingData.seekg((NUM_ROWS * FLOATS_PER_ROW * testPositionToRead) * sizeof(float));
	std::vector<float> poses;
	poses.resize(NUM_ROWS * FLOATS_PER_ROW);
	testingData.read(reinterpret_cast<char*>(poses.data()), sizeof(float) * NUM_ROWS * FLOATS_PER_ROW);
	testingData.close();
	std::cout << poses.size() << std::endl;
	
	// lookup_tables.0 (524288, 2)
	// lookup_tables.1 (524288, 2)
	// lookup_tables.10 (524288, 2)
	// lookup_tables.11 (524288, 2)
	// lookup_tables.12 (524288, 2)
	// lookup_tables.13 (524288, 2)
	// lookup_tables.14 (524288, 2)
	// lookup_tables.15 (524288, 2)
	// lookup_tables.2 (524288, 2)
	// lookup_tables.3 (524288, 2)
	// lookup_tables.4 (524288, 2)
	// lookup_tables.5 (524288, 2)
	// lookup_tables.6 (524288, 2)
	// lookup_tables.7 (524288, 2)
	// lookup_tables.8 (524288, 2)
	// lookup_tables.9 (524288, 2)
	// density_MLP.0.weight (64, 32)
	// density_MLP.0.bias (64,)
	// density_MLP.2.weight (16, 64)
	// density_MLP.2.bias (16,)
	// color_MLP.0.weight (64, 43)
	// color_MLP.0.bias (64,)
	// color_MLP.2.weight (64, 64)
	// color_MLP.2.bias (64,)
	// color_MLP.4.weight (3, 64)
	// color_MLP.4.bias (3,)
	
	std::vector<std::vector<float>> colorMLPWeights(3);
	colorMLPWeights[0].resize(64 * 43);
	colorMLPWeights[1].resize(64 * 64);
	colorMLPWeights[2].resize(3 * 64);
	readFromFile("./weights/color_MLP.0.weight", reinterpret_cast<char*>(colorMLPWeights[0].data()), (64 * 43) * sizeof(float));
	readFromFile("./weights/color_MLP.2.weight", reinterpret_cast<char*>(colorMLPWeights[1].data()), (64 * 64) * sizeof(float));
	readFromFile("./weights/color_MLP.4.weight", reinterpret_cast<char*>(colorMLPWeights[2].data()), ( 3 * 64) * sizeof(float));
	
	std::vector<std::vector<float>> colorMLPBias(3);
	colorMLPBias[0].resize(64);
	colorMLPBias[1].resize(64);
	colorMLPBias[2].resize(3 );
	readFromFile("./weights/color_MLP.0.bias", reinterpret_cast<char*>(colorMLPBias[0].data()), (64) * sizeof(float));
	readFromFile("./weights/color_MLP.2.bias", reinterpret_cast<char*>(colorMLPBias[1].data()), (64) * sizeof(float));
	readFromFile("./weights/color_MLP.4.bias", reinterpret_cast<char*>(colorMLPBias[2].data()), ( 3) * sizeof(float));
	
	
	std::vector<std::vector<float>> densityMLPWeights(2);
	densityMLPWeights[0].resize(64 * 32);
	densityMLPWeights[1].resize(16 * 64);
	readFromFile("./weights/density_MLP.0.weight", reinterpret_cast<char*>(densityMLPWeights[0].data()), (64 * 32) * sizeof(float));
	readFromFile("./weights/density_MLP.2.weight", reinterpret_cast<char*>(densityMLPWeights[1].data()), (16 * 64) * sizeof(float));
	
	std::vector<std::vector<float>> densityMLPBias(3);
	densityMLPBias[0].resize(64);
	densityMLPBias[1].resize(16);
	readFromFile("./weights/density_MLP.0.bias", reinterpret_cast<char*>(densityMLPBias[0].data()), (64) * sizeof(float));
	readFromFile("./weights/density_MLP.2.bias", reinterpret_cast<char*>(densityMLPBias[1].data()), (16) * sizeof(float));
	
	std::vector<std::vector<float>> lookupTables(16);
	for (int i = 0; i < 16; i++) {
		lookupTables[i].resize(524288 * 2);
		std::string fileName = "./weights/lookup_tables." + std::to_string(i) + ".bin";
		readFromFile(fileName, reinterpret_cast<char*>(lookupTables[i].data()), (524288 * 2) * sizeof(float));

		std::cout << i << ": " << lookupTables[i].size() << std::endl;
	}


	std::cout << "running" << std::endl;
	// std::cout << densityMLPBias[0].size() << std::endl;
	// std::cout << densityMLPBias[1].size() << std::endl;
	
	
	std::cout << "\nfinished" << std::endl;


	return 0;


    const uint32_t num_points = size;
    const uint64_t buf_size   = num_points * sizeof(TYPE);
    std::cout << "vecadd vortex2: n=" << num_points << " buf=" << buf_size << "B" << std::endl;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    vx_buffer_h src0_buf=nullptr, src1_buf=nullptr, dst_buf=nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &src0_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &src1_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &dst_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_points = num_points;
    CHECK(vx_buffer_address(src0_buf, &kernel_arg.src0_addr));
    CHECK(vx_buffer_address(src1_buf, &kernel_arg.src1_addr));
    CHECK(vx_buffer_address(dst_buf,  &kernel_arg.dst_addr));

    std::vector<TYPE> h_src0(num_points), h_src1(num_points), h_dst(num_points);
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src0[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
        h_src1[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
    }

    // ----- Async chain: 2 writes → launch → read → 1 wait -----
    // The kernel-args block is passed as a host blob — no args device buffer needed.
    CHECK(vx_enqueue_write(q, src0_buf, 0, h_src0.data(), buf_size, 0,nullptr,nullptr));
    CHECK(vx_enqueue_write(q, src1_buf, 0, h_src1.data(), buf_size, 0,nullptr,nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &num_points, grid, block));

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &kernel_arg;
    li.args_size   = sizeof(kernel_arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];

    vx_event_h launch_ev=nullptr, read_ev=nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size,
                          1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        TYPE ref = h_src0[i] + h_src1[i];
        if (!float_eq(h_dst[i], ref)) {
            if (errors < 16)
                std::printf("*** [%u] expected=%f actual=%f\n", i, ref, h_dst[i]);
            ++errors;
        }
    }

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(dst_buf);
    vx_buffer_release(src1_buf);
    vx_buffer_release(src0_buf);
    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
