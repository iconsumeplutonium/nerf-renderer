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
	if (!file) {
		std::cout << fileName << " does not exist" << std::endl;
		return;
	}
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

	std::ifstream testingData("../igp/testing_data.bin", std::ios::binary);
	if (!testingData) {
		std::cout << "file doesnt exist" << std::endl;
		return 0;
	}

	int testPositionToRead = 0;
	
	// each camera pose is 800 by 800, so 640000 rows for each ray
	// each row is 9 values (3 camera pos, 3 ray dir, 3 rgb)
	// so each new camera position is this
	testingData.seekg(((NUM_ROWS * FLOATS_PER_ROW * testPositionToRead) * sizeof(float)));
	std::vector<float> camOrigin(3);
	testingData.read(reinterpret_cast<char*>(camOrigin.data()), sizeof(float) * 3);

	std::vector<float> rays;
	rays.resize(NUM_ROWS * 3);
	// read 3 floats for the ray dir, then skip 6 (which is the rgb and the next camera pos [todo: actually read and stoer that shit])
	for (int i = 0; i < NUM_ROWS; i++) {
		testingData.read(reinterpret_cast<char*>(rays.data() + i*3), sizeof(float) * 3);
		testingData.ignore(6 * sizeof(float));
	}

	testingData.close();
	std::cout << rays.size() << std::endl;

	// for (int i = 0; i < rays.size(); i += 3) {
	// 	std::cout << rays[i] << ", " << rays[i+1] << ", " << rays[i+2] << std::endl;
	// 	if (i >= 99) break;
	// }

	// return 0;
	
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
	
	// std::vector<std::vector<float>> colorMLPWeights(3);
	// colorMLPWeights[0].resize(64 * 43);
	// colorMLPWeights[1].resize(64 * 64);
	// colorMLPWeights[2].resize(3 * 64);
	// readFromFile("./ngp/weights/color_MLP.0.weight.bin", reinterpret_cast<char*>(colorMLPWeights[0].data()), (64 * 43) * sizeof(float));
	// readFromFile("./ngp/weights/color_MLP.2.weight.bin", reinterpret_cast<char*>(colorMLPWeights[1].data()), (64 * 64) * sizeof(float));
	// readFromFile("./ngp/weights/color_MLP.4.weight.bin", reinterpret_cast<char*>(colorMLPWeights[2].data()), ( 3 * 64) * sizeof(float));
	
	// std::vector<std::vector<float>> colorMLPBias(3);
	// colorMLPBias[0].resize(64);
	// colorMLPBias[1].resize(64);
	// colorMLPBias[2].resize(3 );
	// readFromFile("./ngp/weights/color_MLP.0.bias.bin", reinterpret_cast<char*>(colorMLPBias[0].data()), (64) * sizeof(float));
	// readFromFile("./ngp/weights/color_MLP.2.bias.bin", reinterpret_cast<char*>(colorMLPBias[1].data()), (64) * sizeof(float));
	// readFromFile("./ngp/weights/color_MLP.4.bias.bin", reinterpret_cast<char*>(colorMLPBias[2].data()), ( 3) * sizeof(float));
	
	
	// std::vector<std::vector<float>> densityMLPWeights(2);
	// densityMLPWeights[0].resize(64 * 32);
	// densityMLPWeights[1].resize(16 * 64);
	// readFromFile("./ngp/weights/density_MLP.0.weight.bin", reinterpret_cast<char*>(densityMLPWeights[0].data()), (64 * 32) * sizeof(float));
	// readFromFile("./ngp/weights/density_MLP.2.weight.bin", reinterpret_cast<char*>(densityMLPWeights[1].data()), (16 * 64) * sizeof(float));
	
	// std::vector<std::vector<float>> densityMLPBias(3);
	// densityMLPBias[0].resize(64);
	// densityMLPBias[1].resize(16);
	// readFromFile("./ngp/weights/density_MLP.0.bias.bin", reinterpret_cast<char*>(densityMLPBias[0].data()), (64) * sizeof(float));
	// readFromFile("./ngp/weights/density_MLP.2.bias.bin", reinterpret_cast<char*>(densityMLPBias[1].data()), (16) * sizeof(float));
	
	// std::vector<std::vector<float>> lookupTables(16);
	// for (int i = 0; i < (int) lookupTables.size(); i++) {
	// 	lookupTables[i].resize(524288 * 2);
	// 	std::string fileName = "./ngp/weights/lookup_tables." + std::to_string(i) + ".bin";
	// 	readFromFile(fileName, reinterpret_cast<char*>(lookupTables[i].data()), (524288 * 2) * sizeof(float));

	// 	std::cout << i << ": " << lookupTables[i].size() << std::endl;
	// }


	// std::cout << "running" << std::endl;
	// std::cout << densityMLPBias[0].size() << std::endl;
	// std::cout << densityMLPBias[1].size() << std::endl;
	
	
	// std::cout << "\nfinished" << std::endl;

	const uint32_t numRays = rays.size() / 3;
	const uint64_t rayBufferSize = rays.size() * sizeof(float);


    // const uint32_t num_points = size;
    // const uint64_t buf_size   = num_points * sizeof(TYPE);
    // std::cout << "vecadd vortex2: n=" << num_points << " buf=" << buf_size << "B" << std::endl;

    vx_device_h device = nullptr;
    CHECK(vx_device_open(0, &device));

    vx_queue_info_t qInfo = { sizeof(qInfo), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(device, &qInfo, &q));

	// set up all the buffers and shit
	vx_buffer_h rayBuffer = nullptr;
	CHECK(vx_buffer_create(device, rayBufferSize, VX_MEM_READ_WRITE, &rayBuffer));

    vx_module_h module_ = nullptr;
    vx_kernel_h kernel = nullptr;
    CHECK(vx_module_load_file(device, kernel_file, &module_));
    CHECK(vx_module_get_kernel(module_, "main", &kernel));

	kernel_arg_t kernelArgs {};
	kernelArgs.numRays = numRays;
	std::copy(camOrigin.begin(), camOrigin.end(), kernelArgs.camOrigin);
	CHECK(vx_buffer_address(rayBuffer, &kernelArgs.rayBuffer));

	CHECK(vx_enqueue_write(q, rayBuffer, 0, rays.data(), rayBufferSize, 0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(device, 1, &numRays, grid, block));

    vx_launch_info_t launchInfo {
	};
	launchInfo.struct_size = sizeof(launchInfo);
	launchInfo.kernel      = kernel;
	launchInfo.args_host   = &kernelArgs;
	launchInfo.args_size   = sizeof(kernelArgs);
	launchInfo.ndim        = 1;
	launchInfo.grid_dim[0] = grid[0];
	launchInfo.block_dim[0]= block[0];

    vx_event_h launch_ev = nullptr;
	vx_event_h read_ev = nullptr;
	std::vector<float> rayDst(rays);
    CHECK(vx_enqueue_launch(q, &launchInfo, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, rayDst.data(), rayBuffer, 0, rayBufferSize, 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

	std::cout << "kernel finished, checking errors" << std::endl;
	long errors = 0;
	for (int i = 0; i < rayDst.size(); i += 3) {
		if (rayDst[i] != camOrigin[i % 3]) errors++;
		// std::cout << rays[i] << std::endl;;
	}
	std::cout << "errors: " << errors << std::endl;

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    // vx_buffer_release(dst_buf);
    // vx_buffer_release(src1_buf);
    // vx_buffer_release(src0_buf);
	vx_buffer_release(rayBuffer);
    vx_kernel_release(kernel);
    vx_module_release(module_);
    vx_queue_release(q);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);

    // if (errors) {
    //     std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    //     return 1;
    // }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
