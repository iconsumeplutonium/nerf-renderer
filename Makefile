# this makefile was adapated from vortex/build/tests/regression/vecadd
# normally this variable points to ../../.., but now, vortex lives
# in ../vortex, so this gets the ahbsolute path to that forlder
VORTEX_HOME := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../vortex)

# ROOT_DIR is overwritten in common.mk, so this forces this definition to take precendence
override ROOT_DIR := $(VORTEX_HOME)/build

# config.mk has the env vars for simx and verilator and stuff i think
include $(ROOT_DIR)/config.mk

PROJECT := nerf
SRC_DIR := $(CURDIR)
SRCS    := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n64
KERNEL_LIB := vortex2

# this file has all the definitions needed to build the project
# ig uess its abstracted out of all the test make files into one shared file
# anyways, all the make targets live in this file, including run-simx
include $(ROOT_DIR)/tests/regression/common.mk