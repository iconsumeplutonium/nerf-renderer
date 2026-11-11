VORTEX_HOME ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../vortex)

override ROOT_DIR := $(VORTEX_HOME)/build
include $(ROOT_DIR)/config.mk

PROJECT := nerf

SRC_DIR := $(CURDIR)
SRCS    := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n64

KERNEL_LIB := vortex2

include $(ROOT_DIR)/tests/regression/common.mk