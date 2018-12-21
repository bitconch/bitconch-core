NVCC:=nvcc
GPU_ARCH:=compute_35
GPU_CFLAGS:=--gpu-code=sm_37,sm_50,sm_61,sm_70,$(GPU_ARCH) --gpu-architecture=$(GPU_ARCH)
CFLAGS_release:=--ptxas-options=-v $(GPU_CFLAGS) -O3 -Xcompiler "-Wall -Werror -fPIC -Wno-strict-aliasing"
CFLAGS_debug:=$(CFLAGS_release) -g
CFLAGS:=$(CFLAGS_$V)
