![alt text](https://github.com/caesarchad/BUS/blob/master/MVP/blaze_gpu/img/cuda-downloads-sc18.png "Nvidia GPU Roadmap")

# BLAZE CUDA
One of our latest tools in our cutting edge arsenal to help us to build the next generation blockchain.

## Supported GPU

| GPU Number     | Architecture |
| ------------- | ------------- |
| compute_35  | + Dynamic parallelism support  |
| compute_50, compute_52, and compute_53  | + Maxwell support  |
| compute_60, compute_61, and compute_62	  | + Pascal support  |
| compute_70 and compute_72 | + Volta support  |
| compute_75 | + Turing support  |

https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list


| GPU Feature List     | Features |
| ------------- | ------------- |
| sm_35  | + Dynamic parallelism support  |
| sm_50, sm_52, and sm_53  | + Maxwell support  |
| sm_60, sm_61, and sm_62	  | + Pascal support  |
| sm_70 and sm_72 | + Volta support  |
| sm_75 | + Turing support  |



## Building
After cloning this repo use the makefile in the root to build the tree
with nvcc in your path:

    export PATH=/usr/local/cuda/bin:$PATH
    make -j

The **make -j** specified that the **make** to run many recipes simultaneously. You could find more on https://www.gnu.org/software/make/manual/make.html#Makefile-Arguments.

This should generate the libraries:
* libcuda-crypt.a - ed25519 verify (used by leaders) and chacha (used by validators) cuda implementations
* libcpu-crypt.a - CPU chacha encryption implementation, used by replicators (storage miners)
* libJerasure.so, libgf\_complete.so - CPU erasure code library used for coding blob send

Copy those to the MVP repo:

    cp src/release/libcuda-crypt.a $MVP_ROOT/target/perf-libs
    cp src/cpu-crypt/release/libcpu-crypt.a $MVP_ROOT/target/perf-libs
    cp src/gf-complete/src/.libs/libgf_complete.so $MVP_ROOT/target/perf-libs
    cp src/jerasure/src/.libs/libJerasure.so $MVP_ROOT/target/perf-libs

Build Solana with the cuda & chacha features enabled:

    cd $MVP_ROOT
    go build --release -tags 'cuda'
