ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	cd rustcode/hello && cargo clean && cargo build --release --target x86_64-unknown-linux-gnu
	rm -rf lib/x86_64-unknown-linux-gnu
	mkdir -p lib/x86_64-unknown-linux-gnu
# 	cp rustcode/hello/target/x86_64-unknown-linux-gnu/release/libhello.a lib/x86_64-unknown-linux-gnu/libhello.a
#   copy a for static lib build, copy .so for dynamic lib build
	cp rustcode/hello/target/x86_64-unknown-linux-gnu/release/libhellolib.so lib/x86_64-unknown-linux-gnu/libhellolib.so
	go build -ldflags="-r $(ROOT_DIR)lib/x86_64-unknown-linux-gnu" -o bin/cli -v -x -a github.com/bitconch/bus/cdylib/cli

run: build
	./main
