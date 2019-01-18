ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	cd rustcode/hello && cargo build --release --target x86_64-unknown-linux-gnu
# 	cp rustcode/hello/target/x86_64-unknown-linux-gnu/release/libhello.a lib/x86_64-unknown-linux-gnu/libhello.a
#   copy a for static lib build, copy .so for dynamic lib build
	cp rustcode/hello/target/x86_64-unknown-linux-gnu/release/libhello.so lib/x86_64-unknown-linux-gnu/libhello.so
	go build -ldflags="-r $(ROOT_DIR)lib/x86_64-unknown-linux-gnu" main.go

run: build
	./main
