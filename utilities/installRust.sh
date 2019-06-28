#!/usr/bin/env bash
set -ex

install_rust() {
	echo "Start to install rust"
	curl https://sh.rustup.rs -sSf | sh -s -- -y
	source $HOME/.cargo/env
	rustup -V
	rustc -V
	cargo -V
	echo "rust installed"
}

install_rust
