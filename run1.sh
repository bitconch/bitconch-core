#!/usr/bin/env bash
#cd ./go/src/github.com/caesarchad/rustelo-rust/soros && cargo build --all && ./run.sh
python3 deploy.py
cd ~/go/src/github.com/bitconch/bus/vendor/rustelo-rust/buffett && ./demo/setup.sh && ./demo/leader.sh
