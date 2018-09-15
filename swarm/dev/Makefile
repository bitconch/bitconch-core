.PHONY: build cluster test

default: build

build:
	go build -o bin/swarm    github.com/Bitconch/BUS/cmd/swarm
	go build -o bin/geth     github.com/Bitconch/BUS/cmd/geth
	go build -o bin/bootnode github.com/Bitconch/BUS/cmd/bootnode

cluster: build
	scripts/boot-cluster.sh

test:
	go test -v github.com/Bitconch/BUS/swarm/...
