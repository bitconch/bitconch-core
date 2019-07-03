#!/usr/bin/env bash
set -ex


install_nodejs(){
	echo "Start to install nodejs10"
	curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
	apt-get install -y nodejs
	node --version
	npm --version
}

install_nodejs