#!/usr/bin/env bash
set -ex
install_yarn(){
	echo "Start to install nodejs10"
	curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
	echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
	apt-get update -qq
	apt-get install -y yarn
	yarn --version
}

install_yarn