#!/usr/bin/env bash
set -ex

install_redis(){
	echo "Start to install redis"
	[[ $(uname) = Linux ]] || exit 1
	[[ $USER = root ]] || exit 1

	apt-get --assume-yes install redis
}

install_redis