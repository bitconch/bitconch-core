#!/usr/bin/env bats
load helpers/general

export SCRIPT_LOCATION="scripts/bitconchio_build.sh"
export TEST_LABEL="[bitconchio_build_ubuntu]"

[[ $ARCH == "Linux" ]] || exit 0 # Exit 0 is required for pipeline
[[ $NAME == "Ubuntu" ]] || exit 0 # Exit 0 is required for pipeline
( [[ $VERSION_ID == "18.04" ]] || [[ $VERSION_ID == "16.04" ]] ) || exit 0 # Exit 0 is required for pipeline

# A helper function is available to show output and status: `debug`

# Testing Root user
./tests/bash-bats/modules/root-user.sh
# Testing Options
./tests/bash-bats/modules/dep_script_options.sh
# Testing CMAKE
./tests/bash-bats/modules/cmake.sh
# Testing Clang
./tests/bash-bats/modules/clang.sh
# Testing MongoDB
./tests/bash-bats/modules/mongodb.sh

## Needed to load bitconchio_build_ files properly; it can be empty
@test "${TEST_LABEL} > General" {
    set_system_vars # Obtain current machine's resources and set the necessary variables (like JOBS, etc)

    [[ "$(echo ${VERSION_ID})" == "16.04" ]] && install-package build-essential WETRUN 1>/dev/null || install-package clang WETRUN 1>/dev/null
    run bash -c "printf \"y\n%.0s\" {1..100} | ./$SCRIPT_LOCATION -i /NEWPATH"
    [[ ! -z $(echo "${output}" | grep "Executing: make -j${JOBS}") ]] || exit
    [[ ! -z $(echo "${output}" | grep "Starting BITCONCHIO Dependency Install") ]] || exit
    [[ ! -z $(echo "${output}" | grep python.*) ]] || exit
    [[ ! -z $(echo "${output}" | grep make.*NOT.*found) ]] || exit
    [[ ! -z $(echo "${output}" | grep /NEWPATH.*/src/boost) ]] || exit
    [[ ! -z $(echo "${output}" | grep "make -j${CPU_CORES}") ]] || exit
    [[ ! -z $(echo "${output}" | grep " --with-iostreams --with-date_time") ]] || exit # BOOST
    if [[ "$(echo ${VERSION_ID})" == "18.04" ]]; then
        [[ ! -z $(echo "${output}" | grep llvm-4.0.*found) ]] || exit
    fi
    [[ -z $(echo "${output}" | grep "-   NOT found") ]] || exit
    [[ -z $(echo "${output}" | grep lcov.*found) ]] || exit
    [[ ! -z $(echo "${output}" | grep "BITCONCHIO has been successfully built") ]] || exit
    [[ "$(echo ${VERSION_ID})" == "16.04" ]] && apt autoremove build-essential -y || uninstall-package clang WETRUN
}
