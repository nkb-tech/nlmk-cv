#!/bin/bash

TZ="Europe/Moscow"
CUDA_VERSION="11.8.0"
OS_VERSION="22.04"
TRT_VERSION="8.6.1.6"
TORCH_VERSION="2.1.0"
DOCKER_IMAGE_VERSION="1.0.0"
DOCKER_IMAGE_NAME="nkbtech"

yellow=`tput setaf 3`
green=`tput setaf 2`
violet=`tput setaf 5`
reset_color=`tput sgr0`

ARCH="$(uname -m)"

function fail {
    printf '%s\n' "$1" >&2 ## Send message to stderr.
    exit "${2-1}" ## Return a code specified by $2, or 1 by default.
}

main () {

    if [ "$ARCH" = "x86_64" ]; then
        file="Dockerfile"
    else
        fail "There is no Dockerfile for ${yellow}${ARCH}${reset_color} arch"
    fi

    echo "Building image for ${yellow}${ARCH}${reset_color} arch. from Dockerfile: ${yellow}${file}${reset_color}"
    
    docker build . \
        --file ${file} \
        --tag ${ARCH}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION} \
        --build-arg TZ=${TZ} \
        --build-arg CUDA_VERSION=${CUDA_VERSION} \
        --build-arg OS_VERSION=${OS_VERSION} \
        --build-arg TRT_VERSION=${TRT_VERSION} \
        --build-arg TORCH_VERSION=${TORCH_VERSION}
}

main "$@"; exit;
