#!/bin/bash
# Deployment script for RunPod Video Pipeline

set -e

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_USERNAME="${DOCKER_USERNAME:-your-username}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Handler configurations
declare -A HANDLERS=(
    ["flux"]="handlers/flux"
    ["ltx_video"]="handlers/ltx_video"
    ["wan21"]="handlers/wan21"
    ["codeformer"]="handlers/codeformer"
    ["rife"]="handlers/rife"
    ["realesrgan"]="handlers/realesrgan"
)

build_handler() {
    local name=$1
    local path=$2
    local image="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/vidpipe-${name}:${IMAGE_TAG}"
    
    echo_info "Building ${name}..."
    
    # Copy shared utilities to handler directory
    cp -r shared "${path}/"
    
    docker build -t "${image}" "${path}"
    
    # Cleanup
    rm -rf "${path}/shared"
    
    echo_info "Built: ${image}"
}

push_handler() {
    local name=$1
    local image="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/vidpipe-${name}:${IMAGE_TAG}"
    
    echo_info "Pushing ${name}..."
    docker push "${image}"
    echo_info "Pushed: ${image}"
}

build_all() {
    echo_info "Building all handlers..."
    for name in "${!HANDLERS[@]}"; do
        build_handler "$name" "${HANDLERS[$name]}"
    done
    echo_info "All handlers built!"
}

push_all() {
    echo_info "Pushing all handlers..."
    for name in "${!HANDLERS[@]}"; do
        push_handler "$name"
    done
    echo_info "All handlers pushed!"
}

build_orchestrator() {
    local image="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/vidpipe-orchestrator:${IMAGE_TAG}"
    
    echo_info "Building orchestrator..."
    
    # Copy shared utilities
    cp -r shared orchestrator/
    
    docker build -t "${image}" orchestrator/
    
    # Cleanup
    rm -rf orchestrator/shared
    
    echo_info "Built: ${image}"
}

push_orchestrator() {
    local image="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/vidpipe-orchestrator:${IMAGE_TAG}"
    
    echo_info "Pushing orchestrator..."
    docker push "${image}"
    echo_info "Pushed: ${image}"
}

show_usage() {
    echo "Usage: $0 <command> [handler]"
    echo ""
    echo "Commands:"
    echo "  build [handler]     Build Docker image(s)"
    echo "  push [handler]      Push image(s) to registry"
    echo "  deploy [handler]    Build and push image(s)"
    echo "  orchestrator        Build and push orchestrator"
    echo "  all                 Build and push everything"
    echo ""
    echo "Handlers: flux, ltx_video, wan21, codeformer, rife, realesrgan"
    echo ""
    echo "Environment variables:"
    echo "  DOCKER_REGISTRY     Docker registry (default: docker.io)"
    echo "  DOCKER_USERNAME     Docker Hub username"
    echo "  IMAGE_TAG           Image tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 build flux       Build FLUX handler"
    echo "  $0 deploy all       Build and push all handlers"
    echo "  $0 orchestrator     Build and push orchestrator"
}

# Main script
case "$1" in
    build)
        if [ "$2" == "all" ] || [ -z "$2" ]; then
            build_all
        elif [ -n "${HANDLERS[$2]}" ]; then
            build_handler "$2" "${HANDLERS[$2]}"
        else
            echo_error "Unknown handler: $2"
            exit 1
        fi
        ;;
    push)
        if [ "$2" == "all" ] || [ -z "$2" ]; then
            push_all
        elif [ -n "${HANDLERS[$2]}" ]; then
            push_handler "$2"
        else
            echo_error "Unknown handler: $2"
            exit 1
        fi
        ;;
    deploy)
        if [ "$2" == "all" ] || [ -z "$2" ]; then
            build_all
            push_all
        elif [ -n "${HANDLERS[$2]}" ]; then
            build_handler "$2" "${HANDLERS[$2]}"
            push_handler "$2"
        else
            echo_error "Unknown handler: $2"
            exit 1
        fi
        ;;
    orchestrator)
        build_orchestrator
        push_orchestrator
        ;;
    all)
        build_all
        push_all
        build_orchestrator
        push_orchestrator
        echo_info "All images deployed!"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
