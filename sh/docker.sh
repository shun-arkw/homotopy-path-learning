#!/usr/bin/env bash
set -e

# ====== paths you may want to customize ======
DATASET_DIRS="$HOME/dataset"
DATA_DIRS="$HOME/data"

# ====== docker build config ======
DOCKERFILE_NAME="Dockerfile.sage-julia-alpha"
TARGET_STAGE="torch-2.3.0"
TAG="torch-2.3.0-sage-julia-sysimage"

# （推奨）ホストの ~/.julia をマウントしない：sysimage を使うため再現性と速度を優先
# どうしても共有したい場合は、--with-host-julia オプションで明示的に有効化するようにします
HOST_JULIA_DIR="$HOME/.julia"
CONTAINER_JULIA_DIR="/home/user/.julia"

build() {
  export DOCKER_BUILDKIT=1
  docker build . \
    -f docker/"$DOCKERFILE_NAME" \
    --target "$TARGET_STAGE" \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    -t "$TAG"
}

shell() {
  if [[ "$1" == "--with-host-julia" ]]; then
    JULIA_MOUNT=(-v "$HOST_JULIA_DIR":"$CONTAINER_JULIA_DIR")
  else
    JULIA_MOUNT=()  # デフォルトはマウントしない
  fi

  docker run --rm --gpus all --shm-size=16g -it \
    -v "$(pwd)":/app \
    -v "$DATASET_DIRS":/dataset \
    -v "$DATA_DIRS":/data \
    "${JULIA_MOUNT[@]}" \
    --name homotopy-continuation \
    "$TAG" /bin/bash
}

root() {
  docker run -p 8888:8888 --rm --gpus all --shm-size=16g --user 0:0 -it \
    -v "$(pwd)":/app \
    -v "$DATASET_DIRS":/dataset \
    -v "$DATA_DIRS":/data \
    "$TAG" /bin/bash
}

help() {
  echo "usage: bash docker.sh [build|shell|root|help] [--with-host-julia]"
}

case "$1" in
  build) build ;;
  shell) shift; shell "$@" ;;
  root)  root  ;;
  help|"") help ;;
  *) help ;;
esac
