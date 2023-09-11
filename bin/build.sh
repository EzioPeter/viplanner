PKGROOT="$( realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"/../ )"
echo -e "\e[1;32m[build.sh]: Package root is '$PKGROOT'.\e[0m"

# Build image
VERSION=0.1.0
IMAGE=rslethz/viplanner:$VERSION
echo -e "\e[1;32m[build.sh]: Building image '$IMAGE'.\e[0m"

# NOTE: setting DOCKER_BUILDKIT=0 as bugfix because otherwise the gpu is not found during the build process even if default_runtime is nvidia
DOCKER_BUILDKIT=0 docker build -t "$IMAGE" -f "$PKGROOT/viplanner_ros/Dockerfile" "$PKGROOT/viplanner_ros"
