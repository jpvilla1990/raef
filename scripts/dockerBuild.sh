source .env

REGISTRY_URL="ghcr.io"
IMAGE_NAME="raef_env"
GITHUB_USERNAME="jpvilla1990"
TAG="0.1.0"

echo $GITHUB_TOKEN | docker login ghcr.io -u jpvilla1990 --password-stdin

docker build -t $IMAGE_NAME:$TAG -f dockerfile.raef .

docker tag $IMAGE_NAME:$TAG $REGISTRY_URL/$GITHUB_USERNAME/$IMAGE_NAME:$TAG

docker push $REGISTRY_URL/$GITHUB_USERNAME/$IMAGE_NAME:$TAG