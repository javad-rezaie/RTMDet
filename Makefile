docker-build-mmdetection:
	DOCKER_BUILDKIT=1 docker build \
	-f docker/mmdetection.Dockerfile \
	-t mmdetection .