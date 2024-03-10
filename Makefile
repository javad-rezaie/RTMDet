#---------------------HOMAI------------------------#
# Created on Sun Mar 10 2024
#
# Copyright (c) 2024 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#---------------------HOMAI------------------------#

docker-build-mmdetection:
	DOCKER_BUILDKIT=1 docker build \
	-f docker/mmdetection.Dockerfile \
	-t mmdetection .

docker-pull-mmdetection:
	docker pull homai/openmmlab:pytorch2.1.1-cuda12.1-cudnn8-mmdetection3.3.0
	docker image tag homai/openmmlab:pytorch2.1.1-cuda12.1-cudnn8-mmdetection3.3.0 mmdetection