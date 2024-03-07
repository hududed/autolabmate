build-docker-containers:
	DOCKER_DEFAULT_PLATFORM=linux/arm64 COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose build
run-docker-containers:
	DOCKER_DEFAULT_PLATFORM=linux/arm64 docker compose up
prune-docker-containers:
	docker system prune -a
