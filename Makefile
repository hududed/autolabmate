# Variables
PLATFORM = linux/amd64
IMAGE_NAME = gcr.io/autolabmate-430603/streamlit-app
CONTAINER_NAME = streamlit
REGION = us-central1
SERVICE_NAME = autolabmate

# Targets
build-docker-containers:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME) .

run-docker-containers:
	docker run -p 8080:8080 $(IMAGE_NAME)

prune-docker-containers:
	docker system prune -a

build-and-push:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME) .
	docker push $(IMAGE_NAME)

deploy:
	gcloud run deploy $(SERVICE_NAME) --image $(IMAGE_NAME) --allow-unauthenticated --region $(REGION)

all: build-and-push deploy