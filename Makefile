# Variables
LOCAL_PLATFORM = linux/arm64/v8
CLOUD_PLATFORM = linux/amd64
IMAGE_NAME = gcr.io/autolabmate-430603/streamlit-app
CONTAINER_NAME = streamlit
REGION = us-central1
SERVICE_NAME = autolabmate
STAGING_SERVICE_NAME = autolabmate-staging

# Targets
build-docker-containers-local:
	docker build --platform $(LOCAL_PLATFORM) -t $(IMAGE_NAME) .

run-docker-containers-local:
	docker run --platform $(LOCAL_PLATFORM) -p 8080:8080 $(IMAGE_NAME)

build-docker-containers-cloud:
	docker build --platform $(CLOUD_PLATFORM) -t $(IMAGE_NAME) .

prune-docker-containers:
	docker system prune -a

build-and-push:
	docker build --platform $(CLOUD_PLATFORM) -t $(IMAGE_NAME) .
	docker push $(IMAGE_NAME)

deploy:
	gcloud run deploy $(SERVICE_NAME) --image $(IMAGE_NAME) --allow-unauthenticated --region $(REGION)

deploy-staging:
	gcloud run deploy $(STAGING_SERVICE_NAME) --image $(IMAGE_NAME) --allow-unauthenticated --region $(REGION)

delete-staging:
	gcloud run services delete $(STAGING_SERVICE_NAME) --region $(REGION) --quiet

all: build-and-push deploy

local: build-docker-containers-local run-docker-containers-local

staging: build-and-push deploy-staging