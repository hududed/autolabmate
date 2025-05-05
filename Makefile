# Variables
LOCAL_PLATFORM = linux/arm64/v8
CLOUD_PLATFORM = linux/amd64
IMAGE_NAME = gcr.io/autolabmate-430603/streamlit-app
CONTAINER_NAME = streamlit
REGION = us-east1
SERVICE_NAME = autolabmate
STAGING_SERVICE_NAME = autolabmate-staging
PROJECT_ID = autolabmate-430603

# Targets
auth-gcloud:
	gcloud auth login --quiet
	gcloud config set project $(PROJECT_ID)

build-docker-containers-local:
	docker build --no-cache --platform $(LOCAL_PLATFORM) -t $(IMAGE_NAME) .

run-docker-containers-local:
	docker run --platform $(LOCAL_PLATFORM) -p 8080:8080 $(IMAGE_NAME)

prune-docker-containers:
	docker system prune -a

build-and-push:
	docker build --no-cache --platform $(CLOUD_PLATFORM) -t $(IMAGE_NAME) .
	docker push $(IMAGE_NAME)

allow-unauthenticated:
	gcloud run services add-iam-policy-binding $(SERVICE_NAME) \
        --region $(REGION) \
        --member="allUsers" \
        --role="roles/run.invoker"

deploy:
	gcloud run services replace cloudrun-prod.yaml --region $(REGION)

allow-unauthenticated-staging:
	gcloud run services add-iam-policy-binding $(STAGING_SERVICE_NAME) \
        --region $(REGION) \
        --member="allUsers" \
        --role="roles/run.invoker"

deploy-staging:
	gcloud run services replace cloudrun-staging.yaml --region $(REGION)

delete-staging:
	gcloud run services delete $(STAGING_SERVICE_NAME) --region $(REGION) --quiet

merge-dev-to-main:
	git checkout main
	git merge dev
	git push origin main

production: build-and-push deploy allow-unauthenticated

local: build-docker-containers-local run-docker-containers-local

staging: build-and-push deploy-staging allow-unauthenticated-staging