.PHONY: build

build:
	docker build -t future_generation .

run: build
    aws-vault exec private -- docker run -it -e AWS_REGION -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN -e AWS_SECURITY_TOKEN future_generation