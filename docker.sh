#!/bin/bash

IMAGE_NAME=ilmerkulov/als_airflow:argparse

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME