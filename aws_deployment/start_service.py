#!/usr/bin/env python3

import boto3

# === AWS CONFIG ===
AWS_ACCESS_KEY_ID     = "AKIAUX2WC2TMHPEOOKLH"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION            = "us-east-2"

# === ECS CONFIG ===
CLUSTER_NAME = "streamlit-cluster"
SERVICE_NAME = "streamlit-task-service-uu9sz6hc"

def start_ecs_service():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    ecs = session.client("ecs")

    response = ecs.update_service(
        cluster=CLUSTER_NAME,
        service=SERVICE_NAME,
        desiredCount=1
    )

    print("✅ ECS service started (desired count set to 1)")
    print("Response:", response["service"]["desiredCount"])

if __name__ == "__main__":
    start_ecs_service()