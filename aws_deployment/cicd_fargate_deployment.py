#!/usr/bin/env python3
"""
Deploy 'agrorisk' container to AWS ECS Fargate (public IP, no ALB).

Prereqs on your machine:
  - Docker installed and running
  - Python 3.10+ with boto3
  - Network access to AWS


  AWS ECS Fargate with a public IP (no EC2, no ALB). It:

        Builds your local Docker image

        Creates (or reuses) the ECR repo

        Logs in to ECR and pushes the image

        Ensures the IAM execution role exists

        Finds the default VPC and public subnets

        Creates/updates a Security Group opening ports (8000/8501/8502 by default)

        Registers an ECS Task Definition

        Creates/updates the ECS Service with assignPublicIp=ENABLED

        Waits for a running task and prints its public IP and URLs
"""

import base64
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

# =========================
# ==== USER CONFIG ========
# =========================
AWS_REGION               = "us-east-2"
AWS_ACCOUNT_ID           = "326061184216"

# ⚠️ Hardcoded credentials (avoid committing this file)
AWS_ACCESS_KEY_ID        = "AKIAUX2WC2TMHPEOOKLH"
AWS_SECRET_ACCESS_KEY    = "NV12joJlbLfWowxd90NIKMh/hqmNAJkbrADJSG1g"
AWS_SESSION_TOKEN        = None  # or "YOUR_SESSION_TOKEN" if using STS

# App / infra names
ECR_REPO_NAME            = "agrorisk"
IMAGE_TAG                = "latest"
TASK_FAMILY              = "agrorisk-task"
CLUSTER_NAME             = "streamlit-cluster"
SERVICE_NAME             = "streamlit-task-service-uu9sz6hc"
EXECUTION_ROLE_NAME      = "ecsTaskExecutionRole"  # standard AWS name
TASK_ROLE_NAME           = None  # e.g., "agroriskTaskRole" if your app needs AWS access
LOG_GROUP_NAME           = "/ecs/agrorisk"

# CPU/Memory for Fargate (string values per ECS API)
TASK_CPU                 = "1024"
TASK_MEMORY              = "2048"

# Container ports to publish (Option B mindset: single container, 3 processes)
CONTAINER_PORTS          = [8000, 8501, 8502]

# Docker build context / dockerfile
DOCKER_BUILD_CONTEXT     = "."
DOCKERFILE_PATH          = "Dockerfile"

# Desired count for the service
DESIRED_COUNT            = 1

# Security Group name for the service
SECURITY_GROUP_NAME      = "agrorisk-sg"

# =========================
# ==== HELPERS ============
# =========================

def session():
    return boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=AWS_REGION,
    )

def run(cmd: List[str], check: bool = True, capture_output: bool = False, text: bool = True) -> subprocess.CompletedProcess:
    print(f"→ Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)


def ensure_ecr_repo(ecr, repo_name: str) -> str:
    try:
        resp = ecr.describe_repositories(repositoryNames=[repo_name])
        uri = resp["repositories"][0]["repositoryUri"]
        print(f"✔ ECR repo exists: {uri}")
        return uri
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            print("… Creating ECR repository")
            resp = ecr.create_repository(repositoryName=repo_name)
            uri = resp["repository"]["repositoryUri"]
            print(f"✔ Created ECR repo: {uri}")
            return uri
        raise

"""
def ensure_ecr_repo(ecr, repo_name: str) -> str:
    try:
        resp = ecr.create_repository(repositoryName=repo_name)
        uri = resp["repository"]["repositoryUri"]
        print(f"✔ Created ECR repo: {uri}")
        return uri
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            # Repo exists, get its URI without describe_repositories
            account_id = AWS_ACCOUNT_ID
            region = AWS_REGION
            uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}"
            print(f"✔ ECR repo exists: {uri}")
            return uri
        else:
            raise
"""
def docker_login_to_ecr(ecr):
    print("… Getting ECR auth token")
    auth = ecr.get_authorization_token()
    data = auth["authorizationData"][0]
    token = base64.b64decode(data["authorizationToken"]).decode("utf-8")
    username, password = token.split(":")
    proxy = data["proxyEndpoint"]  # e.g., https://12345.dkr.ecr.us-east-1.amazonaws.com
    # Docker can accept the https URL; both forms typically work
    run(["docker", "login", "-u", username, "-p", password, proxy])

def docker_build_tag_push(repo_uri: str, image_tag: str):
    local_name = f"{ECR_REPO_NAME}:{image_tag}"
    remote_name = f"{repo_uri}:{image_tag}"
    run(["docker", "build", "-t", local_name, "-f", DOCKERFILE_PATH, DOCKER_BUILD_CONTEXT])
    run(["docker", "tag", local_name, remote_name])
    run(["docker", "push", remote_name])

def ensure_exec_role(iam) -> str:
    arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{EXECUTION_ROLE_NAME}"
    try:
        iam.get_role(RoleName=EXECUTION_ROLE_NAME)
        print(f"✔ Execution role exists: {arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print("… Creating ecsTaskExecutionRole")
            trust = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            iam.create_role(RoleName=EXECUTION_ROLE_NAME,
                            AssumeRolePolicyDocument=json.dumps(trust))
            iam.attach_role_policy(
                RoleName=EXECUTION_ROLE_NAME,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
            )
            print(f"✔ Created execution role: {arn}")
        else:
            raise
    return arn

def ensure_task_role(iam) -> Optional[str]:
    if not TASK_ROLE_NAME:
        return None
    arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{TASK_ROLE_NAME}"
    try:
        iam.get_role(RoleName=TASK_ROLE_NAME)
        print(f"✔ Task role exists: {arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"… Creating task role {TASK_ROLE_NAME}")
            trust = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            iam.create_role(RoleName=TASK_ROLE_NAME,
                            AssumeRolePolicyDocument=json.dumps(trust))
            # Attach specific policies your app needs, e.g.:
            # iam.attach_role_policy(RoleName=TASK_ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess")
            print(f"✔ Created task role: {arn} (no policies attached by default)")
        else:
            raise
    return arn

def ensure_log_group(logs):
    try:
        logs.create_log_group(logGroupName=LOG_GROUP_NAME)
        print(f"✔ Created log group: {LOG_GROUP_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
            print(f"✔ Log group exists: {LOG_GROUP_NAME}")
        else:
            raise

def get_default_vpc_and_public_subnets(ec2):
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError("No default VPC found")
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "default-for-az", "Values": ["true"]},
        ]
    )["Subnets"]
    if len(subnets) < 2:
        raise RuntimeError("Need at least two default subnets")
    subnet_ids = [s["SubnetId"] for s in subnets[:2]]
    print(f"✔ Default VPC: {vpc_id}, Subnets: {subnet_ids}")
    return vpc_id, subnet_ids

def ensure_security_group(ec2, vpc_id: str, name: str, ports: List[int]) -> str:
    # Try to find existing SG by name
    resp = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [name]}, {"Name": "vpc-id", "Values": [vpc_id]}]
    )
    if resp["SecurityGroups"]:
        sg_id = resp["SecurityGroups"][0]["GroupId"]
        print(f"✔ Using existing SG: {name} ({sg_id})")
    else:
        sg = ec2.create_security_group(GroupName=name, Description="agrorisk fargate", VpcId=vpc_id)
        sg_id = sg["GroupId"]
        print(f"✔ Created SG: {name} ({sg_id})")

    # Allow inbound on specified ports from anywhere (tighten to your IP for security)
    for port in ports:
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id, IpPermissions=[{
                    "IpProtocol": "tcp",
                    "FromPort": port,
                    "ToPort": port,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                }]
            )
            print(f"  ↳ opened tcp/{port}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidPermission.Duplicate":
                print(f"  ↳ tcp/{port} already allowed")
            else:
                raise
    return sg_id

def ensure_cluster(ecs):
    clusters = ecs.describe_clusters(clusters=[CLUSTER_NAME])["clusters"]
    if not clusters or clusters[0]["status"] == "INACTIVE":
        ecs.create_cluster(clusterName=CLUSTER_NAME)
        print(f"✔ Created ECS cluster: {CLUSTER_NAME}")
    else:
        print(f"✔ ECS cluster exists: {CLUSTER_NAME}")

def register_task_definition(ecs, image_uri: str, exec_role_arn: str, task_role_arn: Optional[str]) -> int:
    container_def = {
        "name": "agrorisk",
        "image": image_uri,
        "essential": True,
        "portMappings": [{"containerPort": p, "protocol": "tcp"} for p in CONTAINER_PORTS],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": LOG_GROUP_NAME,
                "awslogs-region": AWS_REGION,
                "awslogs-stream-prefix": "ecs"
            }
        },
    }
    req = {
        "family": TASK_FAMILY,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": TASK_CPU,
        "memory": TASK_MEMORY,
        "executionRoleArn": exec_role_arn,
        "containerDefinitions": [container_def],
    }
    if task_role_arn:
        req["taskRoleArn"] = task_role_arn

    resp = ecs.register_task_definition(**req)
    rev = resp["taskDefinition"]["revision"]
    print(f"✔ Registered task definition: {TASK_FAMILY}:{rev}")
    return rev

def create_or_update_service(ecs, subnets: List[str], sg_id: str, td_revision: int):
    # Check if service exists
    try:
        svc = ecs.describe_services(cluster=CLUSTER_NAME, services=[SERVICE_NAME])["services"][0]
        exists = svc["status"] != "INACTIVE"
    except (KeyError, IndexError):
        exists = False

    if not exists:
        print("… Creating ECS service")
        ecs.create_service(
            cluster=CLUSTER_NAME,
            serviceName=SERVICE_NAME,
            taskDefinition=f"{TASK_FAMILY}:{td_revision}",
            desiredCount=DESIRED_COUNT,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": subnets,
                    "securityGroups": [sg_id],
                    "assignPublicIp": "ENABLED",
                }
            },
        )
        print(f"✔ Created service: {SERVICE_NAME}")
    else:
        print("… Updating ECS service")
        ecs.update_service(
            cluster=CLUSTER_NAME,
            service=SERVICE_NAME,
            taskDefinition=f"{TASK_FAMILY}:{td_revision}",
            desiredCount=DESIRED_COUNT,
            forceNewDeployment=True,
        )
        print(f"✔ Updated service: {SERVICE_NAME}")

def wait_for_running_task_and_public_ip(ecs, ec2) -> Optional[str]:
    print("… Waiting for task to start (this can take ~1–3 minutes)")
    deadline = time.time() + 600
    task_arn = None
    while time.time() < deadline:
        tasks = ecs.list_tasks(cluster=CLUSTER_NAME, serviceName=SERVICE_NAME)["taskArns"]
        if tasks:
            task_arn = tasks[0]
            desc = ecs.describe_tasks(cluster=CLUSTER_NAME, tasks=[task_arn])["tasks"][0]
            last_status = desc.get("lastStatus")
            desired_status = desc.get("desiredStatus")
            if last_status == "RUNNING":
                # Grab ENI → Public IP
                for att in desc.get("attachments", []):
                    if att.get("type") == "ElasticNetworkInterface":
                        eni_id = next((d["value"] for d in att["details"] if d["name"] == "networkInterfaceId"), None)
                        if eni_id:
                            eni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])["NetworkInterfaces"][0]
                            pip = (eni.get("Association") or {}).get("PublicIp")
                            if pip:
                                print(f"✔ Task is RUNNING. Public IP: {pip}")
                                return pip
            else:
                print(f"  status: last={last_status}, desired={desired_status}")
        time.sleep(5)
    print("⚠ Timed out waiting for RUNNING task with public IP")
    return None

# =========================
# ===== MAIN FLOW =========
# =========================

def main():
    # Boto3 clients
    sess = session()
    ecr = sess.client("ecr")
    iam = sess.client("iam")
    ecs = sess.client("ecs")
    ec2 = sess.client("ec2")
    logs = sess.client("logs")

    # 1) ECR repo + docker push
    repo_uri = ensure_ecr_repo(ecr, ECR_REPO_NAME)
    docker_login_to_ecr(ecr)
    docker_build_tag_push(repo_uri, IMAGE_TAG)

    # 2) IAM roles + logs
    exec_role_arn = ensure_exec_role(iam)
    task_role_arn = ensure_task_role(iam)
    ensure_log_group(logs)

    # 3) Networking
    vpc_id, subnets = get_default_vpc_and_public_subnets(ec2)
    sg_id = ensure_security_group(ec2, vpc_id, SECURITY_GROUP_NAME, CONTAINER_PORTS)

    # 4) Cluster + task def + service
    ensure_cluster(ecs)
    td_rev = register_task_definition(ecs, f"{repo_uri}:{IMAGE_TAG}", exec_role_arn, task_role_arn)
    create_or_update_service(ecs, subnets, sg_id, td_rev)

    # 5) Wait & print access URLs
    pip = wait_for_running_task_and_public_ip(ecs, ec2)
    if pip:
        print("\n=== ACCESS ===")
        if 8000 in CONTAINER_PORTS:
            print(f"API (FastAPI docs): http://{pip}:8000/docs")
        if 8501 in CONTAINER_PORTS:
            print(f"UI 1 (Streamlit):   http://{pip}:8501")
        if 8502 in CONTAINER_PORTS:
            print(f"UI 2 (Streamlit):   http://{pip}:8502")

    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {e}")
        if e.stdout:
            print("STDOUT:\n", e.stdout)
        if e.stderr:
            print("STDERR:\n", e.stderr)
        sys.exit(1)
    except ClientError as e:
        print(f"\n[AWS ERROR] {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n[UNEXPECTED] {e}")
        sys.exit(3)
