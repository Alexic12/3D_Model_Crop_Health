@echo off
setlocal enabledelayedexpansion

REM --- CONFIG ---
set AWS_REGION=us-east-1
set AWS_ACCOUNT_ID=147997127433
set ECR_REPO_NAME=agrorisk
set IMAGE_TAG=latest
set CLUSTER_NAME=streamlit-cluster
set SERVICE_NAME=streamlit-task-service-uu9sz6hc
set TASK_FAMILY=agrorisk-task
set TASK_DEF_FILE=agrorisk-task.json

echo ===============================
echo 🚀 BUILDING DOCKER IMAGE...
echo ===============================
docker build -t %ECR_REPO_NAME%:%IMAGE_TAG% .

echo ===============================
echo 🏷️ ENSURING ECR REPO EXISTS...
echo ===============================
aws ecr describe-repositories --repository-names %ECR_REPO_NAME% --region %AWS_REGION% >NUL 2>&1
if errorlevel 1 (
  aws ecr create-repository --repository-name %ECR_REPO_NAME% --region %AWS_REGION%
)

echo ===============================
echo 🔐 LOGGING IN TO ECR...
echo ===============================
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com

echo ===============================
echo 🏷️ TAGGING IMAGE...
echo ===============================
docker tag %ECR_REPO_NAME%:%IMAGE_TAG% %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%:%IMAGE_TAG%

echo ===============================
echo 📤 PUSHING TO ECR...
echo ===============================
docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%:%IMAGE_TAG%

echo ===============================
echo 📘 REGISTERING TASK DEFINITION...
echo ===============================
aws ecs register-task-definition --cli-input-json file://%TASK_DEF_FILE%

REM --- Get latest revision number for the family ---
FOR /F %%i IN ('aws ecs describe-task-definition --task-definition %TASK_FAMILY% --query "taskDefinition.revision" --output text --region %AWS_REGION%') DO set TASK_REV=%%i
echo ➕ Task definition registered as: %TASK_FAMILY%:%TASK_REV%

echo ===============================
echo 🔁 UPDATING ECS SERVICE...
echo ===============================
aws ecs update-service ^
  --cluster %CLUSTER_NAME% ^
  --service %SERVICE_NAME% ^
  --task-definition %TASK_FAMILY%:%TASK_REV% ^
  --region %AWS_REGION% ^
  --force-new-deployment

echo ===============================
echo ✅ DEPLOY COMPLETE!
echo ===============================
pause