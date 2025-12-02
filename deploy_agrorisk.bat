@echo off
set AWS_REGION=us-east-1
set AWS_ACCOUNT_ID=147997127433
set ECR_REPO_NAME=agrorisk
set IMAGE_TAG=latest
set TASK_FAMILY=agrorisk-task
set CLUSTER_NAME=streamlit-cluster
set SERVICE_NAME=streamlit-task-service-uu9sz6hc

echo Logging into AWS ECR...
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com

echo Ensuring ECR repo exists...
aws ecr describe-repositories --repository-names %ECR_REPO_NAME% --region %AWS_REGION% >NUL 2>&1
if errorlevel 1 (
  aws ecr create-repository --repository-name %ECR_REPO_NAME% --region %AWS_REGION%
)

echo Tagging local image as %ECR_REPO_NAME%:%IMAGE_TAG%...
docker tag agrorisk:latest %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%:%IMAGE_TAG%

echo Pushing image to ECR...
docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%:%IMAGE_TAG%

echo Creating task definition file...
> %TASK_FAMILY%.json echo {
>> %TASK_FAMILY%.json echo   "family": "%TASK_FAMILY%",
>> %TASK_FAMILY%.json echo   "networkMode": "awsvpc",
>> %TASK_FAMILY%.json echo   "requiresCompatibilities": ["FARGATE"],
>> %TASK_FAMILY%.json echo   "cpu": "1024",
>> %TASK_FAMILY%.json echo   "memory": "2048",
>> %TASK_FAMILY%.json echo   "executionRoleArn": "arn:aws:iam::%AWS_ACCOUNT_ID%:role/ecsTaskExecutionRole",
>> %TASK_FAMILY%.json echo   "containerDefinitions": [
>> %TASK_FAMILY%.json echo     {
>> %TASK_FAMILY%.json echo       "name": "agrorisk",
>> %TASK_FAMILY%.json echo       "image": "%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%:%IMAGE_TAG%",
>> %TASK_FAMILY%.json echo       "essential": true,
>> %TASK_FAMILY%.json echo       "portMappings": [
>> %TASK_FAMILY%.json echo         { "containerPort": 8000, "protocol": "tcp" },
>> %TASK_FAMILY%.json echo         { "containerPort": 8501, "protocol": "tcp" },
>> %TASK_FAMILY%.json echo         { "containerPort": 8502, "protocol": "tcp" }
>> %TASK_FAMILY%.json echo       ],
>> %TASK_FAMILY%.json echo       "logConfiguration": {
>> %TASK_FAMILY%.json echo         "logDriver": "awslogs",
>> %TASK_FAMILY%.json echo         "options": {
>> %TASK_FAMILY%.json echo           "awslogs-group": "/ecs/agrorisk",
>> %TASK_FAMILY%.json echo           "awslogs-region": "%AWS_REGION%",
>> %TASK_FAMILY%.json echo           "awslogs-stream-prefix": "ecs"
>> %TASK_FAMILY%.json echo         }
>> %TASK_FAMILY%.json echo       }
>> %TASK_FAMILY%.json echo     }
>> %TASK_FAMILY%.json echo   ]
>> %TASK_FAMILY%.json echo }

echo Registering ECS task definition...
aws ecs register-task-definition --cli-input-json file://%TASK_FAMILY%.json --region %AWS_REGION%

FOR /F %%i IN ('aws ecs describe-task-definition --task-definition %TASK_FAMILY% --query "taskDefinition.revision" --output text --region %AWS_REGION%') DO set TASK_REV=%%i
echo Registered: %TASK_FAMILY%:%TASK_REV%

echo Updating ECS service to new task definition...
aws ecs update-service ^
  --cluster %CLUSTER_NAME% ^
  --service %SERVICE_NAME% ^
  --task-definition %TASK_FAMILY%:%TASK_REV% ^
  --region %AWS_REGION% ^
  --force-new-deployment

echo DONE. Check ECS Console for task status.
pause