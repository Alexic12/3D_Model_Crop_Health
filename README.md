# Crop Health Visualization App

This application allows users to upload an Excel file containing crop health data and visualize it as an interactive 3D surface plot.

## Features

- Upload Excel files with crop data.
- Visualize NDVI values over a geographical grid.
- Color-coded risk levels (Riesgo) from 1 to 5.
- Interactive 3D plot that can be rotated and zoomed.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/my_crop_health_app.git



   
### Set Up Virtual Environment
- **Create Virtual Environment**  
```bash
python -m venv .venv
```


- **Activate Virtual Environment**  
Once the virtual environment is created, you need to activate it. The activation command differs depending on your operating system:

On Windows:
Remember to enable script execution first:

```bash
Set-ExecutionPolicy Unrestricted -Scope Process
```

```bash

.venv\Scripts\activate
```

- **Docker**  
```bash
# Update the package index
sudo yum update -y

# Install Docker
sudo yum install -y docker

# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Add your user to the Docker group (so you don't need sudo)
sudo usermod -aG docker $USER

# Apply group changes (logout & back in, or run)
newgrp docker
```
Check if Docker is running:
```bash
docker --version   # Check installed version
docker ps          # Verify Docker is running (should return empty list)
```

Step 1: Upload Docker image:

On your local Machine run:

```bash
docker save -o myimage.tar myimage:latest
```
(Change myimage:latest to the correct image name and tag.)

Step 2: Transfer the image to EC2
Use scp to transfer the file:

bash
```bash
scp -i your-key.pem myimage.tar ec2-user@your-ec2-ip:/home/ec2-user/
```
```bash
scp -i keys/HIK-SERVER-KEYS.pem 3d_model_img.tar ec2-user@ec2-174-129-171-88.compute-1.amazonaws.com:/home/ec2-user/
```
Replace:

your-key.pem with your EC2 private key.
your-ec2-ip with your EC2 public IP.

Step 3: Load the image on EC2
SSH into your EC2 instance and run:

bash
```bash
docker load -i myimage.tar
```
Verify that the image is now available:

bash
```bash
docker images
```
3. Running a Docker Container
To run your image in a container:

bash
```bash
docker run -d --name my_container -p 8080:80 myimage:latest
```
Explanation:

-d → Run in detached mode (background).
--name my_container → Assign a custom name to the container.
-p 8080:80 → Map port 8080 on EC2 to port 80 inside the container.
myimage:latest → Your uploaded image.
4. Checking Running Containers
List all running containers:

bash
```bash
docker ps
```
List all containers (including stopped ones):

bash
```bash
docker ps -a
```
5. Stopping and Starting Containers
To stop a running container:

bash
```bash
docker stop my_container
```
To start a stopped container:

bash
```bash
docker start my_container
```
To restart a container:

bash
```bash
docker restart my_container
```
To remove a stopped container:

bash
```bash
docker rm my_container
```
6. Viewing Logs and Stats
To see logs of a container:

bash
```bash
docker logs -f my_container
```
To see resource usage (CPU, memory, etc.):

bash
```bash
docker stats
```
7. Removing Images
To remove an unused Docker image:

bash
```bash
docker rmi myimage:latest
```
To remove all unused images:

bash
```bash
docker image prune -a
```
8. Running Docker Containers on Boot (Optional)
If you want the container to start automatically on reboot, run it with:

bash
```bash
docker run -d --restart always --name my_container -p 8080:80 myimage:latest
```
This ensures it starts even after a server reboot.

9. Checking Docker System Info
To check general system status:

bash
```bash
docker info
```
To see disk space used by Docker:

bash
```bash
docker system df
```
To clean up unused Docker data:

bash
```bash
docker system prune -a
```
```bash

```



