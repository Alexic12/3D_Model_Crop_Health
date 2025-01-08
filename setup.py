import subprocess


def install_packages(requirements_file, extra_index_url=None):
    try:
        command = ["pip", "install", "-r", requirements_file]
        if extra_index_url:
            command.extend(["--index-url", extra_index_url])
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages from {requirements_file}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    # Install the remaining packages from requirements.txt
    ## activate virtual environment
    
    install_packages("requirements.txt")
    install_packages("requirements_dev.txt")