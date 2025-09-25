#! /bin/bash
# Function to display help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build container images for the prescription-rag application.
Supports both ARM64 and AMD64 architectures and uses either Podman or Docker.

Required Options:
    --version VERSION   Image version tag (e.g., v1, 1.0.0)

Optional Options:
    --platform ARCH     Target architecture (arm64, amd64, or both) [default: both]
    --arch ARCH         Alias for --platform
    --run              Run the container after building (requires --platform to be specified)
    --env-file FILE    Load environment variables from file (required when using --run)
    --help, -h         Show this help message and exit

Examples:
    $0 --version v1
    $0 --version v1 --platform amd64
    $0 --version v2 --platform arm64 --run --env-file .env
    $0 --version v3 --platform both
    $0 --version v1 --platform amd64 --run --env-file .env
    $0 --help

Architecture Support:
    arm64    Build for ARM64 architecture (Apple Silicon, ARM servers)
    amd64    Build for AMD64/x86_64 architecture (Intel/AMD processors)  
    both     Build for both architectures (default)

Container Runtime:
    The script automatically detects and uses either Podman or Docker.
    Podman is preferred if both are available.

Environment File:
    When using --run, you must specify an environment file with --env-file.
    The file should contain key=value pairs, for example:
        OLLAMA_HOST=http://192.168.0.100:11434
        PRESCRIPTION_PORT=8081
        LOG_LEVEL=INFO
    
    Port Configuration:
        PRESCRIPTION_PORT - Port for the prescription service (default: 8081)
        The container will be accessible on this port on your host machine.
    
    See prescription.env.example for a complete example.

EOF
}

# This script builds container images for the prescription-rag application
# It supports both ARM64 and AMD64 architectures and uses either Podman or Docker
# Use --help for usage information



# Check if we're in the correct directory (should be /prescription-rag)
CURRENT_DIR=$(pwd)
if [[ ! "$CURRENT_DIR" =~ .*/prescription-rag$ ]]; then
    echo "Error: This script must be run from the prescription-rag directory."
    echo "Current directory: $CURRENT_DIR"
    echo "Please navigate to the prescription-rag directory and run the script again."
    echo "Example: cd /path/to/prescription-rag && ./create-image-prescription.sh"
    exit 1
fi

echo "Running from correct directory: $CURRENT_DIR"

# Check for container runtime (podman or docker)
CONTAINER_RUNTIME=""
if command -v podman &> /dev/null; then
    CONTAINER_RUNTIME="podman"
    echo "Using Podman as container runtime"
    
    # Check if logged in to container registry
    if ! podman info --format "{{.Registries}}" | grep -q "quay.io"; then
        echo "Warning: You may not be logged in to quay.io registry."
        echo "To login, run: podman login quay.io"
    fi
elif command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
    echo "Using Docker as container runtime"
    
    # Check if logged in to container registry
    if ! docker info 2>/dev/null | grep -q "Registry:"; then
        echo "Warning: Docker daemon may not be running or you may not be logged in."
        echo "To login to quay.io, run: docker login quay.io"
    fi
else
    echo "Error: Neither Podman nor Docker is installed."
    echo "Please install either Podman or Docker to use this script."
    echo "Podman: https://podman.io/getting-started/installation"
    echo "Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Parse command line arguments
RUN_AFTER_BUILD=false
ARCH=""
VERSION=""
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --run)
            RUN_AFTER_BUILD=true
            shift
            ;;
        --platform|--arch)
            if [[ $# -lt 2 ]] || [[ ! "$2" =~ ^(amd64|arm64|both)$ ]]; then
                echo "Error: $1 requires a value (amd64, arm64, or both)"
                echo "Use --help for usage information"
                exit 1
            fi
            ARCH="$2"
            shift 2
            ;;
        --version)
            if [[ $# -lt 2 ]] || [[ -z "$2" ]]; then
                echo "Error: --version requires a value"
                echo "Use --help for usage information"
                exit 1
            fi
            VERSION="$2"
            shift 2
            ;;
        --env-file)
            if [[ $# -lt 2 ]] || [[ -z "$2" ]]; then
                echo "Error: --env-file requires a file path"
                echo "Use --help for usage information"
                exit 1
            fi
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validation: version is mandatory
if [ -z "$VERSION" ]; then
    echo "Error: --version is mandatory"
    echo "Use --help for usage information"
    exit 1
fi

# Validation: if --run is true, --platform must be specified
if [ "$RUN_AFTER_BUILD" = true ] && [ -z "$ARCH" ]; then
    echo "Error: --run requires --platform to be specified"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$ARCH" ]; then
    ARCH="both"
fi

echo "--------------------------------"
echo "Container runtime: $CONTAINER_RUNTIME"
echo "Building for architecture: $ARCH"
echo "Version: $VERSION"
echo "Running after build: $RUN_AFTER_BUILD"
echo "Starting the script"

# create the requirement file
if command -v uv &> /dev/null; then
    echo "Using uv package manager to compile requirements..."
    uv pip compile ./pyproject.toml -o ./requirements.txt
else
    echo "*** for now the script will assume an upto date requirements.txt file is present in the root directory ***"
    echo "Reason:"
    echo "uv package manager not found. Please install uv first."
    echo "You can install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "or you can install it with: pip install uv"
fi

if [ "$ARCH" = "amd64" ] || [ "$ARCH" = "both" ]; then
    $CONTAINER_RUNTIME pull --arch=amd64  python:3.13-slim
    $CONTAINER_RUNTIME tag docker.io/library/python:3.13-slim  localhost/python:amd64-3.13-slim
    $CONTAINER_RUNTIME build --platform linux/amd64 --build-arg BASE_TAG=amd64-3.13-slim -t prescription:amd64-v$VERSION -f ./Dockerfile.prescription .
    $CONTAINER_RUNTIME tag localhost/prescription:amd64-v$VERSION quay.io/rajivranjan/prescription:amd64-v$VERSION
    $CONTAINER_RUNTIME push quay.io/rajivranjan/prescription:amd64-v$VERSION
fi

if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "both" ]; then
    $CONTAINER_RUNTIME pull --arch=arm64  python:3.13-slim 
    $CONTAINER_RUNTIME tag docker.io/library/python:3.13-slim  localhost/python:arm64-3.13-slim
    $CONTAINER_RUNTIME build --platform linux/arm64 --build-arg BASE_TAG=arm64-3.13-slim -t prescription:arm64-v$VERSION -f ./Dockerfile.prescription .
    $CONTAINER_RUNTIME tag localhost/prescription:arm64-v$VERSION quay.io/rajivranjan/prescription:arm64-v$VERSION
    $CONTAINER_RUNTIME push quay.io/rajivranjan/prescription:arm64-v$VERSION
fi


if [ "$RUN_AFTER_BUILD" = true ]; then
    ENV_OPTIONS=""
    PRESCRIPTION_PORT="8081"  # Default port
    
    if [ -n "$ENV_FILE" ]; then
        if [ -f "$ENV_FILE" ]; then
            ENV_OPTIONS="--env-file $ENV_FILE"
            echo "Using environment file: $ENV_FILE"
            
            # Extract PRESCRIPTION_PORT from environment file
            if grep -q "^PRESCRIPTION_PORT=" "$ENV_FILE"; then
                PRESCRIPTION_PORT=$(grep "^PRESCRIPTION_PORT=" "$ENV_FILE" | cut -d'=' -f2 | sed 's/["\047]//g')
                echo "Using port from env file: $PRESCRIPTION_PORT"
            else
                echo "PRESCRIPTION_PORT not found in $ENV_FILE, using default: $PRESCRIPTION_PORT"
            fi
        else
            echo "Error: Environment file $ENV_FILE not found"
            echo "Use --help for usage information"
            exit 1
        fi
    else
        echo "Error: --run requires --env-file to be specified"
        echo "Use --help for usage information"
        exit 1
    fi
    
    if [ "$ARCH" = "amd64" ]; then
        echo "Running AMD64 container on port $PRESCRIPTION_PORT..."
        $CONTAINER_RUNTIME run -it --rm -p "$PRESCRIPTION_PORT:$PRESCRIPTION_PORT" $ENV_OPTIONS localhost/prescription:amd64-v$VERSION
    elif [ "$ARCH" = "arm64" ]; then
        echo "Running ARM64 container on port $PRESCRIPTION_PORT..."
        $CONTAINER_RUNTIME run -it --rm -p "$PRESCRIPTION_PORT:$PRESCRIPTION_PORT" $ENV_OPTIONS localhost/prescription:arm64-v$VERSION
    fi    
fi