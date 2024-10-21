import subprocess
import sys
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_install_rust():
    try:
        result = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True)
        logger.info(f"Rust is already installed: {result.stdout.strip()}")

        cargo_result = subprocess.run(["cargo", "--version"], check=True, capture_output=True, text=True)
        logger.info(f"Cargo is already installed: {cargo_result.stdout.strip()}")

        # Log the current PATH
        logger.info(f"Current PATH: {os.environ['PATH']}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Error checking Rust/Cargo: {e}")
        logger.info("Rust is not installed. Attempting to install...")
        try:
            if sys.platform.startswith('win'):
                # For Windows
                subprocess.run(["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "https://sh.rustup.rs", "|", "sh"], check=True)
            else:
                # For Unix-like systems
                subprocess.run(["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "https://sh.rustup.rs", "|", "sh", "-s", "--", "-y"], check=True)
            logger.info("Rust installed successfully")
            # Update PATH to include Cargo bin directory
            os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.cargo/bin")
        except subprocess.CalledProcessError as install_error:
            logger.error(f"Failed to install Rust. Please install manually from https://www.rust-lang.org/tools/install: {install_error}", exc_info=True)
            sys.exit(1)

def check_python_version():
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major != 3 or version.minor != 13:
        logger.warning("Python version 3.13 is required for this project. Please update your Python version.")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Checking system requirements for ScriptumAICPU...")
    check_python_version()
    check_and_install_rust()
    logger.info("System check complete. You can now proceed with 'pip install -r requirements.txt'")