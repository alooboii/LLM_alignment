#!/usr/bin/env python3
"""
Simple Python-based installer for alignment methods dependencies

Usage:
    python install.py                    # Install everything
    python install.py --quick            # Skip optional packages
    python install.py --cpu-only         # CPU-only versions
    python install.py --verify           # Verify installation only
"""

import subprocess
import sys
import argparse
import platform
from typing import List, Tuple

# Colors for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def run_command(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print_info(f"Python version: {version_str}")
    
    if version_info.major >= 3 and version_info.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version_str}")
        return False

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print_success("pip is available")
        return True
    except subprocess.CalledProcessError:
        print_error("pip not found")
        return False

def upgrade_pip():
    """Upgrade pip, setuptools, and wheel"""
    print_header("Upgrading pip, setuptools, wheel")
    
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", 
           "pip", "setuptools", "wheel"]
    
    returncode, stdout, stderr = run_command(cmd, check=False)
    
    if returncode == 0:
        print_success("Successfully upgraded pip tools")
        return True
    else:
        print_error("Failed to upgrade pip tools")
        print_error(stderr)
        return False

def check_cuda():
    """Check if CUDA is available"""
    print_header("Checking CUDA")
    
    # Try nvidia-smi
    returncode, stdout, stderr = run_command(["nvidia-smi"], check=False)
    
    if returncode == 0:
        # Extract CUDA version
        for line in stdout.split('\n'):
            if "CUDA Version" in line:
                version = line.split("CUDA Version:")[1].split()[0]
                print_success(f"CUDA found: Version {version}")
                return True
        print_success("NVIDIA GPU found")
        return True
    else:
        print_warning("CUDA not found - will install CPU-only versions")
        return False

def install_pytorch(cpu_only: bool = False):
    """Install PyTorch"""
    print_header("Installing PyTorch")
    
    if cpu_only:
        print_info("Installing PyTorch (CPU-only)")
        cmd = [sys.executable, "-m", "pip", "install", "torch", 
               "torchvision", "torchaudio", 
               "--index-url", "https://download.pytorch.org/whl/cpu"]
    else:
        print_info("Installing PyTorch (with CUDA)")
        cmd = [sys.executable, "-m", "pip", "install", "torch", 
               "torchvision", "torchaudio"]
    
    returncode, stdout, stderr = run_command(cmd, check=False)
    
    if returncode == 0:
        print_success("PyTorch installed")
        return True
    else:
        print_error("Failed to install PyTorch")
        print_error(stderr)
        return False

def install_package(package: str, display_name: str = None) -> bool:
    """Install a single package"""
    if display_name is None:
        display_name = package
    
    print_info(f"Installing {display_name}...")
    
    cmd = [sys.executable, "-m", "pip", "install", package]
    returncode, stdout, stderr = run_command(cmd, check=False)
    
    if returncode == 0:
        print_success(f"{display_name} installed")
        return True
    else:
        print_error(f"Failed to install {display_name}")
        return False

def install_packages(packages: List[str], category: str) -> bool:
    """Install a list of packages"""
    print_header(f"Installing {category}")
    
    success = True
    for package in packages:
        if not install_package(package):
            success = False
    
    return success

def install_core_packages(cpu_only: bool = False, quick: bool = False) -> bool:
    """Install all core packages"""
    
    # Transformers & HuggingFace
    transformers_packages = [
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "accelerate>=0.25.0",
        "huggingface-hub>=0.19.0",
    ]
    install_packages(transformers_packages, "Transformers & HuggingFace Libraries")
    
    # RLHF Libraries
    rlhf_packages = [
        "peft>=0.7.0",
        "trl>=0.7.10",
    ]
    
    if not cpu_only:
        rlhf_packages.append("bitsandbytes>=0.41.0")
    
    install_packages(rlhf_packages, "RLHF Libraries")
    
    # Scientific Computing
    scientific_packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ]
    install_packages(scientific_packages, "Scientific Computing")
    
    # Visualization
    viz_packages = [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.14.0",
    ]
    install_packages(viz_packages, "Visualization")
    
    # Utilities
    util_packages = [
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "pyarrow>=12.0.0",
    ]
    install_packages(util_packages, "Utilities")
    
    # Optional packages
    if not quick:
        optional_packages = [
            "orjson>=3.9.0",
            "omegaconf>=2.3.0",
        ]
        install_packages(optional_packages, "Optional Packages")
    
    return True

def verify_installation():
    """Verify all packages are installed correctly"""
    print_header("Verifying Installation")
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tensorboard", "TensorBoard"),
        ("tqdm", "tqdm"),
    ]
    
    print("\nPackage Verification:")
    print("-" * 60)
    
    all_ok = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{Colors.GREEN}✓{Colors.END} {display_name:20s} {version}")
        except ImportError:
            print(f"{Colors.RED}✗{Colors.END} {display_name:20s} NOT FOUND")
            all_ok = False
    
    # Test PyTorch CUDA
    print("\n" + "-" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            print(f"{Colors.GREEN}✓{Colors.END} CUDA available: {device_name}")
            print(f"  CUDA version: {cuda_version}")
            print(f"  GPU count: {gpu_count}")
        else:
            print(f"{Colors.YELLOW}ℹ{Colors.END} CUDA not available (CPU-only mode)")
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.END} Error checking CUDA: {e}")
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL REQUIRED PACKAGES INSTALLED!{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME PACKAGES MISSING{Colors.END}")
        return False

def test_huggingface_connection():
    """Test connection to HuggingFace Hub"""
    print_header("Testing HuggingFace Connection")
    
    try:
        from transformers import AutoTokenizer
        
        print_info("Downloading test model (gpt2 tokenizer)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print_success("Successfully connected to HuggingFace Hub")
        print_success("Model download works")
        
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print_info(f"Models cached in: {cache_dir}")
        
        return True
    except Exception as e:
        print_error(f"Failed to connect: {e}")
        print_warning("Check your internet connection")
        return False

def print_summary():
    """Print installation summary"""
    print_header("Installation Complete!")
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        ✓ SETUP SUCCESSFUL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next steps:

1. Run quick test:
   python main.py --quick_test

2. Run full pipeline:
   python main.py --full_pipeline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Useful commands:
  - Check GPU: nvidia-smi
  - Monitor training: tail -f main_pipeline.log
  - View results: cat pipeline_results.json

Documentation:
  - Full guide: README.md
  - Quick start: QUICKSTART.md
  - Troubleshooting: PROJECT_SUMMARY.md
    """)

def main():
    parser = argparse.ArgumentParser(
        description='Install dependencies for alignment methods assignment'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Skip optional packages')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Install CPU-only versions')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify installation')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║         Alignment Methods Assignment - Python Installer                 ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Verify only mode
    if args.verify:
        success = verify_installation()
        sys.exit(0 if success else 1)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Upgrade pip
    upgrade_pip()
    
    # Check CUDA
    has_cuda = check_cuda()
    cpu_only = args.cpu_only or not has_cuda
    
    # Install PyTorch
    if not install_pytorch(cpu_only):
        print_error("Failed to install PyTorch. Exiting.")
        sys.exit(1)
    
    # Install core packages
    install_core_packages(cpu_only=cpu_only, quick=args.quick)
    
    # Verify installation
    if not verify_installation():
        print_warning("Some packages failed to install")
    
    # Test HuggingFace
    test_huggingface_connection()
    
    # Print summary
    print_summary()

if __name__ == "__main__":
    main()