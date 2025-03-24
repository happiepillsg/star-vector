from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import os
import sys  # Add this import
import subprocess
import traceback

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Print Python version and environment info for debugging
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Install system dependencies
        try:
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "libcairo2-dev", "pkg-config", "libpango1.0-dev"], check=True)
        except Exception as e:
            print(f"Error installing system dependencies: {e}")
            # Continue anyway, as we'll try to work around it
        
        # Try to patch the import system to avoid cairosvg
        from types import ModuleType
        
        # Create a mock cairosvg module to prevent import errors
        class MockCairoSVG(ModuleType):
            def __init__(self):
                super().__init__("cairosvg")
            
            def svg2png(self, *args, **kwargs):
                print("Mock svg2png called")
                return b""
        
        # Register the mock module
        sys.modules["cairosvg"] = MockCairoSVG()
        
        # Import here to avoid issues with missing dependencies
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            # Set environment variables to control torch behavior
            os.environ["TORCH_DEVICE"] = "cuda"
            os.environ["TOKENIZERS_PARALLELISM"] = "fals
