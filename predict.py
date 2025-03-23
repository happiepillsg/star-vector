from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import os
import sys
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
        import sys
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
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Load model directly from Hugging Face
            model_name = "starvector/starvector-8b-im2svg"
            
            print("Loading model from Hugging Face...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                device_map="auto"
            )
            print("Model loaded successfully!")
            
            print("Loading processor...")
            self.processor = self.model.model.processor
            print("Processor loaded successfully!")
            
        except Exception as e:
            print(f"Error during setup: {e}")
            traceback.print_exc()
            raise

    def predict(
        self,
        image: Path = Input(description="Input image to convert to SVG"),
        max_length: int = Input(description="Maximum length of generated SVG", default=1000),
    ) -> str:
        """Run a single prediction on the model"""
        try:
            # Load and process the image
            print(f"Loading image from {image}...")
            image_pil = Image.open(image)
            
            print("Processing image...")
            processed_image = self.processor(image_pil, return_tensors="pt")['pixel_values']
            
            # Move to GPU
            processed_image = processed_image.to("cuda")
            
            # Ensure correct shape
            if not processed_image.shape[0] == 1:
                processed_image = processed_image.unsqueeze(0)
            
            # Create batch
            batch = {"image": processed_image}
            
            # Generate SVG
            print("Generating SVG...")
            with torch.no_grad():
                raw_svg = self.model.generate_im2svg(batch, max_length=max_length)[0]
            
            print("SVG generation complete!")
            return raw_svg
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            # Return a simple error SVG
            return f"""<svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
                <rect width="300" height="200" fill="#f8f9fa" />
                <text x="150" y="100" font-family="Arial" font-size="14" text-anchor="middle" fill="#dc2626">
                    Error: {str(e)}
                </text>
            </svg>"""
