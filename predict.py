from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import os
from huggingface_hub import hf_hub_download

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Import here to avoid issues with missing dependencies
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # Set environment variables to control torch behavior
        os.environ["TORCH_DEVICE"] = "cuda"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Load model directly from Hugging Face
        model_name = "starvector/starvector-8b-im2svg"
        
        # Install any missing dependencies
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                device_map="auto"
            )
            self.processor = self.model.model.processor
        except Exception as e:
            # If there's an error, print it and try to install missing dependencies
            print(f"Error loading model: {e}")
            raise

    def predict(
        self,
        image: Path = Input(description="Input image to convert to SVG"),
        max_length: int = Input(description="Maximum length of generated SVG", default=1000),
    ) -> str:
        """Run a single prediction on the model"""
        # Load and process the image
        image = Image.open(image)
        processed_image = self.processor(image, return_tensors="pt")['pixel_values']
        
        # Move to GPU
        processed_image = processed_image.to("cuda")
        
        # Ensure correct shape
        if not processed_image.shape[0] == 1:
            processed_image = processed_image.unsqueeze(0)
        
        # Create batch
        batch = {"image": processed_image}
        
        # Generate SVG
        with torch.no_grad():
            raw_svg = self.model.generate_im2svg(batch, max_length=max_length)[0]
        
        return raw_svg
