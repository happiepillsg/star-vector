from typing import Optional, List, Dict, Any, Union
from pathlib import Path as PathLib
from cog import BasePredictor, Input, Path
import torch
import os
import base64
import numpy as np
from PIL import Image
import cairosvg
import io
from transformers import AutoModelForCausalLM, AutoProcessor

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "starvector/starvector-8b-im2svg"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        self.processor = self.model.model.processor

    def predict(
        self,
        prompt: str = Input(description="Text prompt for SVG generation"),
        input_image: Union[Path, None] = Input(
            description="Input image for im2svg task",
            default=None
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", 
            ge=1, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", 
            ge=1.0, le=20.0, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None
        ),
        max_length: int = Input(
            description="Maximum length of the SVG code",
            ge=512, le=8192, default=4096
        ),
        svg_size: int = Input(
            description="Size of the output SVG",
            ge=256, le=1024, default=512
        ),
        convert_to_png: bool = Input(
            description="Convert SVG to PNG",
            default=False
        ),
    ) -> Dict[str, Any]:
        """Run a single prediction on the model"""
        # Set the seed for reproducibility
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="big")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if input_image is not None:
            # Load and process the image
            image = Image.open(input_image)
            processed_image = self.processor(image, return_tensors="pt")['pixel_values']
            
            # Ensure correct shape
            if not processed_image.shape[0] == 1:
                processed_image = processed_image.unsqueeze(0)
            
            # Create batch
            batch = {"image": processed_image}
            
            # Generate SVG
            with torch.no_grad():
                svg_code = self.model.generate_im2svg(batch, max_length=max_length)[0]
        else:
            # Handle text-to-svg generation
            # Note: This assumes the model supports text-to-svg, modify as needed
            with torch.no_grad():
                # This is a placeholder - adjust based on actual model implementation
                # Either implement text-to-svg or throw a meaningful error
                raise NotImplementedError("Text-to-SVG generation is not yet implemented")
        
        result = {
            "svg_code": svg_code,
            "seed": seed,
        }
        
        # Convert SVG to PNG if requested
        if convert_to_png:
            png_data = self.svg_to_png(svg_code, svg_size)
            result["png_base64"] = png_data
        
        return result
    
    def svg_to_png(self, svg_code: str, size: int) -> str:
        """Convert SVG to PNG and return as base64 string"""
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'), 
                                    output_width=size, 
                                    output_height=size)
        return base64.b64encode(png_data).decode('utf-8')

