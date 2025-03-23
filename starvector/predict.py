from cog import BasePredictor, Input, Path
import torch
from PIL import Image
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
        image: Path = Input(description="Input image to convert to SVG"),
        max_length: int = Input(description="Maximum length of generated SVG", default=1000),
    ) -> str:
        """Run a single prediction on the model"""
        # Load and process the image
        image = Image.open(image)
        processed_image = self.processor(image, return_tensors="pt")['pixel_values']
        
        # Ensure correct shape
        if not processed_image.shape[0] == 1:
            processed_image = processed_image.unsqueeze(0)
        
        # Create batch
        batch = {"image": processed_image}
        
        # Generate SVG
        with torch.no_grad():
            raw_svg = self.model.generate_im2svg(batch, max_length=max_length)[0]
        
        return raw_svg
