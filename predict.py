import os
import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional, List, Dict, Any, Union
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from cog import BasePredictor, Input, Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Set environment variables for better performance
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Cache for loaded models to avoid reloading
        self.loaded_models = {}
        self.current_model_key = None
        
        # We'll load models on-demand to save memory
        logger.info("Setup complete. Models will be loaded on first request.")
        
    def load_model(self, model_name, task="im2svg", engine="hf"):
        """Load a specific model configuration"""
        model_key = f"{model_name}_{engine}"
        
        # Return cached model if already loaded
        if model_key in self.loaded_models:
            logger.info(f"Using cached model: {model_key}")
            self.current_model_key = model_key
            return True
            
        logger.info(f"Loading model: {model_name} for task: {task} using engine: {engine}")
        
        # Determine torch dtype based on model size
        torch_dtype = torch.bfloat16 if "8b" in model_name else torch.float16
        
        try:
            if engine == "hf":
                # Load model using Hugging Face Transformers
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map="auto",
                    use_cache=True  # Enable KV caching for faster inference
                )
                
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                tokenizer = processor.tokenizer
                
                self.loaded_models[model_key] = {
                    "model": model,
                    "processor": processor,
                    "tokenizer": tokenizer,
                    "engine": "hf"
                }
                
            elif engine == "vllm":
                # Import vLLM only when needed
                try:
                    from vllm import LLM, SamplingParams
                    
                    # vLLM requires different initialization
                    model = LLM(
                        model=model_name,
                        dtype="bfloat16" if "8b" in model_name else "float16",
                        trust_remote_code=True,
                        gpu_memory_utilization=0.9,
                        max_model_len=16000  # Based on config max_length
                    )
                    
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    
                    self.loaded_models[model_key] = {
                        "model": model,
                        "processor": processor,
                        "tokenizer": tokenizer,
                        "engine": "vllm"
                    }
                except ImportError:
                    logger.warning("vLLM not available, falling back to HF implementation")
                    return self.load_model(model_name, task, "hf")
            
            self.current_model_key = model_key
            logger.info(f"Model {model_name} loaded successfully with engine {engine}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise e

    def predict(
        self,
        task: str = Input(
            description="Task type: 'im2svg' for image-to-SVG or 'text2svg' for text-to-SVG",
            choices=["im2svg", "text2svg"],
            default="im2svg"
        ),
        input_image: Optional[Path] = Input(
            description="Input image for im2svg task",
            default=None
        ),
        input_text: Optional[str] = Input(
            description="Input text prompt for text2svg task",
            default=None
        ),
        model_size: str = Input(
            description="Model size to use",
            choices=["1b", "8b"],
            default="8b"
        ),
        engine: str = Input(
            description="Generation engine to use",
            choices=["hf", "vllm"],
            default="hf"
        ),
        temperature: float = Input(
            description="Temperature for generation (higher = more creative, lower = more deterministic)",
            default=0.7,
            ge=0.0,
            le=2.0
        ),
        top_p: float = Input(
            description="Top-p sampling parameter (nucleus sampling)",
            default=0.95,
            ge=0.0,
            le=1.0
        ),
        max_length: int = Input(
            description="Maximum length of generated SVG",
            default=None,  # Will be set based on model
            ge=10,
            le=16000
        ),
        min_length: int = Input(
            description="Minimum length of generated SVG",
            default=10,
            ge=10,
            le=1000
        ),
        num_beams: int = Input(
            description="Number of beams for beam search (1 = greedy/sampling)",
            default=1,
            ge=1,
            le=5
        ),
        do_sample: bool = Input(
            description="Whether to use sampling for generation (false = greedy decoding)",
            default=True
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty (1.0 = no penalty)",
            default=1.0,
            ge=1.0,
            le=2.0
        ),
        length_penalty: float = Input(
            description="Length penalty (>1.0 favors longer sequences, <1.0 favors shorter ones)",
            default=0.5,
            ge=0.0,
            le=2.0
        ),
        logit_bias: float = Input(
            description="Bias for the SVG end token (higher = more likely to end SVG properly)",
            default=5.0,
            ge=0.0,
            le=10.0
        ),
        num_generations: int = Input(
            description="Number of SVGs to generate",
            default=1,
            ge=1,
            le=5
        ),
        return_base64: bool = Input(
            description="Return SVG as base64 encoded string (useful for embedding)",
            default=False
        ),
        return_png: bool = Input(
            description="Convert SVG to PNG and return as base64",
            default=False
        ),
        seed: int = Input(
            description="Random seed for reproducibility (0 = random seed)",
            default=0
        )
    ) -> Dict[str, Any]:
        """Run a single prediction on the model"""
        # Set random seed if provided
        if seed > 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Validate inputs
        if task == "im2svg" and input_image is None:
            raise ValueError("Input image is required for im2svg task")
        if task == "text2svg" and input_text is None:
            raise ValueError("Input text is required for text2svg task")
        
        # Determine model name based on size and task
        model_name = f"starvector/starvector-{model_size}-{task}"
        
        # Set default max_length based on model size if not provided
        if max_length is None:
            max_length = 16000 if model_size == "8b" else 7800
        
        # Load model if not already loaded or if different from current model
        self.load_model(model_name, task, engine)
        
        # Get current model components
        current_model = self.loaded_models[self.current_model_key]
        model = current_model["model"]
        processor = current_model["processor"]
        tokenizer = current_model["tokenizer"]
        current_engine = current_model["engine"]
        
        # Prepare inputs based on task
        if task == "im2svg":
            # Process image for im2svg
            image = Image.open(input_image).convert("RGB")
            
            # Resize image based on model size
            im_size = 384 if model_size == "8b" else 224
            image = image.resize((im_size, im_size), Image.LANCZOS)
            
            if current_engine == "hf":
                # Process image with the processor for HF
                inputs = processor(images=image, return_tensors="pt")
                # Move inputs to the appropriate device
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            else:
                # For vLLM, we need to handle the image processing differently
                pixel_values = processor.image_processor(images=image, return_tensors="pt").pixel_values
                prompt = ""  # Empty prompt for image-only input
                inputs = {"prompt": prompt, "images": image}
                
        else:  # text2svg
            # Process text for text2svg
            if current_engine == "hf":
                inputs = processor(text=input_text, return_tensors="pt")
                # Move inputs to the appropriate device
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            else:
                inputs = {"prompt": input_text}
        
        # Set generation parameters
        if current_engine == "hf":
            # HF generation config
            generation_config = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "temperature": temperature if do_sample else 1.0,
                "do_sample": do_sample,
                "top_p": top_p if do_sample else 1.0,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "num_return_sequences": num_generations,
            }
            
            # Add logit bias for SVG end token if specified
            if logit_bias > 0:
                svg_end_token_id = tokenizer.convert_tokens_to_ids("</svg>")
                if svg_end_token_id:
                    generation_config["logit_bias"] = {svg_end_token_id: logit_bias}
            
            # Generate SVG
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_config)
            
            # Decode the generated SVGs
            generated_svgs = []
            for i in range(outputs.shape[0]):
                svg = processor.decode(outputs[i], skip_special_tokens=True)
                generated_svgs.append(self.post_process_svg(svg))
                
        else:  # vLLM
            # vLLM sampling parameters
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=max_length,
                min_tokens=min_length,
                temperature=temperature if do_sample else 0.0,
                top_p=top_p if do_sample else 1.0,
                repetition_penalty=repetition_penalty,
                n=num_generations,
                use_beam_search=(num_beams > 1),
                best_of=num_beams if num_beams > 1 else None,
                len_penalty=length_penalty,
                stop=["</svg>"]  # Stop generation at SVG end tag
            )
            
            # Generate with vLLM
            outputs = model.generate(
                prompts=[inputs["prompt"]],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            # Extract generated text
            generated_svgs = []
            for output in outputs:
                svg = output.outputs[0].text
                generated_svgs.append(self.post_process_svg(svg))
        
        # Prepare results
        results = {
            "model": model_name,
            "task": task,
            "engine": current_engine,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "logit_bias": logit_bias
            }
        }
        
        # Add SVGs to results
        if num_generations == 1:
            svg_content = generated_svgs[0]
            results["svg"] = svg_content
            if return_base64:
                results["svg_base64"] = self.svg_to_base64(svg_content)
            if return_png:
                results["png_base64"] = self.svg_to_png_base64(svg_content)
        else:
            results["svgs"] = generated_svgs
            if return_base64:
                results["svgs_base64"] = [self.svg_to_base64(svg) for svg in generated_svgs]
            if return_png:
                results["pngs_base64"] = [self.svg_to_png_base64(svg) for svg in generated_svgs]
        
        return results
    
    def post_process_svg(self, svg_text):
        """Clean up and format the generated SVG"""
        # Ensure the SVG is properly formatted
        if not svg_text.startswith("<svg"):
            svg_start = svg_text.find("<svg")
            if svg_start != -1:
                svg_text = svg_text[svg_start:]
            else:
                return ""  # No SVG tag found
        
        if not svg_text.endswith("</svg>"):
            svg_end = svg_text.rfind("</svg>")
            if svg_end != -1:
                svg_text = svg_text[:svg_end+6]
            else:
                # Try to close the SVG tag if it's open but not closed
                if "<svg" in svg_text and "</svg>" not in svg_text:
                    svg_text += "</svg>"
        
        # Fix common SVG issues
        # 1. Remove XML declaration if present (can cause issues in some viewers)
        if svg_text.startswith("<?xml"):
            xml_end = svg_text.find("?>")
            if xml_end != -1:
                svg_text = svg_text[xml_end+2:].strip()
        
        # 2. Ensure SVG has proper namespace
        if 'xmlns="http://www.w3.org/2000/svg"' not in svg_text and "<svg" in svg_text:
            svg_text = svg_text.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        
        # 3. Fix common errors in path data
        svg_text = svg_text.replace("NaN", "0")
        
        return svg_text
    
    def svg_to_base64(self, svg_text):
        """Convert SVG text to base64 encoded string"""
        if not svg_text:
            return ""
        return base64.b64encode(svg_text.encode('utf-8')).decode('utf-8')
    
    def svg_to_png_base64(self, svg_text):
        """Convert SVG to PNG and return as base64"""
        if not svg_text:
            return ""
            
        try:
            # Use CairoSVG to convert SVG to PNG
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
            return base64.b64encode(png_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting SVG to PNG: {str(e)}")
            return ""
