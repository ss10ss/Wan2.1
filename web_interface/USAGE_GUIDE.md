# Wan2.1 Web Interface Guide

This guide provides detailed instructions on how to use the Wan2.1 web interface for generating videos using AI.

## Getting Started

1. Make sure you have set up the Wan2.1 models according to the main project instructions.
2. Install the web interface dependencies:
   ```
   cd web_interface
   npm install
   ```
3. Start the server:
   ```
   npm start
   ```
4. Open your browser and navigate to http://localhost:3000

## Text-to-Video Generation

Generate videos from text descriptions:

1. Navigate to the "Text to Video" section from the homepage or the navigation bar.
2. Enter a detailed text prompt describing the video you want to generate.
3. Adjust the generation parameters:
   - **Resolution**: Select output resolution (higher resolutions require more VRAM)
   - **Steps**: Number of diffusion steps (higher values = better quality but slower generation)
   - **Guidance Scale**: How closely to follow the text prompt (higher values = more literal interpretation)
   - **Shift Scale**: Controls motion intensity (higher values = more motion)
   - **Seed**: Optional value for reproducible results (leave at -1 for random)
   - **Negative Prompt**: Elements to avoid in the generated video
4. Click the "Generate Video" button and wait for the process to complete.
5. Once generated, you can view and download the video from the results page.

Example prompts:
- "A serene lake at sunset with mountains in the background, cinematic 4k"
- "A cute cat playing with a ball of yarn on a wooden floor, detailed, soft lighting"

## Image-to-Video Generation

Convert a still image into a dynamic video:

1. Navigate to the "Image to Video" section.
2. Upload an image file (JPG or PNG, max 10MB).
3. Optionally enter a prompt to guide the video generation.
4. Adjust the generation parameters (similar to Text-to-Video).
5. Click "Generate Video" and wait for processing.
6. The result page will show both the original image and the generated video.

Tips for good results:
- Use high-quality, well-lit images
- For better control over motion, use a descriptive prompt
- Adjust the shift scale to control the intensity of motion

## First-Last Frame to Video

Create a video transition between two images:

1. Navigate to the "First-Last Frame" section.
2. Upload two images: the first frame and the last frame of your desired video.
3. Adjust the resolution, steps, and guidance scale.
4. Click "Generate Video" and wait for processing.
5. The result page will display both frames and the generated video.

This mode is excellent for creating:
- Character animation transitions
- Scene transitions
- Object transformations

## Understanding Parameters

- **Resolution**: Higher resolution produces more detailed videos but requires more VRAM and processing time.
- **Steps**: More steps result in higher quality but longer generation time. 25-30 steps is a good balance.
- **Guidance Scale**: Controls how closely the generation follows the prompt.
  - Lower values (5-7): More creative but might deviate from the prompt
  - Higher values (7-9): More literal interpretation of the prompt
- **Shift Scale**: Controls motion intensity.
  - For Text-to-Video: 1.0 is standard
  - For Image-to-Video: 5.0 is recommended as a starting point
- **Seed**: For reproducible results, set a specific seed value. Using the same seed with the same parameters will produce similar results.

## Troubleshooting

- **Video generation fails**: Check that your GPU has sufficient VRAM for the selected resolution
- **Slow generation**: Reduce resolution or number of steps
- **Poor quality results**: Increase steps, adjust guidance scale, or improve your prompt
- **Out of memory errors**: Close other GPU-intensive applications or reduce resolution

## System Requirements

- Node.js 14+
- Python 3.8+
- CUDA-compatible GPU (minimum 8GB VRAM recommended)
- Storage space for generated videos

## Extending the Interface

Developers can extend the web interface by:
1. Adding new routes in app.js
2. Creating new EJS templates in the views folder
3. Adding new Python scripts in the scripts folder for additional Wan2.1 features
