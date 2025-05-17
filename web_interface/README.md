# Wan2.1 Web Interface

This is a web interface for the Wan2.1 video generation model, built with Express.js and EJS templating engine. The interface provides an easy-to-use web UI for creating videos with Wan2.1's powerful AI models.

## Features

- **Text-to-Video**: Generate videos from textual descriptions
- **Image-to-Video**: Convert still images into dynamic videos
- **First-Last-Frame-to-Video**: Create videos from first and last frames
- Responsive and modern UI with Bootstrap
- File upload support
- Video preview and download
- Parameter customization for video generation

## Installation

1. Make sure you have Node.js (14+) and npm installed
2. Install the dependencies:

```bash
cd web_interface
npm install
```

3. Make sure you have the Python dependencies installed for Wan2.1:

```bash
pip install -r ../requirements.txt
```

## Usage

Start the server:

```bash
cd web_interface
npm start
```

For development with auto-reloading:

```bash
cd web_interface
npm run dev
```

The web interface will be available at http://localhost:3000

## Available Routes

- `/` - Home page with overview of available features
- `/text-to-video` - Generate videos from text descriptions
- `/image-to-video` - Convert images to videos
- `/fl-to-video` - Create videos from first and last frames

## Directory Structure

- `app.js` - Main Express.js application file
- `public/` - Static files
  - `css/` - Stylesheets
  - `js/` - Client-side JavaScript
  - `images/` - Images used in the interface
  - `uploads/` - Temporary storage for uploaded images
  - `generated/` - Output directory for generated videos
- `scripts/` - Python scripts that interface with Wan2.1 models
  - `t2v_script.py` - Text-to-video script
  - `i2v_script.py` - Image-to-video script
  - `fl2v_script.py` - First-Last-Frame-to-video script
- `views/` - EJS templates
  - `index.ejs` - Home page
  - `text-to-video.ejs` - Text to video page
  - `image-to-video.ejs` - Image to video page
  - `fl-to-video.ejs` - First-Last frame to video page
  - `result.ejs` - Video result display page
  - `partials/` - Reusable EJS components
    - `head.ejs` - HTML head section
    - `header.ejs` - Navigation header
    - `footer.ejs` - Page footer
    - `scripts.ejs` - Common JavaScript includes

## Requirements

- Node.js 14+
- Python 3.8+
- CUDA-compatible GPU for video generation
- Required disk space for video storage

## Customization

- Modify the EJS templates in `views/` to customize the UI
- Adjust the video generation parameters in `scripts/` directory
- Add your own CSS styles in `public/css/style.css`
- Extend the functionality with new routes in `app.js`

## Integrating with Other Websites

To integrate this web interface with another website:

1. **API Approach**:
   - Create additional API routes in `app.js` that return JSON responses
   - Make AJAX calls from your main website to these endpoints

2. **Iframe Embedding**:
   - Host this application on a subdomain
   - Embed specific pages using iframes in your main website

3. **Direct Integration**:
   - Copy the relevant EJS templates and adapt them to your existing website template system
   - Make sure to include the necessary scripts and stylesheets
   - Update your server-side code to call the Python scripts in a similar way

## For Developers

If you're developing an extension to this web interface:

1. Use nodemon for hot reloading during development:
   ```
   npx nodemon app.js
   ```

2. The `run_in_terminal` function in app.js handles Python script execution. This can be extended to support more Wan2.1 features.

3. The Python scripts in the `scripts/` directory serve as adapters between the web interface and the actual Wan2.1 Python modules.

## License

The Wan2.1 Web Interface is under the same license as the Wan2.1 project.
