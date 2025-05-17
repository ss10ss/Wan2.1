const express = require('express');
const path = require('path');
const multer = require('multer');
const { exec } = require('child_process');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const bodyParser = require('body-parser');
const session = require('express-session');
const flash = require('connect-flash');

// Initialize app
const app = express();
const port = process.env.PORT || 3000;

// Configure middleware
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use(session({
  secret: 'wan-video-secret',
  resave: false,
  saveUninitialized: true
}));
app.use(flash());

// Set up EJS as view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, 'public/uploads/'));
  },
  filename: function (req, file, cb) {
    const uniqueFilename = `${uuidv4()}${path.extname(file.originalname)}`;
    cb(null, uniqueFilename);
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: function (req, file, cb) {
    // Accept images only
    if (!file.originalname.match(/\.(jpg|jpeg|png|JPG|JPEG|PNG)$/)) {
      return cb(new Error('Only image files are allowed!'), false);
    }
    cb(null, true);
  }
});

// Routes
app.get('/', (req, res) => {
  res.render('index', {
    title: 'Wan2.1 - Video Generation',
    message: req.flash('message'),
    error: req.flash('error')
  });
});

// Text to Video route
app.get('/text-to-video', (req, res) => {
  res.render('text-to-video', {
    title: 'Text to Video - Wan2.1',
    message: req.flash('message'),
    error: req.flash('error')
  });
});

app.post('/text-to-video', (req, res) => {
  const { prompt, resolution, steps, guideScale, shiftScale, seed, negativePrompt } = req.body;
  
  // Generate a unique ID for this generation
  const generationId = uuidv4();
  const outputPath = path.join(__dirname, 'public/generated', `${generationId}.mp4`);
  
  // Prepare the command to run the Python script
  const pythonScript = path.join(__dirname, '../gradio/t2v_14B_singleGPU.py');
  const command = `python ${pythonScript} --prompt "${prompt}" --resolution "${resolution}" --steps ${steps} --guide_scale ${guideScale} --shift_scale ${shiftScale} --seed ${seed || -1} --n_prompt "${negativePrompt || ''}" --output "${outputPath}"`;
  
  // Execute the command
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`Execution error: ${error}`);
      req.flash('error', 'An error occurred during video generation.');
      return res.redirect('/text-to-video');
    }
    
    console.log(`Generation output: ${stdout}`);
    
    if (fs.existsSync(outputPath)) {
      req.flash('message', 'Video generated successfully!');
      return res.render('result', {
        title: 'Generation Result - Wan2.1',
        videoPath: `/generated/${generationId}.mp4`,
        prompt: prompt
      });
    } else {
      req.flash('error', 'Failed to generate video.');
      return res.redirect('/text-to-video');
    }
  });
});

// Image to Video route
app.get('/image-to-video', (req, res) => {
  res.render('image-to-video', {
    title: 'Image to Video - Wan2.1',
    message: req.flash('message'),
    error: req.flash('error')
  });
});

app.post('/image-to-video', upload.single('image'), (req, res) => {
  if (!req.file) {
    req.flash('error', 'Please upload an image.');
    return res.redirect('/image-to-video');
  }
  
  const { prompt, resolution, steps, guideScale, shiftScale, seed, negativePrompt } = req.body;
  const imagePath = path.join('/uploads', req.file.filename);
  const absoluteImagePath = path.join(__dirname, 'public', imagePath);
  
  // Generate a unique ID for this generation
  const generationId = uuidv4();
  const outputPath = path.join(__dirname, 'public/generated', `${generationId}.mp4`);
  
  // Prepare the command to run the Python script
  const pythonScript = path.join(__dirname, '../gradio/i2v_14B_singleGPU.py');
  const command = `python ${pythonScript} --image "${absoluteImagePath}" --prompt "${prompt}" --resolution "${resolution}" --steps ${steps} --guide_scale ${guideScale} --shift_scale ${shiftScale} --seed ${seed || -1} --n_prompt "${negativePrompt || ''}" --output "${outputPath}"`;
  
  // Execute the command
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`Execution error: ${error}`);
      req.flash('error', 'An error occurred during video generation.');
      return res.redirect('/image-to-video');
    }
    
    console.log(`Generation output: ${stdout}`);
    
    if (fs.existsSync(outputPath)) {
      req.flash('message', 'Video generated successfully!');
      return res.render('result', {
        title: 'Generation Result - Wan2.1',
        videoPath: `/generated/${generationId}.mp4`,
        prompt: prompt,
        imagePath: imagePath
      });
    } else {
      req.flash('error', 'Failed to generate video.');
      return res.redirect('/image-to-video');
    }
  });
});

// First-Last Frame to Video route
app.get('/fl-to-video', (req, res) => {
  res.render('fl-to-video', {
    title: 'First-Last Frame to Video - Wan2.1',
    message: req.flash('message'),
    error: req.flash('error')
  });
});

app.post('/fl-to-video', upload.fields([
  { name: 'firstFrame', maxCount: 1 },
  { name: 'lastFrame', maxCount: 1 }
]), (req, res) => {
  if (!req.files || !req.files['firstFrame'] || !req.files['lastFrame']) {
    req.flash('error', 'Please upload both first and last frame images.');
    return res.redirect('/fl-to-video');
  }
  
  const firstFramePath = path.join('/uploads', req.files['firstFrame'][0].filename);
  const lastFramePath = path.join('/uploads', req.files['lastFrame'][0].filename);
  const absoluteFirstFramePath = path.join(__dirname, 'public', firstFramePath);
  const absoluteLastFramePath = path.join(__dirname, 'public', lastFramePath);
  
  const { resolution, steps, guideScale, seed } = req.body;
  
  // Generate a unique ID for this generation
  const generationId = uuidv4();
  const outputPath = path.join(__dirname, 'public/generated', `${generationId}.mp4`);
  
  // Prepare the command to run the Python script
  const pythonScript = path.join(__dirname, '../gradio/fl2v_14B_singleGPU.py');
  const command = `python ${pythonScript} --first "${absoluteFirstFramePath}" --last "${absoluteLastFramePath}" --resolution "${resolution}" --steps ${steps} --guide_scale ${guideScale} --seed ${seed || -1} --output "${outputPath}"`;
  
  // Execute the command
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`Execution error: ${error}`);
      req.flash('error', 'An error occurred during video generation.');
      return res.redirect('/fl-to-video');
    }
    
    console.log(`Generation output: ${stdout}`);
    
    if (fs.existsSync(outputPath)) {
      req.flash('message', 'Video generated successfully!');
      return res.render('result', {
        title: 'Generation Result - Wan2.1',
        videoPath: `/generated/${generationId}.mp4`,
        firstFramePath: firstFramePath,
        lastFramePath: lastFramePath
      });
    } else {
      req.flash('error', 'Failed to generate video.');
      return res.redirect('/fl-to-video');
    }
  });
});

// Create directories if they don't exist
const dirs = [
  path.join(__dirname, 'public/uploads'),
  path.join(__dirname, 'public/generated')
];

dirs.forEach(dir => {
  if (!fs.existsSync(dir)){
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
