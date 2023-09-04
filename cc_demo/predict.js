// =====================================================
// Get references to HTML elements
// =====================================================
const imageInput      = document.getElementById('imageInput');
const selectedImage   = document.getElementById('selectedImage');

const loadModelButton = document.getElementById('loadModelButton');
var predictButton     = document.getElementById('predictButton');
var heatmapContainer  = document.getElementById('heatmapContainer');

let model; // Declare a global variable to hold the loaded model
let modelPath = 'https://huggingface.co/muasifk/CSRNet_lite/resolve/main/CSRNet_lite.onnx';

// =====================================================
// Load image
// =====================================================
function imageInputButton(event) {
   const file = event.target.files[0]; // seelct first file
   if (file) {
       const imageURL = URL.createObjectURL(file);
       selectedImage.src = imageURL;
       
       const canvas = document.createElement('canvas');
       const ctx    = canvas.getContext('2d');
       // Set the canvas dimensions to match the image dimensions
       canvas.width = selectedImage.width;
       canvas.height= selectedImage.height;
       // Draw the image onto the canvas
       ctx.drawImage(selectedImage, 0, 0);
       // Get the pixel data from the canvas
       const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
       console.log('Selected Image Width:', imageData);
       return selectedImage;
   }
}
// Attach the event listener to the image input
imageInput.addEventListener('change', imageInputButton);

//imageInput.addEventListener('change', function(event) {
////    const selectedImageElement = imageInputButton(event);
//    if (selectedImageElement) {
//        // Example: Display the width of the selected image
//        console.log('Selected Image Width:', selectedImage.width);
//    } else {
//        console.log('No image selected.');
//    }
//});


// =====================================================
// Load model
// =====================================================
class modelLoaderButton {
   constructor() {
       this.model = null; // Initialize model to null
   }

   async loadModel(modelPath) {
       try {
           // Initialize ONNX Runtime Web
//           await ort.initialize({ backendHint: 'webgl' }); // webgl, WASM

           // Load the ONNX model
           this.model = ort.InferenceSession.create(modelPath);

           // Save the model to local storage for future use
           localStorage.setItem('onnxModel', modelPath);
           console.log('Model loaded successfully');
       } catch (error) {
           console.error('Error loading the model:', error);
           throw error;
       }
   }

   getModel() {
       return this.model;
   }
}

// Create an instance of the ModelLoader class
const modelLoader = new modelLoaderButton();

// Handle the "Load Model" button click
loadModelButton.addEventListener('click', async () => {
   await modelLoader.loadModel(modelPath);
});



// =====================================================
// Predict Button
// =====================================================
predictButton.addEventListener('click', async () => {
   try {
       if (!modelLoader.getModel()) {
           window.alert("Model has not been loaded yet.");
           return;
       }
       const predictions = await predict(selectedImage); // Pass the selected image through predict()
       displayHeatmap(predictions); // Display the heatmap based on predictions
   } catch (error) {
       // Handle errors
   }
});


// =====================================================
// Load the ONNX model and perform inference
//====================================================== 
async function predict(img) {
   try {
//       const ort = require('onnxruntime-web');
       // Load the ONNX model
//       const model = await ort.InferenceSession.create(modelUrl);
       const model = modelLoader.getModel();

       
       // Prepare the input tensor (assuming your model expects a tensor)
//       const img_tensor = new ort.Tensor(new Float32Array(img.data), 'float32', [1, 3, img.height, img.width]);
       console.log('>>>>>>', img)
       const img_tensor = new ort.Tensor('float32', new Float32Array(img.data), [1, 3, img.height, img.width]);  // new ort.Tensor("float32", img_arr, dims);
       console.log('>>> Input Tensor Shape:', img_tensor.shape);

       // Run inference
       const outputMap = await model.run({ input: img_tensor });
       console.log('Output Tensor Shape:', outputMap.shape);

       // Extract the output tensor (modify 'output' to match your model's output name)
       const outputTensor = outputMap.values().next().value;
       console.log('Output Tensor Shape:', outputTensor.shape);

       // Get the predictions as a JavaScript array
       const predictions = Array.from(outputTensor.data);

       // You can process the predictions here
       console.log('Prediction is ready')
       return predictions;
       
   } catch (error) {
       console.error('Error during inference:', error);
       throw error;
   }
}

// =====================================================
// Define a function to display the heatmap
// =====================================================
function displayHeatmap(predictions) {
   const canvas = document.createElement('canvas');
   canvas.width = selectedImage.width; // Match canvas size to image size
   canvas.height = selectedImage.height;
   heatmapContainer.innerHTML = ''; // Clear previous content
   heatmapContainer.appendChild(canvas);
   const ctx = canvas.getContext('2d');
   const imageData = ctx.createImageData(canvas.width, canvas.height);
   // Modify the imageData based on your heatmap format
   // Example: Fill the red channel with heatmap values
   for (let i = 0; i < canvas.width * canvas.height; i++) {
       const value = predictions[i]; // Replace with how your prediction data is structured
       imageData.data[i * 4] = value * 255; // Red channel
       imageData.data[i * 4 + 3] = 255; // Alpha channel
   }
   ctx.putImageData(imageData, 0, 0);
}

