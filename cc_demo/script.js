//================================================================
// Get references to the button and canvas elements
//================================================================

const imgContainer    = document.getElementById("img");
const dmapContainer   = document.getElementById("dmap");
const countElement    = document.getElementById("count");

const loadImageButton = document.getElementById('loadImageButton');
const loadModelButton = document.getElementById('loadModelButton');
const predictButton   = document.getElementById('predictButton');

const imageCanvas     = document.getElementById('imageCanvas');
const processedCanvas = document.getElementById('processedCanvas');
const ctx             = imageCanvas.getContext('2d');
const ctx2            = processedCanvas.getContext('2d');

// Define the model URL
//let modelUrl = 'https://huggingface.co/muasifk/CSRNet_lite/resolve/main/CSRNet_lite.onnx';
//let modelUrl = 'https://huggingface.co/muasifk/MCNN/resolve/main/MCNN.onnx';
let modelUrl = 'https://huggingface.co/muasifk/CSRNet/resolve/main/model1_A.onnx'



imageCanvas.width = 1024; 
imageCanvas.height = 768;


//================================================================
// Load and Display image
//================================================================
//loadImageButton.onclick = // Try this
// Add a click event listener to the "Load Image" button
loadImageButton.addEventListener('click', () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            // Add an event listener for when the FileReader finishes loading the image
            reader.onload = (e) => {
                const img = new Image();
                img.src = e.target.result;

                const img2 = img; // For display
                // Draw the image on the canvas (img is resized to canvas size)
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height); // display=none in css to not show
                    console.log('Image is resized to:', img.width, img.height);//, img.channels);
                    
                    // Plot using html
                    // Set the width and height of the "img" element
                    // Calculate the desired dimensions (e.g., scaling down by a factor)
                    const desiredWidth = img2.width / 4; 
                    const desiredHeight = img2.height / 4; 
                    imgContainer.style.width = `${desiredWidth}px`;
                    imgContainer.style.height = `${desiredHeight}px`;
                    dmapContainer.style.width = `${desiredWidth}px`;
                    dmapContainer.style.height = `${desiredHeight}px`;
                
                    imgContainer.innerHTML = "";
                    imgContainer.appendChild(img2);

                    
//                    loadImageButton.innerText = "Load Image";
//                    loadImageButton.disabled = false;
                    
                    console.log('Resized plotted as', img.width, img.height);//, img.channels);
                    loadImageButton.style.backgroundColor = 'green';
                    predictButton.style.backgroundColor   = '#333';
                    
                };
            };

            // Read the selected file as a data URL
            reader.readAsDataURL(file);
        }
    });

    // Trigger a click event on the file input to open the file selection dialog
    fileInput.click();
});

//================================================================
// Load model
//================================================================

let model = null;
let modelLoaded = false;

async function loadModel(modelUrl) {
    try {
        const cachedModel = localStorage.getItem(modelUrl);
        if (cachedModel) {
            console.log('Loading model from local storage...');
            const model = await ort.InferenceSession.create(cachedModel);
            loadModelButton.style.backgroundColor = 'green';
            console.log('Model loaded from local storage.');
            return model;
        } else {
            console.log(`Model not found in local storage. Loading from ${modelUrl}...`);
            const model = await ort.InferenceSession.create(modelUrl);
            loadModelButton.style.backgroundColor = 'green';
            console.log('Model loaded from URL.');
            localStorage.setItem('onnxModel', modelUrl);
            return model;
        }
    } catch (error) {
        console.error('Error loading the model:', error);
        throw error;
    }
}

// Usage: Call this function when you want to load the model
async function main() {
    try {
        const model = await loadModel(modelUrl);
        modelLoaded = true;
        return model; // Use the loaded model for inference or further processing
    } catch (error) {
    }
}

// Call the main function to load the model when needed (load when the page loads)
//let model = main();
// Load model when "Load Model" button is clicked.
loadModelButton.addEventListener('click', async () => {
//    model = await main();
    if (!modelLoaded) {
        loadModelButton.innerText = "Loading model...";
        model = await main();
        loadModelButton.innerText = "Loading complete";
    }
});



//================================================================
// Preprocessing images
//================================================================

async function preprocessInput(inputImage) {
//    console.log('Before preprocess', inputImage.data);
    const width  = inputImage.width;
    const height = inputImage.height;
    const data   = inputImage.data;
    const rgbData = new Float32Array(width * height * 3); // 3 channels for RGB
    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        rgbData[j] = r/255;
        rgbData[j + 1] = g/255;
        rgbData[j + 2] = b/255;
    }
//    console.log('After preprocess (input to model)', rgbData);
    return rgbData;
}


//================================================================
// Output to heatmap
//================================================================

function postprocess(tensor) {
  const shape = tensor.dims;
  const numRows = shape[2]; // Assuming height is the third dimension
  const numCols = shape[3]; // Assuming width is the fourth dimension
  const data2D = [];
  const data = tensor.data;
  for (let i = 0; i < numRows; i++) {
    const row = [];
    for (let j = 0; j < numCols; j++) {
      const dataIndex = i * numCols + j;
      const value = data[dataIndex];
      row.push(value);
    }
    data2D.push(row);
  }
  return data2D;
}


function createRandom2DArray(rows, cols) {
  const randomArray = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      // Generate a random float between 0 and 1
      const randomValue = Math.random();
      row.push(randomValue);
    }
    randomArray.push(row);
  }
  return randomArray;
}





//================================================================
// Run Prediction
//================================================================
// Function to perform inference and display the output
async function predictImage(model, inputImage) {
    try {
        console.log('Running inference..')
        // Preprocess (remove alpha channel)
        const processedImage = await preprocessInput(inputImage);
        const imageArray = new Float32Array(processedImage);
        var inputTensor = new ort.Tensor(imageArray, [1, 3, 768, 1024]);
//        console.log('Input to model (inputTensor)', inputTensor); // Use .dims for dimensions
        
        // Run inference
        const outputMap = await model.run({ input: inputTensor });
        
        // Postprocessing
        const outputTensor = outputMap.output;
//        console.log('Model output (outputTensor)', outputTensor);
        
        // Plot density map
        const data2D = postprocess(outputTensor);
//        console.log('data2D', data2D);
        const trace = {z: data2D, type: 'heatmap', colorscale: 'Jet', showscale: false};
        const layout = {xaxis: {tickvals: [], ticktext: [], showticklabels: false}, yaxis: {tickvals: [], ticktext: [], showticklabels: false}, autosize: true};
        const data = [trace];
        const config = {responsive: false};
        Plotly.newPlot('dmap', data, layout, {displayModeBar: false});

        // Manually adjust the size of the plot to fit within the container
//        const plotContainer = document.getElementById('dmap');
//        const plot = plotContainer.getElementsByClassName('js-plotly-plot')[0];
//        plot.style.width = '100%'; // Set the width to 100% to fill the container
//        plot.style.height = '100%'; // Set the height to 100% to fill the container
        
        var count = parseInt(outputTensor.data.reduce(function(acc, val) { return acc + val; }, 0)); 
        countElement.textContent = `${count}`;
//        console.log('Count:', count);
        
        // Done
        console.log('Inference complete');
    } catch (error) {
        console.error('Error during inference:', error);
        throw error;
    }
}

// Add an event listener to the "Predict" button
let inferenceDone = false;
predictButton.addEventListener('click', async () => {
    try {
        if (!model) {
            window.alert("Model has not been loaded yet.");
            return;
        }
        const inputImage = ctx.getImageData(0,0, imageCanvas.width, imageCanvas.height); // { willReadFrequently: true }
        const inputImageChannels = inputImage.data.length / (inputImage.width * inputImage.height);
//        await predictImage(model, inputImage);
        
        if (!inferenceDone) {
            predictButton.innerText = "Running inference...";
            predictButton.disabled  = true;
            predictButton.style.backgroundColor = 'orange';
            await predictImage(model, inputImage);
            predictButton.disabled  = false;
            predictButton.innerText = "Count People";
        }
            predictButton.style.backgroundColor = 'green';
            loadImageButton.style.backgroundColor = '#333';
        
    } catch (error) {
        // Handle errors
    }
});


//================================================================
// Utils
//================================================================

function getVariableInfo(variable) {
    const type = typeof variable;

    // For arrays and objects, determine their shape or size
    let shapeSize = null;
    if (Array.isArray(variable)) {
        shapeSize = `Array with ${variable.length} elements`;
    } else if (type === 'object') {
        if (variable === null) {
            shapeSize = 'Null';
        } else {
            const keys = Object.keys(variable);
            shapeSize = `Object with ${keys.length} properties`;
        }
    }

    return {
        type,
        shapeSize,
    };
}