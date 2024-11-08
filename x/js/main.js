// Initialize Go WASM
const go = new Go();
WebAssembly.instantiateStreaming(fetch("wasm/main.wasm"), go.importObject).then((result) => {
    go.run(result.instance);
});

document.getElementById('processButton').addEventListener('click', () => {
    const input = document.getElementById('inputImage');
    if (input.files.length === 0) {
        alert('Please select an image file.');
        return;
    }
    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        let img = new Image();
        img.onload = function() {
            processImage(img);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

function processImage(img) {
    let src = cv.imread(img);
    let srcData = src.data;
    let width = src.cols;
    let height = src.rows;
    let channels = src.channels();

    // Call Go/WASM function to process the image
    let processedData = ProcessImage(srcData, width, height, channels);

    // Create a new Mat from processed data
    let processedMat = cv.matFromArray(height, width, cv.CV_8UC4, processedData);

    cv.imshow('canvasOutput', processedMat);
    src.delete();
    processedMat.delete();
}
