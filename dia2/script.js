let model;

async function loadModel() {
    model = await tf.loadLayersModel('./modelo_tfjs/model.json');

    console.log('Modelo cargado');
}
//C:\examen-python-final\dia2\modelo_tfjs\model.json

loadModel();

const imageUpload = document.getElementById('imageUpload');
const imageCanvas = document.getElementById('imageCanvas');
const ctx = imageCanvas.getContext('2d');
const predictionElement = document.getElementById('prediction');

imageUpload.addEventListener('change', handleImageUpload);

function handleImageUpload(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            ctx.drawImage(img, 0, 0, 28, 28);
            predict();
        }
        img.src = e.target.result;
    }
    reader.readAsDataURL(file);
}

function preprocessImage(imageData) {
    // Convertir a escala de grises y normalizar
    const grayscaleImage = new Float32Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        const grayscale = (r + g + b) / 3;
        grayscaleImage[i / 4] = grayscale / 255.0;
    }

    // Remodelar para que coincida con la entrada del modelo (1, 28, 28, 1)
    return tf.tensor4d(grayscaleImage, [1, 28, 28, 1]);
}

async function predict() {
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const preprocessedTensor = preprocessImage(imageData);

    const predictions = model.predict(preprocessedTensor);
    const predictedClass = predictions.argMax(-1).dataSync()[0];

    const classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
    predictionElement.innerText = 'Predicción: ' + classNames[predictedClass];

    preprocessedTensor.dispose(); // Liberar memoria del tensor
}