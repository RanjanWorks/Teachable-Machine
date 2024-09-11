const VIDEO = document.getElementById("webcam");
const eye = document.getElementById("eye");
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");
const RESET_BUTTON = document.getElementById("reset");
let editClass = document.querySelectorAll("#edit");
const TRAIN_BUTTON = document.getElementById("train");
let toaddafter = document.querySelector(".classes");
let alladdbtn = document.querySelectorAll(".add");
let wrapper = document.querySelector(".wrapper");
let dataC = document.querySelectorAll("[data-c]");
const modelSaveButton = document.getElementById("saveModel");
let datavce = document.querySelector("[data-vce]");
let expand = document.getElementById("expand");
let bars = document.querySelectorAll(".bar");
let names = document.querySelectorAll(".names");
let trainbar = document.querySelector(".pbar");
let classcount = 2;
let dataCount = 1;
const preview = document.querySelectorAll(".preview");
let datapr = document.querySelector("[data-pr]");
const addClass = document.getElementById("addcls");
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];
let voiceState = false;

expand.addEventListener("click", () => {
  wrapper.classList.toggle("shrink");
});

let hiddenClasses = document.querySelectorAll("[data-hidden]");
let box1 = document.querySelector("[data-show]");
let box2 = document.querySelector("[data-hide]");
ENABLE_CAM_BUTTON.addEventListener("click", enableCam);
TRAIN_BUTTON.addEventListener("click", trainAndPredict);
RESET_BUTTON.addEventListener("click", reset);
var videoPlaying = false;
let loader = document.querySelector(".loader");

// Just add more buttons in HTML to allow classification of more classes of data!

let dataCollectorButtons = document.querySelectorAll(".add");
dataCollectorButtons.forEach((btn) => {
  btn.addEventListener("mousedown", gatherDataForClass);
  btn.addEventListener("mouseup", gatherDataForClass);
  CLASS_NAMES.push(btn.getAttribute("data-name"));
});

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

function removeOpacity() {
  alladdbtn.forEach((btn) => {
    btn.classList.remove("opc");
  });
}
function addOpacity() {
  alladdbtn.forEach((btn) => {
    btn.classList.add("opc");
  });
}

async function loadMobileNetFeatureModel() {
  const URL =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
  let loadingPercentage = 0;

  mobilenet = await tf.loadGraphModel(URL, {
    fromTFHub: true,
    onProgress: function (fraction) {
      loadingPercentage = Math.round(fraction * 100);
      document.querySelector(
        "[data-load]"
      ).innerHTML = `Loading... ${loadingPercentage}%`;
    },
  });
  loader.style.display = "none";
  tf.tidy(function () {
    let answer = mobilenet.predict(
      tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
    );
  });
}

loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(
  tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
);
model.add(
  tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
);

model.summary();

model.compile({
  optimizer: "adam",

  loss:
    CLASS_NAMES.length === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ["accuracy"],
});

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

let isCameraFlipped = false;
let mediaStream;

function enableCam() {
  showCamera("none", "flex");

  if (hasGetUserMedia()) {
    // getUserMedia parameters.
    const constraints = {
      video: {
        facingMode: isCameraFlipped ? "user" : "environment", // Use the front camera if flipped
        width: 640,
        height: 480,
      },
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      mediaStream = stream;
      VIDEO.srcObject = stream;
      removeOpacity();
      VIDEO.addEventListener("loadeddata", function () {
        VIDEO.style.transform = "scaleX(-1)";
        // VIDEO.style.transform = isCameraFlipped ? "scaleX(-1)" : "scaleX(1)";
        videoPlaying = true;
      });
    });
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
}

// Button click event listener for flipping the camera
const flipCameraButton = document.getElementById("record");
flipCameraButton.addEventListener("click", function () {
  isCameraFlipped = !isCameraFlipped; // Toggle the camera flipped flag
  mediaStream.getTracks().forEach(function (track) {
    track.stop(); // Stop the current media stream
  });
  enableCam(); // Re-enable the camera with the updated settings
});

// Initial camera setup

function showCamera(hide, show) {
  box1.style.display = hide;
  box2.style.display = show;
}

document.getElementById("can").addEventListener("click", close1);

function close1() {
  showCamera("flex", "none");
  stopVideoStream();
  addOpacity();
  reset();
  resetBarsAndBtns();
}

function stopVideoStream() {
  const stream = VIDEO.srcObject;
  const tracks = stream.getTracks();
  console.log(VIDEO);
  tracks.forEach(function (track) {
    track.stop();
  });

  VIDEO.srcObject = null;
  videoPlaying = false;
}

let toapend;
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute("data-1hot"));
  toapend = this.parentElement.querySelector(".preview");
  gatherDataState =
    gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

function calculateFeaturesOnCurrentFrame() {
  return tf.tidy(function () {
    // Grab pixels from current VIDEO frame.
    let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);

    // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
    let resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );

    let normalizedTensorFrame = resizedTensorFrame.div(255);

    return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
  });
}

var editableElements = document.getElementsByClassName("cls");
editableElements.forEach((elem, i) => {
  elem.addEventListener("input", (e) => {
    handleEdit(e, i);
  });
});

function handleEdit(event, i) {
  var editedText = event.target.innerText;
  CLASS_NAMES.splice(i, 1, editedText);
}

// Create a canvas element to convert the image data to a data URL
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");
canvas.width = MOBILE_NET_INPUT_WIDTH;
canvas.height = MOBILE_NET_INPUT_HEIGHT;

function dataGatherLoop() {
  // Only gather data if webcam is on and a relevant button is pressed.
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    // Ensure tensors are cleaned up.
    let imageFeatures = calculateFeaturesOnCurrentFrame();

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    // Draw the video frame on the canvas
    context.drawImage(
      VIDEO,
      0,
      0,
      MOBILE_NET_INPUT_WIDTH,
      MOBILE_NET_INPUT_HEIGHT
    );

    // Create an <img> element and set its src attribute to the captured image data
    const capturedImage = document.createElement("img");
    capturedImage.src = canvas.toDataURL();

    // Append the image element to the imageContainer div

    toapend.scrollTop = toapend.scrollHeight;
    toapend.append(capturedImage);
    // Initialize array index element if currently undefined
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    // Increment counts of examples for user interface to show
    examplesCount[gatherDataState]++;

    for (let n = 0; n < CLASS_NAMES.length; n++) {
      dataC[n].innerHTML = examplesCount[n] || "0";
    }

    window.requestAnimationFrame(dataGatherLoop);
  }
}

async function trainAndPredict() {
  datapr.innerHTML = "Initializing...";
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  let inputsAsTensor = tf.stack(trainingDataInputs);

  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  predict = true;
  predictLoop();
}

function logProgress(epoch, logs) {
  let dat = Math.round(epoch * 11.1) + "%";
  trainbar.style.width = dat;
  datapr.innerHTML = dat;
  if (epoch >= 9) {
    setTimeout(() => {
      expand.click();
    }, 1000);
    datapr.innerHTML = "Completed";
    flipCameraButton.setAttribute("disabled", "true");
  }
}

let previousClassName = "";

function predictLoop() {
  if (predict) {
    tf.tidy(function () {
      let imageFeatures = calculateFeaturesOnCurrentFrame();
      let prediction = model.predict(imageFeatures.expandDims()).squeeze();
      let predictionArray = prediction.arraySync();

      for (let i = 0; i < CLASS_NAMES.length; i++) {
        let className = CLASS_NAMES[i];
        let confidence = Math.round(predictionArray[i] * 100);
        names[i].innerHTML = className;
        bars[i].style.width = confidence + "%";
        bars[i].innerHTML = confidence + "%";

        if (confidence >= 99 && className !== previousClassName && voiceState) {
          speak(className);
          previousClassName = className;
        }
      }
    });

    window.requestAnimationFrame(predictLoop);
  }
}

// Function to speak the className
function speak(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = window.speechSynthesis.getVoices()[12];
  speechSynthesis.speak(utterance);
}

// Call the predictLoop function to start the prediction loop

function reset() {
  predict = false;
  examplesCount.splice(0);
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.splice(0);
  trainingDataOutputs.splice(0);
  clearImages();
  resetBarsAndBtns();
}

function clearImages() {
  document.querySelectorAll("img").forEach((i) => i.remove());
}

function resetBarsAndBtns() {
  bars.forEach((b) => (b.style.width = 0));
  dataC.forEach((d) => (d.innerHTML = 0));
  datapr.innerHTML = "Train";
  trainbar.style.width = 0;

  flipCameraButton.disabled = false;
  previousClassName = "";
}

let vce = document.getElementById("voice");
vce.addEventListener("change", (e) => {
  if (e.target.checked) {
    voiceState = true;
    datavce.innerHTML = "volume_up";
  } else {
    datavce.innerHTML = "volume_off";
    voiceState = false;
  }
});

editClass.forEach((cls) => {
  cls.addEventListener("click", () => {
    let select = cls.parentElement.parentElement;
    let h33 = select.querySelector(".cls");
    selectText(h33);
    h33.focus();
  });
});

function selectText(element) {
  var range, selection;

  if (document.body.createTextRange) {
    // For Internet Explorer
    range = document.body.createTextRange();
    range.moveToElementText(element);
    range.select();
  } else if (window.getSelection) {
    // For modern browsers
    selection = window.getSelection();
    range = document.createRange();
    range.selectNodeContents(element);
    selection.removeAllRanges();
    selection.addRange(range);
  }
}
