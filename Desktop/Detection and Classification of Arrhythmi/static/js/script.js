/**
 * ECG Arrhythmia Web Application Frontend Client Script.
 * Provides functions for loading sample heartbeats, drawing signal preview charts,
 * handling form inputs, and dynamic UI interactions.
 */

document.addEventListener("DOMContentLoaded", function () {
  console.log("ECG Arrhythmia ML Application Client Script Loaded.");

  const sampleSelect = document.getElementById("sampleSelect");
  const signalInput = document.getElementById("signal_data");
  const previewCanvas = document.getElementById("signalCanvas");

  let sampleDataCache = null;

  // Load sample beats from API if element exists
  if (sampleSelect && signalInput) {
    fetch("/api/sample-beats")
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success" && data.samples) {
          sampleDataCache = data.samples;
        }
      })
      .catch((err) => console.error("Error loading sample beats:", err));

    sampleSelect.addEventListener("change", function () {
      const selectedClass = this.value;
      if (sampleDataCache && sampleDataCache[selectedClass]) {
        const sampleObj = sampleDataCache[selectedClass];
        signalInput.value = sampleObj.signal.join(", ");
        drawSignalCanvas(sampleObj.signal, selectedClass);
      }
    });
  }

  // Draw signal preview canvas
  function drawSignalCanvas(vector, labelStr) {
    if (!previewCanvas) return;
    const ctx = previewCanvas.getContext("2d");
    const width = previewCanvas.width;
    const height = previewCanvas.height;

    ctx.clearRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!vector || vector.length === 0) return;

    // Plot waveform
    ctx.beginPath();
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;

    const minVal = Math.min(...vector);
    const maxVal = Math.max(...vector);
    const range = maxVal - minVal || 1;

    for (let i = 0; i < vector.length; i++) {
      const x = (i / (vector.length - 1)) * (width - 20) + 10;
      const normY = (vector[i] - minVal) / range;
      const y = height - 15 - normY * (height - 30);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
});
