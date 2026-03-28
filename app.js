/**
 * SmileFace - Client-side face smile animation generator
 * Uses face-api.js for landmark detection and canvas for warping
 */

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.14/model/';

const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewImg = document.getElementById('preview-img');
const controls = document.getElementById('controls');
const intensitySlider = document.getElementById('intensity-slider');
const intensityValue = document.getElementById('intensity-value');
const speedSlider = document.getElementById('speed-slider');
const speedValue = document.getElementById('speed-value');
const generateBtn = document.getElementById('generate-btn');
const statusEl = document.getElementById('status');
const progressBarContainer = document.getElementById('progress-bar-container');
const progressBar = document.getElementById('progress-bar');
const resultArea = document.getElementById('result-area');
const resultGif = document.getElementById('result-gif');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');
const workCanvas = document.getElementById('work-canvas');
const ctx = workCanvas.getContext('2d');

let loadedImage = null;
let landmarks = null;
let modelsLoaded = false;
let generatedBlobUrl = null;

// --- Upload handling ---

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showStatus('Please upload an image file', true);
        return;
    }
    const reader = new FileReader();
    reader.onload = e => {
        const img = new Image();
        img.onload = () => {
            loadedImage = img;
            previewImg.src = e.target.result;
            previewImg.hidden = false;
            uploadPlaceholder.hidden = true;
            controls.hidden = false;
            resultArea.hidden = true;
            hideStatus();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// --- Sliders ---

intensitySlider.addEventListener('input', () => {
    intensityValue.textContent = intensitySlider.value;
});
speedSlider.addEventListener('input', () => {
    speedValue.textContent = speedSlider.value;
});

// --- Generate ---

generateBtn.addEventListener('click', async () => {
    if (!loadedImage) return;
    generateBtn.disabled = true;
    resultArea.hidden = true;

    try {
        showStatus('Loading face detection models...');
        showProgress(10);
        await loadModels();

        showStatus('Detecting face landmarks...');
        showProgress(25);
        landmarks = await detectLandmarks(loadedImage);

        if (!landmarks) {
            showStatus('No face detected. Please try a clearer photo with a visible face.', true);
            generateBtn.disabled = false;
            hideProgress();
            return;
        }

        showStatus('Generating smile frames...');
        showProgress(40);
        const frames = generateFrames(loadedImage, landmarks, 8, Number(intensitySlider.value));

        showStatus('Encoding GIF...');
        showProgress(70);
        const gifBlob = await encodeGif(frames, Number(speedSlider.value));

        if (generatedBlobUrl) URL.revokeObjectURL(generatedBlobUrl);
        generatedBlobUrl = URL.createObjectURL(gifBlob);
        resultGif.src = generatedBlobUrl;
        resultArea.hidden = false;
        showProgress(100);
        showStatus('Done! Your smiling loop is ready.');
    } catch (err) {
        console.error(err);
        showStatus('Error: ' + err.message, true);
    } finally {
        generateBtn.disabled = false;
    }
});

// --- Reset ---

resetBtn.addEventListener('click', () => {
    loadedImage = null;
    landmarks = null;
    previewImg.hidden = true;
    previewImg.src = '';
    uploadPlaceholder.hidden = false;
    controls.hidden = true;
    resultArea.hidden = true;
    hideStatus();
    hideProgress();
    fileInput.value = '';
});

// --- Download ---

downloadBtn.addEventListener('click', () => {
    if (!generatedBlobUrl) return;
    const a = document.createElement('a');
    a.href = generatedBlobUrl;
    a.download = 'smile_loop.gif';
    a.click();
});

// --- Model loading ---

async function loadModels() {
    if (modelsLoaded) return;
    if (typeof faceapi !== 'undefined') {
        try {
            await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
            modelsLoaded = true;
            return;
        } catch (e) {
            console.warn('face-api.js models failed to load, using heuristic fallback', e);
        }
    }
    // Fallback: models not available, will use heuristic landmarks
    modelsLoaded = true;
}

// --- Face detection ---

async function detectLandmarks(img) {
    // Try face-api.js first
    if (typeof faceapi !== 'undefined' && faceapi.nets.ssdMobilenetv1.isLoaded) {
        const maxDim = 1024;
        let detectCanvas = document.createElement('canvas');
        let scale = 1;
        if (img.width > maxDim || img.height > maxDim) {
            scale = maxDim / Math.max(img.width, img.height);
        }
        detectCanvas.width = Math.round(img.width * scale);
        detectCanvas.height = Math.round(img.height * scale);
        const dCtx = detectCanvas.getContext('2d');
        dCtx.drawImage(img, 0, 0, detectCanvas.width, detectCanvas.height);

        const detection = await faceapi
            .detectSingleFace(detectCanvas)
            .withFaceLandmarks();

        if (detection) {
            const pts = detection.landmarks.positions.map(p => ({
                x: p.x / scale,
                y: p.y / scale
            }));

            return {
                all: pts,
                jaw: pts.slice(0, 17),
                leftEyebrow: pts.slice(17, 22),
                rightEyebrow: pts.slice(22, 27),
                nose: pts.slice(27, 36),
                leftEye: pts.slice(36, 42),
                rightEye: pts.slice(42, 48),
                outerMouth: pts.slice(48, 60),
                innerMouth: pts.slice(60, 68),
            };
        }
    }

    // Fallback: heuristic face landmarks based on average face proportions
    // Assumes a roughly centered face photo (portrait)
    return generateHeuristicLandmarks(img.width, img.height);
}

function generateHeuristicLandmarks(w, h) {
    // Average face proportions (centered face)
    const cx = w * 0.5;
    const faceTop = h * 0.15;
    const faceBottom = h * 0.85;
    const faceH = faceBottom - faceTop;
    const faceW = faceH * 0.72;

    const eyeY = faceTop + faceH * 0.38;
    const noseY = faceTop + faceH * 0.58;
    const mouthY = faceTop + faceH * 0.72;
    const chinY = faceTop + faceH * 0.95;

    const eyeSpacing = faceW * 0.32;
    const mouthW = faceW * 0.38;
    const jawW = faceW * 0.5;

    // Build 68-point-compatible landmark subsets
    const jaw = [];
    for (let i = 0; i <= 16; i++) {
        const t = i / 16;
        const angle = Math.PI * t - Math.PI / 2;
        jaw.push({
            x: cx + jawW * Math.cos(angle) * (1 + 0.2 * Math.sin(angle)),
            y: faceTop + faceH * 0.3 + faceH * 0.65 * (0.5 + 0.5 * Math.sin(angle))
        });
    }

    const makeEye = (ecx, ecy, ew) => {
        const eh = ew * 0.35;
        return [
            { x: ecx - ew / 2, y: ecy },
            { x: ecx - ew / 4, y: ecy - eh / 2 },
            { x: ecx + ew / 4, y: ecy - eh / 2 },
            { x: ecx + ew / 2, y: ecy },
            { x: ecx + ew / 4, y: ecy + eh / 2 },
            { x: ecx - ew / 4, y: ecy + eh / 2 },
        ];
    };

    const eyeW = faceW * 0.22;
    const leftEye = makeEye(cx - eyeSpacing, eyeY, eyeW);
    const rightEye = makeEye(cx + eyeSpacing, eyeY, eyeW);

    // Outer mouth: 12 points around mouth
    const outerMouth = [];
    for (let i = 0; i < 12; i++) {
        const angle = (2 * Math.PI * i) / 12 - Math.PI;
        const rx = mouthW / 2;
        const ry = mouthW * 0.22;
        outerMouth.push({
            x: cx + rx * Math.cos(angle),
            y: mouthY + ry * Math.sin(angle)
        });
    }

    // Nose points
    const nose = [];
    for (let i = 0; i < 9; i++) {
        const t = i / 8;
        nose.push({
            x: cx + (i < 4 ? 0 : faceW * 0.08 * Math.sin(2 * Math.PI * t)),
            y: eyeY + (noseY - eyeY) * t
        });
    }

    return {
        all: [],
        jaw,
        leftEyebrow: leftEye.slice(0, 5).map(p => ({ x: p.x, y: p.y - eyeW * 0.5 })),
        rightEyebrow: rightEye.slice(0, 5).map(p => ({ x: p.x, y: p.y - eyeW * 0.5 })),
        nose,
        leftEye,
        rightEye,
        outerMouth,
        innerMouth: outerMouth.slice(0, 8).map(p => ({
            x: cx + (p.x - cx) * 0.6,
            y: mouthY + (p.y - mouthY) * 0.6
        })),
    };
}

// --- Frame generation with mesh warping ---

function generateFrames(img, lm, numFrames, maxIntensity) {
    const frames = [];
    const w = img.width;
    const h = img.height;

    // Limit canvas size for performance
    const maxCanvasDim = 800;
    let scale = 1;
    if (w > maxCanvasDim || h > maxCanvasDim) {
        scale = maxCanvasDim / Math.max(w, h);
    }
    const cw = Math.round(w * scale);
    const ch = Math.round(h * scale);

    workCanvas.width = cw;
    workCanvas.height = ch;

    // Precompute source image data at canvas scale
    ctx.drawImage(img, 0, 0, cw, ch);
    const srcData = ctx.getImageData(0, 0, cw, ch);

    // Scale landmarks
    const scaledLm = {
        jaw: lm.jaw.map(p => ({ x: p.x * scale, y: p.y * scale })),
        outerMouth: lm.outerMouth.map(p => ({ x: p.x * scale, y: p.y * scale })),
        innerMouth: lm.innerMouth.map(p => ({ x: p.x * scale, y: p.y * scale })),
        nose: lm.nose.map(p => ({ x: p.x * scale, y: p.y * scale })),
        leftEye: lm.leftEye.map(p => ({ x: p.x * scale, y: p.y * scale })),
        rightEye: lm.rightEye.map(p => ({ x: p.x * scale, y: p.y * scale })),
    };

    // Compute smile warp parameters from landmarks
    const mouthCenter = getCenter(scaledLm.outerMouth);
    const mouthLeft = scaledLm.outerMouth[0];   // point 48
    const mouthRight = scaledLm.outerMouth[6];  // point 54
    const mouthWidth = dist(mouthLeft, mouthRight);

    // Warp region: area around the mouth and lower face
    const jawBottom = scaledLm.jaw[8]; // chin
    const noseBottom = scaledLm.nose[6]; // nose tip (index 33)

    // Generate 8 frames with cosine intensity for seamless loop
    // intensity(t) = (1 - cos(2*PI*t/N)) / 2, peaks at t=N/2
    for (let i = 0; i < numFrames; i++) {
        const t = (1 - Math.cos(2 * Math.PI * i / numFrames)) / 2;
        const intensity = t * maxIntensity;

        const frameData = warpSmile(srcData, cw, ch, scaledLm, mouthCenter, mouthWidth, intensity);
        ctx.putImageData(frameData, 0, 0);

        // Create a canvas snapshot for gif.js
        const frameCanvas = document.createElement('canvas');
        frameCanvas.width = cw;
        frameCanvas.height = ch;
        frameCanvas.getContext('2d').drawImage(workCanvas, 0, 0);
        frames.push(frameCanvas);

        showProgress(40 + Math.round(30 * (i + 1) / numFrames));
    }

    return frames;
}

function warpSmile(srcData, w, h, lm, mouthCenter, mouthWidth, intensity) {
    const dst = new ImageData(new Uint8ClampedArray(srcData.data), w, h);
    const src = srcData.data;
    const dstData = dst.data;

    if (intensity === 0) return dst;

    // Influence radius for the smile warp
    const radius = mouthWidth * 1.8;

    // Key control points for smile:
    // - Mouth corners pull outward and slightly up
    // - Lower lip moves slightly down
    // - Cheeks move up
    const mouthLeft = lm.outerMouth[0];
    const mouthRight = lm.outerMouth[6];
    const mouthTop = lm.outerMouth[3];
    const mouthBottomOuter = lm.outerMouth[9];

    // Define warp sources: for each pixel, compute inverse warp to find source position
    // Smile = mouth corners move out+up, cheeks lift, slight squint

    const warpPoints = [];

    // Mouth corners: pull outward and up
    const cornerStrength = intensity * 0.06;
    warpPoints.push({
        cx: mouthLeft.x, cy: mouthLeft.y,
        dx: -cornerStrength * mouthWidth * 0.15,
        dy: -cornerStrength * mouthWidth * 0.12,
        r: radius * 0.7
    });
    warpPoints.push({
        cx: mouthRight.x, cy: mouthRight.y,
        dx: cornerStrength * mouthWidth * 0.15,
        dy: -cornerStrength * mouthWidth * 0.12,
        r: radius * 0.7
    });

    // Cheeks: push up (nasolabial fold effect)
    const leftCheek = {
        x: (mouthLeft.x + lm.leftEye[0].x) / 2,
        y: (mouthLeft.y + lm.leftEye[0].y) / 2 + mouthWidth * 0.15
    };
    const rightCheek = {
        x: (mouthRight.x + lm.rightEye[3].x) / 2,
        y: (mouthRight.y + lm.rightEye[3].y) / 2 + mouthWidth * 0.15
    };
    const cheekStrength = intensity * 0.04;
    warpPoints.push({
        cx: leftCheek.x, cy: leftCheek.y,
        dx: 0, dy: -cheekStrength * mouthWidth * 0.1,
        r: radius * 0.8
    });
    warpPoints.push({
        cx: rightCheek.x, cy: rightCheek.y,
        dx: 0, dy: -cheekStrength * mouthWidth * 0.1,
        r: radius * 0.8
    });

    // Lower lip slight stretch
    const lipStrength = intensity * 0.03;
    warpPoints.push({
        cx: mouthBottomOuter.x, cy: mouthBottomOuter.y,
        dx: 0, dy: lipStrength * mouthWidth * 0.05,
        r: radius * 0.4
    });

    // Upper lip slight lift
    warpPoints.push({
        cx: mouthTop.x, cy: mouthTop.y,
        dx: 0, dy: -lipStrength * mouthWidth * 0.04,
        r: radius * 0.35
    });

    // Eye squint (slight narrowing for natural smile)
    const squintStrength = intensity * 0.015;
    const leftEyeBottom = lm.leftEye[4];
    const rightEyeBottom = lm.rightEye[4];
    warpPoints.push({
        cx: leftEyeBottom.x, cy: leftEyeBottom.y,
        dx: 0, dy: -squintStrength * mouthWidth * 0.08,
        r: radius * 0.35
    });
    warpPoints.push({
        cx: rightEyeBottom.x, cy: rightEyeBottom.y,
        dx: 0, dy: -squintStrength * mouthWidth * 0.08,
        r: radius * 0.35
    });

    // Apply inverse warp: for each destination pixel, find source
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let dx = 0, dy = 0;
            for (const wp of warpPoints) {
                const distSq = (x - wp.cx) ** 2 + (y - wp.cy) ** 2;
                const rSq = wp.r ** 2;
                if (distSq < rSq) {
                    // Smooth falloff using squared cosine
                    const d = Math.sqrt(distSq);
                    const weight = Math.cos(Math.PI * d / (2 * wp.r));
                    const w2 = weight * weight;
                    dx -= wp.dx * w2;
                    dy -= wp.dy * w2;
                }
            }

            const sx = x + dx;
            const sy = y + dy;

            // Bilinear interpolation
            const sx0 = Math.floor(sx);
            const sy0 = Math.floor(sy);
            const sx1 = Math.min(sx0 + 1, w - 1);
            const sy1 = Math.min(sy0 + 1, h - 1);
            const fx = sx - sx0;
            const fy = sy - sy0;

            if (sx0 >= 0 && sx0 < w && sy0 >= 0 && sy0 < h) {
                const i00 = (sy0 * w + sx0) * 4;
                const i10 = (sy0 * w + sx1) * 4;
                const i01 = (sy1 * w + sx0) * 4;
                const i11 = (sy1 * w + sx1) * 4;
                const idx = (y * w + x) * 4;

                for (let c = 0; c < 4; c++) {
                    dstData[idx + c] = Math.round(
                        src[i00 + c] * (1 - fx) * (1 - fy) +
                        src[i10 + c] * fx * (1 - fy) +
                        src[i01 + c] * (1 - fx) * fy +
                        src[i11 + c] * fx * fy
                    );
                }
            }
        }
    }

    return dst;
}

// --- GIF encoding ---

function encodeGif(frameCanvases, delay) {
    const encoder = new GifEncoder(frameCanvases[0].width, frameCanvases[0].height, {
        delay: delay,
        repeat: 0
    });

    frameCanvases.forEach(canvas => {
        encoder.addFrame(canvas, delay);
    });

    return Promise.resolve(encoder.render());
}

// --- Utilities ---

function getCenter(points) {
    const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
    const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
    return { x: cx, y: cy };
}

function dist(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function showStatus(msg, isError = false) {
    statusEl.hidden = false;
    statusEl.textContent = msg;
    statusEl.className = 'status' + (isError ? ' error' : '');
}

function hideStatus() {
    statusEl.hidden = true;
}

function showProgress(pct) {
    progressBarContainer.hidden = false;
    progressBar.style.width = pct + '%';
}

function hideProgress() {
    progressBarContainer.hidden = true;
    progressBar.style.width = '0%';
}
