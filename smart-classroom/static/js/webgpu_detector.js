/**
 * webgpu_detector.js — In-Browser Client Inference Engine
 * Runs YOLOv8 with WebGPU hardware acceleration and WASM SIMD fallback via ONNX Runtime Web.
 */

class WebGPUDetector {
  constructor(options = {}) {
    this.modelPath = options.modelPath || '/models/best.onnx';
    this.confThreshold = options.confThreshold || 0.40;
    this.iouThreshold = options.iouThreshold || 0.45;
    this.inputSize = 640;
    this.session = null;
    this.activeBackend = 'initializing';
    this.isModelLoaded = false;
    this.warmupComplete = false;

    // Offscreen canvas for preprocessing
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.inputSize;
    this.canvas.height = this.inputSize;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
  }

  /**
   * Detect WebGPU availability in the client browser.
   */
  async checkWebGPUSupport() {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) return true;
      } catch (e) {
        console.warn('[WebGPU] Hardware check failed:', e);
      }
    }
    return false;
  }

  /**
   * Initialize ONNX Runtime session with WebGPU and WASM fallback.
   */
  async loadModel(onProgress = null) {
    if (this.session) return this.session;

    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime Web library (ort) is not loaded.');
    }

    // Configure WASM SIMD & Multi-threading parameters
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);

    const hasWebGPU = await this.checkWebGPUSupport();
    const providers = hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'];

    if (onProgress) onProgress({ status: 'loading', provider: providers[0] });

    try {
      console.log(`[WebGPUDetector] Loading session with execution providers:`, providers);
      const sessionOptions = {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      };

      this.session = await ort.InferenceSession.create(this.modelPath, sessionOptions);
      this.activeBackend = hasWebGPU ? 'webgpu' : 'wasm';
      this.isModelLoaded = true;

      // Perform a dry-run warmup inference to compile WebGPU shaders
      await this.warmup();

      if (onProgress) onProgress({ status: 'ready', provider: this.activeBackend });
      console.log(`[WebGPUDetector] Ready on backend: ${this.activeBackend.toUpperCase()}`);
      return this.session;
    } catch (err) {
      console.warn(`[WebGPUDetector] Primary load failed, attempting WASM fallback:`, err);
      // Fallback directly to WASM
      try {
        this.session = await ort.InferenceSession.create(this.modelPath, {
          executionProviders: ['wasm'],
        });
        this.activeBackend = 'wasm';
        this.isModelLoaded = true;
        await this.warmup();
        if (onProgress) onProgress({ status: 'ready', provider: 'wasm' });
        return this.session;
      } catch (fallbackErr) {
        this.activeBackend = 'failed';
        throw fallbackErr;
      }
    }
  }

  /**
   * Warmup inference pass to allocate GPU buffers and avoid first-frame lag.
   */
  async warmup() {
    const dummy = new Float32Array(1 * 3 * this.inputSize * this.inputSize);
    const tensor = new ort.Tensor('float32', dummy, [1, 3, this.inputSize, this.inputSize]);
    try {
      await this.session.run({ images: tensor });
      this.warmupComplete = true;
    } catch (e) {
      console.warn('[WebGPUDetector] Warmup pass notice:', e);
    }
  }

  /**
   * Preprocess source image or video frame into normalized planar tensor [1, 3, 640, 640].
   */
  preprocess(source, origW, origH) {
    this.ctx.drawImage(source, 0, 0, this.inputSize, this.inputSize);
    const imgData = this.ctx.getImageData(0, 0, this.inputSize, this.inputSize).data;

    const numPixels = this.inputSize * this.inputSize;
    const floatData = new Float32Array(3 * numPixels);

    // Reorder from RGBA packed to planar RGB normalized [0, 1]
    for (let i = 0; i < numPixels; i++) {
      floatData[i] = imgData[i * 4] / 255.0; // Red channel
      floatData[numPixels + i] = imgData[i * 4 + 1] / 255.0; // Green channel
      floatData[2 * numPixels + i] = imgData[i * 4 + 2] / 255.0; // Blue channel
    }

    return new ort.Tensor('float32', floatData, [1, 3, this.inputSize, this.inputSize]);
  }

  /**
   * Execute inference on an HTML image, video, or canvas element.
   */
  async detect(source) {
    if (!this.session) {
      await this.loadModel();
    }

    const origW = source.videoWidth || source.naturalWidth || source.width;
    const origH = source.videoHeight || source.naturalHeight || source.height;

    if (!origW || !origH) {
      return { detections: [], inferenceTimeMs: 0, backend: this.activeBackend };
    }

    const t0 = performance.now();
    const inputTensor = this.preprocess(source, origW, origH);
    const prepTimeMs = performance.now() - t0;

    const t1 = performance.now();
    const feeds = { images: inputTensor };
    const results = await this.session.run(feeds);
    const inferenceTimeMs = Math.round(performance.now() - t1);

    const outputTensor = results.output0 || results[Object.keys(results)[0]];
    const rawData = outputTensor.data;

    const detections = this.postprocess(rawData, origW, origH);

    return {
      detections,
      inferenceTimeMs,
      prepTimeMs: Math.round(prepTimeMs),
      backend: this.activeBackend,
      imageWidth: origW,
      imageHeight: origH,
    };
  }

  /**
   * Parse YOLOv8 output tensor [1, 5, 8400] and perform Non-Maximum Suppression (NMS).
   */
  postprocess(data, origW, origH) {
    const numAnchors = 8400;
    const scaleX = origW / this.inputSize;
    const scaleY = origH / this.inputSize;

    const boxes = [];

    // YOLOv8 layout: 0: cx, 1: cy, 2: w, 3: h, 4: score
    for (let i = 0; i < numAnchors; i++) {
      const score = data[4 * numAnchors + i];
      if (score >= this.confThreshold) {
        const cx = data[0 * numAnchors + i];
        const cy = data[1 * numAnchors + i];
        const w = data[2 * numAnchors + i];
        const h = data[3 * numAnchors + i];

        const x1 = Math.max(0, (cx - w / 2) * scaleX);
        const y1 = Math.max(0, (cy - h / 2) * scaleY);
        const x2 = Math.min(origW, (cx + w / 2) * scaleX);
        const y2 = Math.min(origH, (cy + h / 2) * scaleY);

        boxes.push({
          bbox: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
          confidence: parseFloat(score.toFixed(3)),
          label: 'person',
        });
      }
    }

    return this.nonMaxSuppression(boxes, this.iouThreshold);
  }

  /**
   * Fast Non-Maximum Suppression.
   */
  nonMaxSuppression(boxes, iouThreshold) {
    boxes.sort((a, b) => b.confidence - a.confidence);
    const selected = [];
    const active = new Array(boxes.length).fill(true);

    for (let i = 0; i < boxes.length; i++) {
      if (!active[i]) continue;
      selected.push(boxes[i]);

      const [x1A, y1A, x2A, y2A] = boxes[i].bbox;
      const areaA = (x2A - x1A) * (y2A - y1A);

      for (let j = i + 1; j < boxes.length; j++) {
        if (!active[j]) continue;

        const [x1B, y1B, x2B, y2B] = boxes[j].bbox;
        const areaB = (x2B - x1B) * (y2B - y1B);

        const interX1 = Math.max(x1A, x1B);
        const interY1 = Math.max(y1A, y1B);
        const interX2 = Math.min(x2A, x2B);
        const interY2 = Math.min(y2A, y2B);

        const interW = Math.max(0, interX2 - interX1);
        const interH = Math.max(0, interY2 - interY1);
        const interArea = interW * interH;

        const unionArea = areaA + areaB - interArea;
        const iou = unionArea > 0 ? interArea / unionArea : 0;

        if (iou > iouThreshold) {
          active[j] = false;
        }
      }
    }

    return selected;
  }
}

// Global export for vanilla script inclusion or ES imports
if (typeof window !== 'undefined') {
  window.WebGPUDetector = WebGPUDetector;
}
