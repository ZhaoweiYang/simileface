/**
 * Minimal GIF89a encoder for looping animations.
 * No external dependencies, no web workers needed.
 * Supports LZW compression and Netscape looping extension.
 */
class GifEncoder {
    constructor(width, height, options = {}) {
        this.width = width;
        this.height = height;
        this.delay = options.delay || 100; // ms
        this.repeat = options.repeat !== undefined ? options.repeat : 0; // 0 = loop forever
        this.frames = [];
    }

    addFrame(canvas, delay) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, this.width, this.height);
        this.frames.push({
            data: imageData.data,
            delay: delay !== undefined ? delay : this.delay
        });
    }

    render() {
        const frames = this.frames;
        const w = this.width;
        const h = this.height;

        // Quantize all frames to 256-color palettes
        const quantized = frames.map(f => this._quantize(f.data, w, h));

        const bytes = [];

        // GIF Header
        this._writeString(bytes, 'GIF89a');

        // Logical Screen Descriptor
        this._writeShort(bytes, w);
        this._writeShort(bytes, h);
        // Global color table flag=1, color resolution=7 (8 bits), sort=0, size=7 (256 colors)
        bytes.push(0xF7);
        bytes.push(0); // background color index
        bytes.push(0); // pixel aspect ratio

        // Global Color Table (use first frame's palette)
        const globalPalette = quantized[0].palette;
        for (let i = 0; i < 256; i++) {
            bytes.push(globalPalette[i * 3]);
            bytes.push(globalPalette[i * 3 + 1]);
            bytes.push(globalPalette[i * 3 + 2]);
        }

        // Netscape Application Extension (for looping)
        bytes.push(0x21); // Extension Introducer
        bytes.push(0xFF); // Application Extension
        bytes.push(11);   // Block size
        this._writeString(bytes, 'NETSCAPE2.0');
        bytes.push(3);    // Sub-block size
        bytes.push(1);    // Sub-block ID
        this._writeShort(bytes, this.repeat); // Loop count (0 = forever)
        bytes.push(0);    // Block terminator

        // Frames
        for (let i = 0; i < frames.length; i++) {
            const q = quantized[i];
            const delayCs = Math.round(frames[i].delay / 10); // convert ms to centiseconds

            // Graphic Control Extension
            bytes.push(0x21); // Extension Introducer
            bytes.push(0xF9); // Graphic Control Label
            bytes.push(4);    // Block size
            bytes.push(0);    // Disposal: none, no transparent
            this._writeShort(bytes, delayCs);
            bytes.push(0);    // Transparent color index (not used)
            bytes.push(0);    // Block terminator

            if (i === 0) {
                // First frame: use global color table
                bytes.push(0x2C); // Image Separator
                this._writeShort(bytes, 0); // left
                this._writeShort(bytes, 0); // top
                this._writeShort(bytes, w);
                this._writeShort(bytes, h);
                bytes.push(0); // No local color table
            } else {
                // Subsequent frames: local color table
                bytes.push(0x2C); // Image Separator
                this._writeShort(bytes, 0);
                this._writeShort(bytes, 0);
                this._writeShort(bytes, w);
                this._writeShort(bytes, h);
                // Local color table flag=1, interlace=0, sort=0, size=7 (256)
                bytes.push(0x87);
                for (let j = 0; j < 256; j++) {
                    bytes.push(q.palette[j * 3]);
                    bytes.push(q.palette[j * 3 + 1]);
                    bytes.push(q.palette[j * 3 + 2]);
                }
            }

            // LZW compressed image data
            this._writeLzw(bytes, q.indices, 8);
        }

        // GIF Trailer
        bytes.push(0x3B);

        return new Blob([new Uint8Array(bytes)], { type: 'image/gif' });
    }

    _quantize(rgba, w, h) {
        // Median-cut color quantization to 256 colors
        const pixelCount = w * h;
        const pixels = [];

        // Sample pixels (every pixel for small images, subsample for large)
        const step = pixelCount > 100000 ? 4 : 1;
        for (let i = 0; i < pixelCount; i += step) {
            const idx = i * 4;
            pixels.push([rgba[idx], rgba[idx + 1], rgba[idx + 2]]);
        }

        const palette = this._medianCut(pixels, 256);
        const paletteFlat = new Uint8Array(256 * 3);
        for (let i = 0; i < palette.length; i++) {
            paletteFlat[i * 3] = palette[i][0];
            paletteFlat[i * 3 + 1] = palette[i][1];
            paletteFlat[i * 3 + 2] = palette[i][2];
        }

        // Build k-d tree for fast nearest-neighbor lookup
        const indices = new Uint8Array(pixelCount);
        // Use simple cache for speed
        const cache = new Map();

        for (let i = 0; i < pixelCount; i++) {
            const idx = i * 4;
            const r = rgba[idx], g = rgba[idx + 1], b = rgba[idx + 2];
            // Quantize to 5 bits for cache key
            const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);

            if (cache.has(key)) {
                indices[i] = cache.get(key);
            } else {
                let bestIdx = 0;
                let bestDist = Infinity;
                for (let j = 0; j < palette.length; j++) {
                    const dr = r - palette[j][0];
                    const dg = g - palette[j][1];
                    const db = b - palette[j][2];
                    const d = dr * dr + dg * dg + db * db;
                    if (d < bestDist) {
                        bestDist = d;
                        bestIdx = j;
                    }
                }
                indices[i] = bestIdx;
                cache.set(key, bestIdx);
            }
        }

        return { palette: paletteFlat, indices };
    }

    _medianCut(pixels, maxColors) {
        if (pixels.length === 0) {
            const result = [];
            for (let i = 0; i < maxColors; i++) result.push([0, 0, 0]);
            return result;
        }

        const boxes = [{ pixels: pixels }];

        while (boxes.length < maxColors) {
            // Find box with largest range to split
            let bestBox = -1;
            let bestRange = -1;
            let bestChannel = 0;

            for (let i = 0; i < boxes.length; i++) {
                if (boxes[i].pixels.length < 2) continue;
                for (let c = 0; c < 3; c++) {
                    let min = 255, max = 0;
                    for (const p of boxes[i].pixels) {
                        if (p[c] < min) min = p[c];
                        if (p[c] > max) max = p[c];
                    }
                    const range = max - min;
                    if (range > bestRange) {
                        bestRange = range;
                        bestBox = i;
                        bestChannel = c;
                    }
                }
            }

            if (bestBox === -1) break;

            const box = boxes[bestBox];
            box.pixels.sort((a, b) => a[bestChannel] - b[bestChannel]);
            const mid = box.pixels.length >> 1;

            boxes[bestBox] = { pixels: box.pixels.slice(0, mid) };
            boxes.push({ pixels: box.pixels.slice(mid) });
        }

        // Compute average color per box
        const palette = boxes.map(box => {
            let r = 0, g = 0, b = 0;
            for (const p of box.pixels) {
                r += p[0]; g += p[1]; b += p[2];
            }
            const n = box.pixels.length || 1;
            return [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
        });

        // Pad to maxColors
        while (palette.length < maxColors) {
            palette.push([0, 0, 0]);
        }

        return palette;
    }

    _writeLzw(bytes, indices, minCodeSize) {
        bytes.push(minCodeSize);

        const clearCode = 1 << minCodeSize;
        const eoiCode = clearCode + 1;

        let codeSize = minCodeSize + 1;
        let nextCode = eoiCode + 1;
        const maxCode = 4096;

        let table = new Map();
        const output = [];

        // Bit packing
        let bitBuf = 0;
        let bitCount = 0;

        function emitCode(code) {
            bitBuf |= code << bitCount;
            bitCount += codeSize;
            while (bitCount >= 8) {
                output.push(bitBuf & 0xFF);
                bitBuf >>= 8;
                bitCount -= 8;
            }
        }

        function resetTable() {
            table = new Map();
            codeSize = minCodeSize + 1;
            nextCode = eoiCode + 1;
        }

        emitCode(clearCode);
        resetTable();

        let prefix = indices[0];

        for (let i = 1; i < indices.length; i++) {
            const k = indices[i];
            const key = (prefix << 12) | k; // Compact key for typical palette sizes

            if (table.has(key)) {
                prefix = table.get(key);
            } else {
                emitCode(prefix);

                if (nextCode < maxCode) {
                    table.set(key, nextCode++);
                    if (nextCode > (1 << codeSize) && codeSize < 12) {
                        codeSize++;
                    }
                } else {
                    emitCode(clearCode);
                    resetTable();
                }

                prefix = k;
            }
        }

        emitCode(prefix);
        emitCode(eoiCode);

        // Flush remaining bits
        if (bitCount > 0) {
            output.push(bitBuf & 0xFF);
        }

        // Write sub-blocks (max 255 bytes each)
        let pos = 0;
        while (pos < output.length) {
            const blockSize = Math.min(255, output.length - pos);
            bytes.push(blockSize);
            for (let i = 0; i < blockSize; i++) {
                bytes.push(output[pos++]);
            }
        }
        bytes.push(0); // Block terminator
    }

    _writeString(bytes, str) {
        for (let i = 0; i < str.length; i++) {
            bytes.push(str.charCodeAt(i));
        }
    }

    _writeShort(bytes, val) {
        bytes.push(val & 0xFF);
        bytes.push((val >> 8) & 0xFF);
    }
}
