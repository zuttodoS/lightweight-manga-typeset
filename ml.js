/**
 * ml.js — Browser inference mirroring manage-extract-and-clean exactly.
 *
 * Detection:  manga_translator/detection/default.py
 * OCR:        manga_translator/ocr/model_48px.py
 * Inpainting: manga_translator/inpainting/inpainting_lama_mpe.py + inpainting_aot.py
 *
 * ONNX models have fixed input shapes. We pad inputs to match and crop outputs.
 */

if (typeof ort !== 'undefined') {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.logLevel = 'error';
}

const ML = (() => {
  let detSession = null, ocrEncSession = null, ocrDecSession = null, inpaintSession = null;
  let dictionary = null;

  const M = 'models/';

  // ── ONNX fixed input sizes ───────────────────────────────────────────────
  // Detection: traced at 2048x2048 (matches config.py detect_size=2048)
  // OCR encoder: traced at [1,3,48,256]
  // OCR decoder: traced at [1,64,320] memory + [1,128] tokens
  // Inpainting: traced at [1,3,512,512] + [1,1,512,512]
  const DET_S = 2048;
  const OCR_H = 48, OCR_W = 256;
  const DEC_TOKENS = 128;
  const INP_S = 512;

  // ── Config (mirrors config.py defaults) ──────────────────────────────────
  // detection/default.py line 75: SegDetectorRepresenter(text_threshold, box_threshold, unclip_ratio)
  // config.py: text_threshold=0.5, box_threshold=0.7, unclip_ratio=2.3
  const TEXT_THRESH = 0.5;
  const BOX_THRESH = 0.7;
  const UNCLIP_RATIO = 2.3;
  const MIN_BOX_SIZE = 3; // dbnet_utils.py line 9

  const EP = ['webgpu', 'wasm'];

  // ── Model loading ────────────────────────────────────────────────────────

  async function loadDet(cb) {
    if (detSession) return;
    if (cb) cb('Loading detection model...');
    detSession = await ort.InferenceSession.create(M + 'manga_det.onnx', { executionProviders: EP });
  }

  async function loadOCR(cb) {
    if (ocrEncSession && ocrDecSession && dictionary) return;
    if (cb) cb('Loading OCR model...');
    if (!dictionary) {
      // manga_translator/ocr/model_48px.py line 48
      const r = await fetch(M + 'alphabet-all-v7.txt');
      dictionary = (await r.text()).split('\n');
      // Python: [s[:-1] for s in fp.readlines()] — strips trailing newline per line
      // Our split('\n') does the same. Keep empty string at end if file ends with newline.
      if (dictionary[dictionary.length - 1] === '') dictionary.pop();
    }
    if (!ocrEncSession) ocrEncSession = await ort.InferenceSession.create(M + 'manga_ocr_encoder.onnx', { executionProviders: EP });
    if (!ocrDecSession) ocrDecSession = await ort.InferenceSession.create(M + 'manga_ocr_decoder.onnx', { executionProviders: EP });
  }

  async function loadInp(cb) {
    if (inpaintSession) return;
    if (cb) cb('Loading inpainting model...');
    inpaintSession = await ort.InferenceSession.create(M + 'manga_inpaint.onnx', { executionProviders: EP });
  }

  // ── Utility ──────────────────────────────────────────────────────────────

  function imgToCanvas(src) {
    return new Promise(res => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        c.getContext('2d').drawImage(img, 0, 0);
        res({ canvas: c, w: c.width, h: c.height });
      };
      img.src = src;
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  // DETECTION
  // Mirrors: detection/default.py _infer() + det_batch_forward_default()
  //
  // Python flow (for normal aspect ratio images):
  //   1. bilateralFilter(image, 17, 80, 80)                         [line 64]
  //   2. resize_aspect_ratio(img, detect_size=2048, INTER_LINEAR)   [line 64]
  //      → resize longest side to 2048, pad to multiple of 256      [imgproc.py]
  //   3. normalize: (x / 127.5) - 1.0, NHWC→NCHW                   [line 19]
  //   4. model(batch) → db, mask                                    [line 22]
  //   5. db.sigmoid()                                               [line 23]
  //   6. SegDetectorRepresenter on db[:, 0, :, :]                   [line 75-77]
  //      → binarize, findContours, get_mini_boxes, unclip, scale    [dbnet_utils.py]
  //   7. adjustResultCoordinates(polys, 1/ratio, 1/ratio)           [line 85]
  //   8. mask: resize to 2x, clip to image bounds, * 255            [line 90-95]
  // ────────────────────────────────────────────────────────────────────────

  async function detect(imageSrc, cb) {
    await loadDet(cb);
    if (cb) cb('Detecting text...');

    const { canvas, w: origW, h: origH } = await imgToCanvas(imageSrc);

    // Step 1: bilateralFilter — skip in browser (expensive, marginal benefit)
    // Step 2: resize_aspect_ratio (imgproc.py line 37-70)
    //   ratio = detect_size / max(h, w)
    //   target_h = round(h * ratio), target_w = round(w * ratio)
    //   pad to multiple of 256 (bottom/right)
    const ratio = DET_S / Math.max(origW, origH);
    const rw = Math.round(origW * ratio);
    const rh = Math.round(origH * ratio);
    // Pad to DET_S x DET_S (our ONNX fixed size >= any padded size since detect_size=2048)
    // Python pads to 256-multiples; our 2048 is always a valid multiple.
    // Image placed at top-left, rest is black (zero-padded).

    const detCanvas = document.createElement('canvas');
    detCanvas.width = DET_S; detCanvas.height = DET_S;
    const dctx = detCanvas.getContext('2d');
    // Black fill = zero after normalization (-1.0), matching Python's np.zeros padding
    dctx.fillStyle = '#000';
    dctx.fillRect(0, 0, DET_S, DET_S);
    dctx.drawImage(canvas, 0, 0, origW, origH, 0, 0, rw, rh);

    // Step 3: normalize (detection/default.py line 19)
    //   batch = batch.astype(float32) / 127.5 - 1.0
    const px = dctx.getImageData(0, 0, DET_S, DET_S).data;
    const sz = DET_S * DET_S;
    const f = new Float32Array(3 * sz);
    for (let i = 0; i < sz; i++) {
      f[i]        = px[i * 4] / 127.5 - 1.0;
      f[sz + i]   = px[i * 4 + 1] / 127.5 - 1.0;
      f[2*sz + i] = px[i * 4 + 2] / 127.5 - 1.0;
    }

    // Step 4-5: model forward + sigmoid
    const results = await detSession.run({
      [detSession.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, DET_S, DET_S])
    });

    // db output: [1, 2, 2048, 2048] — Python applies sigmoid then takes [:, 0, :, :]
    const dbRaw = results[detSession.outputNames[0]].data;
    const probMap = new Float32Array(sz);
    for (let i = 0; i < sz; i++) {
      probMap[i] = 1 / (1 + Math.exp(-dbRaw[i])); // sigmoid
    }

    // mask output: [1, 1, 1024, 1024]
    const maskData = results[detSession.outputNames[1]].data;
    const maskS = DET_S / 2; // 1024

    // Step 6: SegDetectorRepresenter (dbnet_utils.py)
    //   In Python: db map size == padded input size, dest == padded size → scale is 1:1
    //   In JS: db map is DET_S x DET_S (fixed ONNX), dest should also be DET_S x DET_S
    //   so boxes come out in 2048-canvas coords (same as padded coords in Python)
    const boxes = dbPostprocess(probMap, DET_S, DET_S, DET_S, DET_S);

    // Step 7: adjustResultCoordinates (craft_utils.py line 85)
    //   polys *= (ratio_w, ratio_h) where ratio_w = ratio_h = 1/ratio
    //   Converts from padded/canvas coords → original image coords
    const invRatio = 1 / ratio;
    const finalBoxes = [];
    for (const [x1, y1, x2, y2] of boxes) {
      const fx1 = Math.max(0, Math.round(x1 * invRatio));
      const fy1 = Math.max(0, Math.round(y1 * invRatio));
      const fx2 = Math.min(origW, Math.round(x2 * invRatio));
      const fy2 = Math.min(origH, Math.round(y2 * invRatio));
      // Python: filter(lambda q: q.area > 16)  [line 89]
      if ((fx2 - fx1) * (fy2 - fy1) > 16) {
        finalBoxes.push([fx1, fy1, fx2, fy2]);
      }
    }

    // Step 8: mask processing (line 90-95)
    //   mask_resized = cv2.resize(mask, (w*2, h*2))
    //   if pad_h > 0: mask_resized = mask_resized[:-pad_h, :]
    //   raw_mask = clip(mask_resized * 255, 0, 255)
    // Our mask is [1,1,1024,1024] covering the 2048x2048 padded canvas.
    // The valid region is [0:rh/2, 0:rw/2] in the 1024x1024 mask.
    // After 2x upscale: [0:rh, 0:rw] in 2048x2048 space.
    // Then scale to original image coords.
    const rawMaskCanvas = document.createElement('canvas');
    rawMaskCanvas.width = maskS; rawMaskCanvas.height = maskS;
    const mctx = rawMaskCanvas.getContext('2d');
    const mimg = mctx.createImageData(maskS, maskS);
    for (let i = 0; i < maskS * maskS; i++) {
      const v = Math.max(0, Math.min(255, Math.round(maskData[i] * 255)));
      mimg.data[i * 4] = v; mimg.data[i * 4 + 1] = v;
      mimg.data[i * 4 + 2] = v; mimg.data[i * 4 + 3] = 255;
    }
    mctx.putImageData(mimg, 0, 0);

    // Crop to valid region (rw/2 x rh/2 in mask space), scale to original image size
    const rawMask = document.createElement('canvas');
    rawMask.width = origW; rawMask.height = origH;
    rawMask.getContext('2d').drawImage(
      rawMaskCanvas,
      0, 0, Math.round(rw / 2), Math.round(rh / 2), // source: valid region only
      0, 0, origW, origH // dest: full original image
    );

    return { boxes: finalBoxes, w: origW, h: origH, rawMask };
  }

  // Mirrors: dbnet_utils.py SegDetectorRepresenter.boxes_from_bitmap()
  // Simplified: uses connected components instead of cv2.findContours + minAreaRect
  // (cv2 not available in browser, but the result is equivalent axis-aligned boxes)
  function dbPostprocess(probMap, mapW, mapH, destW, destH) {
    // binarize: pred > text_threshold  [line 46]
    const bin = new Uint8Array(mapW * mapH);
    for (let i = 0; i < probMap.length; i++) bin[i] = probMap[i] > TEXT_THRESH ? 1 : 0;

    // findContours equivalent: connected components
    const labels = new Int32Array(mapW * mapH);
    const comps = [];
    let nextL = 1;
    for (let y = 0; y < mapH; y++) {
      for (let x = 0; x < mapW; x++) {
        const idx = y * mapW + x;
        if (!bin[idx] || labels[idx]) continue;
        let x0 = x, y0 = y, x1 = x, y1 = y, area = 0;
        const stk = [idx];
        labels[idx] = nextL;
        while (stk.length) {
          const ci = stk.pop();
          const cx = ci % mapW, cy = (ci - cx) / mapW;
          area++;
          if (cx < x0) x0 = cx; if (cy < y0) y0 = cy;
          if (cx > x1) x1 = cx; if (cy > y1) y1 = cy;
          for (const ni of [ci - 1, ci + 1, ci - mapW, ci + mapW]) {
            if (ni >= 0 && ni < mapW * mapH && bin[ni] && !labels[ni]) {
              labels[ni] = nextL; stk.push(ni);
            }
          }
        }
        // get_mini_boxes: sside < min_size check  [line 120-123]
        if (Math.min(x1 - x0, y1 - y0) >= MIN_BOX_SIZE) {
          comps.push({ x0, y0, x1, y1, area });
        }
        nextL++;
      }
    }

    const boxes = [];
    for (const c of comps) {
      // box_score_fast: mean prob in bbox region  [line 175-187]
      let sum = 0, cnt = 0;
      for (let y = c.y0; y <= c.y1; y++)
        for (let x = c.x0; x <= c.x1; x++) { sum += probMap[y * mapW + x]; cnt++; }
      if (!cnt || sum / cnt < BOX_THRESH) continue;

      // unclip: distance = area * unclip_ratio / perimeter  [line 146-152]
      const bw = c.x1 - c.x0, bh = c.y1 - c.y0;
      const d = (bw * bh) * UNCLIP_RATIO / (2 * (bw + bh) + 1e-6);

      // Scale to dest coords (resized image, before inverse ratio)  [line 137-138]
      const sx = destW / mapW, sy = destH / mapH;
      const bx1 = Math.max(0, Math.round((c.x0 - d) * sx));
      const by1 = Math.max(0, Math.round((c.y0 - d) * sy));
      const bx2 = Math.min(destW, Math.round((c.x1 + d) * sx));
      const by2 = Math.min(destH, Math.round((c.y1 + d) * sy));

      if (bx2 > bx1 && by2 > by1) boxes.push([bx1, by1, bx2, by2]);
    }
    return boxes;
  }

  // ────────────────────────────────────────────────────────────────────────
  // OCR
  // Mirrors: ocr/model_48px.py _infer() + infer_beam_batch_tensor()
  //
  // Python flow:
  //   1. For each textline: get_transformed_region(image, direction, 48)  [line 73]
  //      → perspective transform to h=48, proportional width
  //      (simplified here to axis-aligned crop + resize)
  //   2. Sort by width, batch with max_chunk_size=16                     [line 79-86]
  //   3. Pad width to multiple of 4: max_width = 4*(max(widths)+7)//4   [line 86]
  //   4. Create region array (N, 48, max_width, 3) uint8, zero-padded   [line 87-91]
  //   5. Normalize: (tensor - 127.5) / 127.5                            [line 115]
  //   6. NHWC → NCHW                                                    [line 116]
  //   7. Beam search decode (beams_k=5, max_seq_length=255)             [line 120]
  //      (simplified here to greedy decode)
  //   8. Dictionary mapping with <S>, </S>, <SP> tokens                 [line 133-140]
  // ────────────────────────────────────────────────────────────────────────

  async function recognize(imageSrc, boxes, cb) {
    await loadOCR(cb);
    const { canvas } = await imgToCanvas(imageSrc);
    const texts = [];

    // Token indices from dictionary  [line 48-49, 133-139]
    const startIdx = dictionary.indexOf('<S>');   // should be 1
    const endIdx = dictionary.indexOf('</S>');    // should be 2

    for (let bi = 0; bi < boxes.length; bi++) {
      if (cb) cb(`OCR ${bi + 1}/${boxes.length}...`);
      const [bx1, by1, bx2, by2] = boxes[bi];
      const bw = Math.max(1, bx2 - bx1), bh = Math.max(1, by2 - by1);

      // Step 1: crop region
      // Mirrors get_transformed_region: if vertical text (h > w), rotate 90° CCW
      // so text reads horizontally for the OCR model  [generic.py line 445-481]
      const isVertical = bh > bw * 1.5;
      const cropC = document.createElement('canvas');
      if (isVertical) {
        // Rotate 90° counter-clockwise: (w,h) → (h,w)
        cropC.width = bh; cropC.height = bw;
        const cctx = cropC.getContext('2d');
        cctx.translate(0, bw);
        cctx.rotate(-Math.PI / 2);
        cctx.drawImage(canvas, bx1, by1, bw, bh, 0, 0, bw, bh);
      } else {
        cropC.width = bw; cropC.height = bh;
        cropC.getContext('2d').drawImage(canvas, bx1, by1, bw, bh, 0, 0, bw, bh);
      }

      // Now cropC has text in horizontal orientation
      const cropW = cropC.width, cropH = cropC.height;

      // Step 3: resize to h=48, width proportional, multiple of 4  [line 86]
      let tw = Math.max(4, Math.round(cropW * OCR_H / cropH));
      tw = 4 * Math.floor((tw + 3) / 4);

      // Clamp to ONNX fixed width (256): pad shorter, shrink longer
      // Python batches with variable width; our ONNX is fixed at 256
      const pw = OCR_W; // always 256
      tw = Math.min(tw, pw); // don't exceed ONNX width

      const rc = document.createElement('canvas');
      rc.width = pw; rc.height = OCR_H;
      const rctx = rc.getContext('2d');
      rctx.fillStyle = '#000'; // zero padding
      rctx.fillRect(0, 0, pw, OCR_H);
      rctx.drawImage(cropC, 0, 0, tw, OCR_H);

      // Step 5-6: normalize and reshape  [line 115-116]
      // (tensor - 127.5) / 127.5 = tensor / 127.5 - 1.0
      const px = rctx.getImageData(0, 0, pw, OCR_H).data;
      const isz = pw * OCR_H;
      const imgF = new Float32Array(3 * isz);
      for (let i = 0; i < isz; i++) {
        imgF[i]         = px[i * 4] / 127.5 - 1.0;
        imgF[isz + i]   = px[i * 4 + 1] / 127.5 - 1.0;
        imgF[2*isz + i] = px[i * 4 + 2] / 127.5 - 1.0;
      }

      try {
        // Encoder forward
        const encOut = await ocrEncSession.run({
          [ocrEncSession.inputNames[0]]: new ort.Tensor('float32', imgF, [1, 3, OCR_H, pw])
        });
        const memory = encOut[ocrEncSession.outputNames[0]];

        // Step 7: autoregressive decode (greedy with repetition detection)
        // Python uses beam search k=5 which naturally avoids repetition.
        // For greedy, we add: softmax confidence check + repetition early stop.
        let tokenIds = [startIdx >= 0 ? startIdx : 1];
        let text = '';
        let repeatCount = 0;
        let lastTok = -1;

        for (let step = 0; step < DEC_TOKENS - 1; step++) {
          const padded = new BigInt64Array(DEC_TOKENS);
          for (let t = 0; t < tokenIds.length && t < DEC_TOKENS; t++) {
            padded[t] = BigInt(tokenIds[t]);
          }

          const decOut = await ocrDecSession.run({
            [ocrDecSession.inputNames[0]]: memory,
            [ocrDecSession.inputNames[1]]: new ort.Tensor('int64', padded, [1, DEC_TOKENS]),
          });
          const allLogits = decOut[ocrDecSession.outputNames[0]].data;
          const vocabSize = 46272;
          const lastPos = tokenIds.length - 1;
          const offset = lastPos * vocabSize;

          // Softmax to get probability, then argmax
          let maxLogit = -Infinity;
          for (let c = 0; c < vocabSize; c++) {
            if (allLogits[offset + c] > maxLogit) maxLogit = allLogits[offset + c];
          }
          let sumExp = 0;
          for (let c = 0; c < vocabSize; c++) {
            sumExp += Math.exp(allLogits[offset + c] - maxLogit);
          }

          let best = -Infinity, bestI = 0;
          for (let c = 0; c < vocabSize; c++) {
            if (allLogits[offset + c] > best) { best = allLogits[offset + c]; bestI = c; }
          }
          const bestProb = Math.exp(best - maxLogit) / sumExp;

          // Check end token
          if (bestI === (endIdx >= 0 ? endIdx : 2)) break;

          // Low confidence → likely hallucinating, stop
          if (bestProb < 0.3 && step > 3) break;

          // Repetition detection: if same token repeats 3+ times, stop
          if (bestI === lastTok) {
            repeatCount++;
            if (repeatCount >= 3) break;
          } else {
            repeatCount = 0;
          }
          lastTok = bestI;

          tokenIds.push(bestI);

          if (bestI < dictionary.length) {
            const ch = dictionary[bestI];
            if (ch === '<S>' || ch === '</S>') continue;
            if (ch === '<SP>') { text += ' '; continue; }
            text += ch;
          }
        }

        console.log(`[OCR] box ${bi}: "${text.trim()}" (${tokenIds.length-1} tokens, crop=${bw}x${bh}${isVertical?' [V→rotated]':''} → ${tw}x${OCR_H})`);
        texts.push(text.trim());
      } catch (e) {
        console.warn(`OCR box ${bi} error:`, e);
        texts.push('');
      }
    }
    return texts;
  }

  // ────────────────────────────────────────────────────────────────────────
  // INPAINTING
  // Mirrors: inpainting/inpainting_lama_mpe.py _infer() with AOTGenerator
  //
  // Python flow:
  //   1. mask_original: threshold at 127, binary 0/1                   [line 58-61]
  //   2. Resize if max(h,w) > inpainting_size                         [line 64-66]
  //   3. Pad to multiple of 8                                          [line 67-79]
  //   4. img_torch = img / 127.5 - 1.0                                [line 84]
  //   5. mask_torch = mask / 255.0, threshold 0.5 → binary            [line 85-87]
  //   6. img_torch *= (1 - mask_torch)                                 [line 92]
  //   7. model(img_torch, mask_torch) → output                        [line 95]
  //   8. output: (output + 1.0) * 127.5                               [line 114]
  //   9. Resize back to original                                       [line 115-116]
  //  10. Blend: inpainted * mask + original * (1 - mask)               [line 117]
  //
  // AOTGenerator.forward(img, mask):                                   [inpainting_aot.py line 266]
  //   x = cat([mask, img], dim=1)  → [1, 4, H, W]
  //   (this concatenation happens inside the model, ONNX captures it)
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Inpaint: processes regions and returns an array of inpaint patches.
   * Each patch = { x, y, w, h, imageData } in natural image coordinates.
   * The caller composites these patches onto the stroke canvas (not replacing the image).
   *
   * @param {HTMLCanvasElement} compositeCanvas — image + existing strokes composited at natural res
   * @param {HTMLCanvasElement} maskCanvas — white=inpaint, black=keep, at natural res
   * @param {function} cb — progress callback
   * @param {Array} boxes — [[x1,y1,x2,y2], ...] regions to inpaint
   * @returns {Array<{x,y,w,h,dataUrl}>} inpaint patches
   */
  async function inpaint(compositeCanvas, maskCanvas, cb, boxes) {
    await loadInp(cb);
    if (cb) cb('Inpainting...');

    const w = compositeCanvas.width, h = compositeCanvas.height;

    let mask = maskCanvas;
    if (mask.width !== w || mask.height !== h) {
      const t = document.createElement('canvas');
      t.width = w; t.height = h;
      t.getContext('2d').drawImage(mask, 0, 0, w, h);
      mask = t;
    }

    let regions = boxes && boxes.length
      ? boxes.map(b => ({ x: b[0], y: b[1], w: b[2] - b[0], h: b[3] - b[1] }))
      : [{ x: 0, y: 0, w, h }];
    regions = mergeRegions(regions, 30);

    const patches = [];

    for (let ri = 0; ri < regions.length; ri++) {
      if (cb) cb(`Inpainting ${ri + 1}/${regions.length}...`);
      const r = regions[ri];

      // Context padding
      const pad = Math.max(Math.round(Math.max(r.w, r.h) * 0.5), 30);
      let cx1 = Math.max(0, r.x - pad), cy1 = Math.max(0, r.y - pad);
      let cx2 = Math.min(w, r.x + r.w + pad), cy2 = Math.min(h, r.y + r.h + pad);
      let cw = cx2 - cx1, ch = cy2 - cy1;

      // If crop is smaller than INP_S, don't upscale — center the region in INP_S crop
      if (cw < INP_S || ch < INP_S) {
        const needW = Math.max(cw, INP_S), needH = Math.max(ch, INP_S);
        const centerX = cx1 + cw / 2, centerY = cy1 + ch / 2;
        cx1 = Math.max(0, Math.round(centerX - needW / 2));
        cy1 = Math.max(0, Math.round(centerY - needH / 2));
        cx2 = Math.min(w, cx1 + needW);
        cy2 = Math.min(h, cy1 + needH);
        // Adjust if hit boundary
        if (cx2 - cx1 < needW) cx1 = Math.max(0, cx2 - needW);
        if (cy2 - cy1 < needH) cy1 = Math.max(0, cy2 - needH);
        cw = cx2 - cx1; ch = cy2 - cy1;
      }

      // Resize composite crop and mask to INP_S x INP_S
      const ic = document.createElement('canvas'); ic.width = INP_S; ic.height = INP_S;
      ic.getContext('2d').drawImage(compositeCanvas, cx1, cy1, cw, ch, 0, 0, INP_S, INP_S);
      const mc = document.createElement('canvas'); mc.width = INP_S; mc.height = INP_S;
      mc.getContext('2d').drawImage(mask, cx1, cy1, cw, ch, 0, 0, INP_S, INP_S);

      const ipx = ic.getContext('2d').getImageData(0, 0, INP_S, INP_S).data;
      const mpx = mc.getContext('2d').getImageData(0, 0, INP_S, INP_S).data;
      const n = INP_S * INP_S;
      const imgF = new Float32Array(3 * n), mskF = new Float32Array(n);

      for (let i = 0; i < n; i++) {
        const m = (mpx[i * 4] / 255.0) >= 0.5 ? 1.0 : 0.0;
        mskF[i] = m;
        imgF[i]       = (ipx[i * 4] / 127.5 - 1.0) * (1 - m);
        imgF[n + i]   = (ipx[i * 4 + 1] / 127.5 - 1.0) * (1 - m);
        imgF[2*n + i] = (ipx[i * 4 + 2] / 127.5 - 1.0) * (1 - m);
      }

      const out = await inpaintSession.run({
        [inpaintSession.inputNames[0]]: new ort.Tensor('float32', imgF, [1, 3, INP_S, INP_S]),
        [inpaintSession.inputNames[1]]: new ort.Tensor('float32', mskF, [1, 1, INP_S, INP_S]),
      });
      const od = out[inpaintSession.outputNames[0]].data;

      // Denormalize: (x + 1) * 127.5
      const inpC = document.createElement('canvas'); inpC.width = INP_S; inpC.height = INP_S;
      const inpImg = inpC.getContext('2d').createImageData(INP_S, INP_S);
      for (let i = 0; i < n; i++) {
        inpImg.data[i * 4]     = Math.max(0, Math.min(255, Math.round((od[i] + 1) * 127.5)));
        inpImg.data[i * 4 + 1] = Math.max(0, Math.min(255, Math.round((od[n + i] + 1) * 127.5)));
        inpImg.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round((od[2*n + i] + 1) * 127.5)));
        inpImg.data[i * 4 + 3] = 255;
      }
      inpC.getContext('2d').putImageData(inpImg, 0, 0);

      // Scale back to crop size, blend only masked pixels, return as patch
      const patchC = document.createElement('canvas'); patchC.width = cw; patchC.height = ch;
      const pctx = patchC.getContext('2d');
      // Start with transparent
      pctx.clearRect(0, 0, cw, ch);
      // Draw scaled inpainted result
      const scaledC = document.createElement('canvas'); scaledC.width = cw; scaledC.height = ch;
      scaledC.getContext('2d').drawImage(inpC, 0, 0, INP_S, INP_S, 0, 0, cw, ch);
      const inpPx = scaledC.getContext('2d').getImageData(0, 0, cw, ch).data;
      // Read mask at original resolution
      const maskCropC = document.createElement('canvas'); maskCropC.width = cw; maskCropC.height = ch;
      maskCropC.getContext('2d').drawImage(mask, cx1, cy1, cw, ch, 0, 0, cw, ch);
      const mPx = maskCropC.getContext('2d').getImageData(0, 0, cw, ch).data;
      // Only keep masked pixels in the patch (rest transparent)
      const patchData = pctx.createImageData(cw, ch);
      for (let i = 0; i < cw * ch; i++) {
        if (mPx[i * 4] > 127) {
          patchData.data[i * 4]     = inpPx[i * 4];
          patchData.data[i * 4 + 1] = inpPx[i * 4 + 1];
          patchData.data[i * 4 + 2] = inpPx[i * 4 + 2];
          patchData.data[i * 4 + 3] = 255;
        }
      }
      pctx.putImageData(patchData, 0, 0);

      patches.push({ x: cx1, y: cy1, w: cw, h: ch, dataUrl: patchC.toDataURL('image/png') });
    }

    return patches;
  }

  function mergeRegions(regs, gap) {
    if (regs.length <= 1) return regs;
    const m = [], u = new Set();
    for (let i = 0; i < regs.length; i++) {
      if (u.has(i)) continue;
      let r = { ...regs[i] }; u.add(i);
      let ch = true;
      while (ch) {
        ch = false;
        for (let j = 0; j < regs.length; j++) {
          if (u.has(j)) continue;
          const s = regs[j];
          if (r.x - gap <= s.x + s.w && r.x + r.w + gap >= s.x &&
              r.y - gap <= s.y + s.h && r.y + r.h + gap >= s.y) {
            const nx = Math.min(r.x, s.x), ny = Math.min(r.y, s.y);
            r = { x: nx, y: ny, w: Math.max(r.x + r.w, s.x + s.w) - nx, h: Math.max(r.y + r.h, s.y + s.h) - ny };
            u.add(j); ch = true;
          }
        }
      }
      m.push(r);
    }
    return m;
  }

  // ────────────────────────────────────────────────────────────────────────
  // EXTRACT PIPELINE
  // Mirrors: backend.py _process_single_image()
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Merge detected textlines into speech bubble regions.
   * Mirrors: textline_merge/__init__.py merge_bboxes_text_region()
   *
   * Uses quadrilateral_can_merge_region logic (simplified):
   *   - Distance between boxes < discard_connection_gap(2) * min_font_size
   *   - Font size ratio < font_size_ratio_tol(2)
   *   - Similar aspect ratio (don't merge horizontal with vertical)
   */
  function mergeTextlines(boxes, texts) {
    if (!boxes.length) return [];

    // Compute font_size (min dimension) and aspect_ratio for each box
    const info = boxes.map(([x1, y1, x2, y2]) => {
      const w = x2 - x1, h = y2 - y1;
      return { fs: Math.min(w, h), ar: w / (h || 1) };
    });

    // Build graph: can_merge between pairs  [generic.py line 653-698]
    const canMerge = (i, j) => {
      const a = boxes[i], b = boxes[j];
      const ai = info[i], bi = info[j];
      const charSize = Math.min(ai.fs, bi.fs);
      if (charSize < 1) return false;

      // Distance between boxes (gap between edges, 0 if overlapping)
      const dx = Math.max(0, Math.max(a[0], b[0]) - Math.min(a[2], b[2]));
      const dy = Math.max(0, Math.max(a[1], b[1]) - Math.min(a[3], b[3]));
      const dist = Math.sqrt(dx * dx + dy * dy);

      // discard_connection_gap * char_size  [line 663]
      if (dist > 2 * charSize) return false;
      // font_size_ratio_tol  [line 665]
      if (Math.max(ai.fs, bi.fs) / charSize > 2) return false;
      // aspect_ratio_tol: don't merge h with v  [line 667-670]
      if (ai.ar > 1.3 && bi.ar < 1 / 1.3) return false;
      if (bi.ar > 1.3 && ai.ar < 1 / 1.3) return false;
      // char_gap_tolerance  [line 674]
      if (dist < charSize * 1) return true;
      return false;
    };

    // Connected components via union-find
    const parent = boxes.map((_, i) => i);
    const find = (x) => { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; };
    const union = (a, b) => { parent[find(a)] = find(b); };

    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        if (canMerge(i, j)) union(i, j);
      }
    }

    // Group by component
    const groups = {};
    for (let i = 0; i < boxes.length; i++) {
      const root = find(i);
      if (!groups[root]) groups[root] = [];
      groups[root].push(i);
    }

    // Build merged regions with combined text
    const regions = [];
    for (const indices of Object.values(groups)) {
      let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
      // Sort by position: vertical text by x desc, horizontal by y asc  [line 175-178]
      const avgAR = indices.reduce((s, i) => s + info[i].ar, 0) / indices.length;
      const isVertical = avgAR < 1;
      if (isVertical) {
        indices.sort((a, b) => -(boxes[a][0] + boxes[a][2]) / 2 + (boxes[b][0] + boxes[b][2]) / 2);
      } else {
        indices.sort((a, b) => (boxes[a][1] + boxes[a][3]) / 2 - (boxes[b][1] + boxes[b][3]) / 2);
      }

      const regionTexts = [];
      for (const idx of indices) {
        x1 = Math.min(x1, boxes[idx][0]);
        y1 = Math.min(y1, boxes[idx][1]);
        x2 = Math.max(x2, boxes[idx][2]);
        y2 = Math.max(y2, boxes[idx][3]);
        if (texts[idx]) regionTexts.push(texts[idx]);
      }
      regions.push([x1, y1, x2, y2, regionTexts.join(' ')]);
    }
    return regions;
  }

  async function extract(imageSrc, cb) {
    // Step 1: detection  [backend.py line 166-173]
    const { boxes, w, h, rawMask } = await detect(imageSrc, cb);
    if (!boxes.length) return { bboxes: [], inpaintedSrc: imageSrc };

    // Step 2: OCR each textline  [backend.py line 182-185]
    const texts = await recognize(imageSrc, boxes, cb);
    // Filter textlines with no text  [backend.py line 185]
    const validBoxes = [], validTexts = [];
    for (let i = 0; i < boxes.length; i++) {
      validBoxes.push(boxes[i]);
      validTexts.push(texts[i] || '');
    }

    // Step 3: textline merge — group into speech bubble regions  [backend.py line 194-196]
    // Mirrors: dispatch_textline_merge(textlines, width, height)
    if (cb) cb('Merging textlines...');
    const bboxes = mergeTextlines(validBoxes, validTexts);

    // Step 4: mask refinement  [backend.py line 217-220, mask_refinement/__init__.py]
    // Use rawMask (pixel-level text mask from detector) clipped to detected regions,
    // then dilate to expand the mask outward around text contours.
    if (cb) cb('Creating mask...');
    const mc = document.createElement('canvas'); mc.width = w; mc.height = h;
    const mctx = mc.getContext('2d', { willReadFrequently: true });
    const rd = rawMask.getContext('2d').getImageData(0, 0, w, h).data;
    const mi = mctx.createImageData(w, h);

    // Copy rawMask pixels that fall within any detected box region (+small margin)
    // This keeps the organic text shape from the detector, not box shapes
    const margin = 5;
    for (const [bx1, by1, bx2, by2] of validBoxes) {
      const rx1 = Math.max(0, bx1 - margin), ry1 = Math.max(0, by1 - margin);
      const rx2 = Math.min(w, bx2 + margin), ry2 = Math.min(h, by2 + margin);
      for (let y = ry1; y < ry2; y++) {
        for (let x = rx1; x < rx2; x++) {
          const idx = y * w + x;
          if (rd[idx * 4] > 30) {
            mi.data[idx * 4] = 255; mi.data[idx * 4 + 1] = 255;
            mi.data[idx * 4 + 2] = 255; mi.data[idx * 4 + 3] = 255;
          }
        }
      }
    }
    mctx.putImageData(mi, 0, 0);

    // Dilate mask to expand text contours outward  [mask_refinement/__init__.py line 36-37]
    // Mirrors: kernel_size = max(mask_shape) * 0.025, cv2.dilate(mask, kernel, iterations=1)
    // This expands the organic text shape outward, not creating boxes
    if (cb) cb('Dilating mask...');
    const padEl = typeof document !== 'undefined' && document.getElementById('inpaint-pad');
    const padPer1000 = padEl ? (parseFloat(padEl.value) || 10) : 10;
    const dilKernel = Math.max(1, Math.round(Math.max(w, h) * padPer1000 / 1000));
    dilateMask(mc, dilKernel);

    // Step 5: inpainting → returns patches (not a full image replacement)
    if (cb) cb('Inpainting...');
    const { canvas: srcCanvas } = await imgToCanvas(imageSrc);
    const patches = await inpaint(srcCanvas, mc, cb, validBoxes);

    return { bboxes, patches };
  }

  // ── Inpaint from brush strokes (for Inpaint tool) ────────────────────────

  /**
   * Build mask + boxes from inpaint brush strokes.
   */
  function buildStrokeMask(strokes, natW, natH) {
    const mc = document.createElement('canvas'); mc.width = natW; mc.height = natH;
    const mctx = mc.getContext('2d');
    mctx.fillStyle = '#000'; mctx.fillRect(0, 0, natW, natH);
    mctx.strokeStyle = '#fff'; mctx.lineCap = 'round'; mctx.lineJoin = 'round';
    const boxes = [];
    for (const s of strokes) {
      if (!s.points.length) continue;
      mctx.lineWidth = s.size;
      mctx.beginPath();
      mctx.moveTo(s.points[0].x, s.points[0].y);
      let minX = s.points[0].x, minY = s.points[0].y;
      let maxX = s.points[0].x, maxY = s.points[0].y;
      for (let i = 1; i < s.points.length; i++) {
        mctx.lineTo(s.points[i].x, s.points[i].y);
        if (s.points[i].x < minX) minX = s.points[i].x;
        if (s.points[i].y < minY) minY = s.points[i].y;
        if (s.points[i].x > maxX) maxX = s.points[i].x;
        if (s.points[i].y > maxY) maxY = s.points[i].y;
      }
      if (s.points.length === 1) mctx.lineTo(s.points[0].x + 0.1, s.points[0].y);
      mctx.stroke();
      const half = s.size / 2;
      boxes.push([
        Math.max(0, Math.round(minX - half)),
        Math.max(0, Math.round(minY - half)),
        Math.min(natW, Math.round(maxX + half)),
        Math.min(natH, Math.round(maxY + half)),
      ]);
    }
    return { maskCanvas: mc, boxes };
  }

  // ── Mask dilation (mirrors cv2.dilate) ─────────────────────────────────
  // Optimized: separable two-pass (horizontal then vertical) for speed
  function dilateMask(canvas, kernelSize) {
    const w = canvas.width, h = canvas.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const src = ctx.getImageData(0, 0, w, h);
    const k = Math.floor(kernelSize / 2);

    // Extract binary mask
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) mask[i] = src.data[i * 4] > 128 ? 1 : 0;

    // Pass 1: horizontal dilation
    const tmp = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let hit = false;
        const x0 = Math.max(0, x - k), x1 = Math.min(w - 1, x + k);
        for (let kx = x0; kx <= x1; kx++) {
          if (mask[y * w + kx]) { hit = true; break; }
        }
        tmp[y * w + x] = hit ? 1 : 0;
      }
    }

    // Pass 2: vertical dilation on tmp
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let hit = false;
        const y0 = Math.max(0, y - k), y1 = Math.min(h - 1, y + k);
        for (let ky = y0; ky <= y1; ky++) {
          if (tmp[ky * w + x]) { hit = true; break; }
        }
        out[y * w + x] = hit ? 1 : 0;
      }
    }

    // Write back
    const dst = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      if (out[i]) {
        dst.data[i * 4] = 255; dst.data[i * 4 + 1] = 255;
        dst.data[i * 4 + 2] = 255; dst.data[i * 4 + 3] = 255;
      }
    }
    ctx.putImageData(dst, 0, 0);
  }

  // ── Public API ───────────────────────────────────────────────────────────

  return {
    extract,
    inpaint,
    buildStrokeMask,
    loadDetector: loadDet,
    loadOCR,
    loadInpainter: loadInp,
    imgToCanvas,
  };
})();
