let segmenter
const sample = new OffscreenCanvas(1, 1)
const sampleContext = sample.getContext('2d', { willReadFrequently: true })
let alpha

function resize(canvas, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
}

// The chair in this shot is dark and nearly neutral. Follow each outside edge only
// as far as the first continuous strip of Alesya's warmer hair or blouse.
function trimChairEdge(alpha, colors, start, middle, step) {
  let sawChair = false
  let warmPixels = 0
  for (let i = start; i !== middle; i += step) {
    const pixel = i * 4
    const opacity = alpha[i]
    const red = colors[pixel], green = colors[pixel + 1], blue = colors[pixel + 2]
    const warmth = red - blue
    if (opacity > 10 && Math.max(red, green, blue) < 74 && warmth < 19) sawChair = true
    warmPixels = opacity > 105 && warmth > 28 ? warmPixels + 1 : 0
    if (warmPixels < 2) continue
    if (sawChair) {
      const hairEdge = i - step
      for (let j = start; j !== hairEdge; j += step) {
        const feather = Math.max(0, 1 - Math.abs(hairEdge - j) / 3)
        alpha[j] *= feather * feather
      }
    }
    break
  }
}

function makeGuide(frame, mask) {
  const { width, height } = mask
  const values = mask.getAsFloat32Array()
  if (alpha?.length !== values.length) alpha = new Uint8ClampedArray(values.length)

  for (let i = 0; i < values.length; i++) {
    const confidence = values[i]
    const edge = Math.min(1, Math.max(0, (confidence - 0.12) / 0.76))
    alpha[i] = 255 * edge * edge * (3 - 2 * edge)
  }
  if (width !== sample.width || height !== sample.height) {
    resize(sample, width, height)
    sampleContext.drawImage(frame, 0, 0, width, height)
  }
  const colors = sampleContext.getImageData(0, 0, width, height).data
  let left = width, right = -1, top = height, bottom = -1
  for (let y = 0; y < height; y++) {
    const row = y * width
    const middle = row + Math.floor(width / 2)
    trimChairEdge(alpha, colors, row, middle, 1)
    trimChairEdge(alpha, colors, row + width - 1, middle, -1)
    for (let x = 0; x < width; x++) {
      if (alpha[row + x] < 166) continue
      left = Math.min(left, x)
      right = Math.max(right, x)
      top = Math.min(top, y)
      bottom = y
    }
  }
  if (right - left < width * 0.06 || bottom - top < height * 0.12) return null

  const scale = Math.min(1, 192 / Math.max(width, height))
  const guideWidth = Math.max(1, Math.round(width * scale))
  const guideHeight = Math.max(1, Math.round(height * scale))
  const guide = new Uint8Array(guideWidth * guideHeight)
  for (let y = 0; y < guideHeight; y++) {
    const row = Math.floor((y + .5) * height / guideHeight) * width
    for (let x = 0; x < guideWidth; x++) {
      guide[y * guideWidth + x] = alpha[row + Math.floor((x + .5) * width / guideWidth)]
    }
  }

  // The inner and outer bounds leave room for small head movements while a
  // new inference is pending. The solid core preserves facial detail.
  const radius = Math.max(2, Math.round(Math.min(guideWidth, guideHeight) * .08))
  const horizontal = matteBounds(guide, guide, guideWidth, guideHeight, radius, true)
  const { inside, outside } = matteBounds(horizontal.inside, horizontal.outside, guideWidth, guideHeight, radius, false)
  const anchors = []
  for (let y = 1; y < guideHeight - 1; y++) {
    const row = Math.floor((y + .5) * height / guideHeight) * width
    for (let x = 1; x < guideWidth - 1; x++) {
      const i = y * guideWidth + x
      if (inside[i] < 200 || (inside[i - 1] >= 200 && inside[i + 1] >= 200 &&
          inside[i - guideWidth] >= 200 && inside[i + guideWidth] >= 200)) continue
      const pixel = (row + Math.floor((x + .5) * width / guideWidth)) * 4
      if (colors[pixel] - colors[pixel + 2] >= 36) anchors.push(i)
    }
  }
  return { width: guideWidth, height: guideHeight, inside, outside, anchors: Uint16Array.from(anchors) }
}

function matteBounds(minimum, maximum, width, height, radius, horizontal) {
  const inside = new Uint8Array(minimum.length)
  const outside = new Uint8Array(maximum.length)
  const step = horizontal ? 1 : width
  const side = horizontal ? width : height
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const position = horizontal ? x : y
      const start = i - Math.min(position, radius) * step
      const end = i + Math.min(side - 1 - position, radius) * step
      // A subject touching the camera edge gives us no margin on that side.
      let low = horizontal && (position < radius || position >= side - radius) ? 0 : minimum[i]
      let high = maximum[i]
      for (let j = start; j <= end && (low > 0 || high < 255); j += step) {
        low = Math.min(low, minimum[j])
        high = Math.max(high, maximum[j])
      }
      inside[i] = low
      outside[i] = high
    }
  }
  return { inside, outside }
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const { FilesetResolver, ImageSegmenter } = await import(data.assets.runtime)
      const fileset = await FilesetResolver.isSimdSupported() ? data.assets.simd : data.assets.scalar
      segmenter = await ImageSegmenter.createFromOptions(fileset, {
        canvas: new OffscreenCanvas(1, 1),
        baseOptions: { modelAssetPath: data.assets.model, delegate: 'CPU' },
        runningMode: 'VIDEO', outputConfidenceMasks: true, outputCategoryMask: false,
      })
      self.postMessage({ type: 'ready' })
      return
    }
    const frame = data.frame
    let transferred = false
    try {
      const width = frame.displayWidth ?? frame.width
      const height = frame.displayHeight ?? frame.height
      const scale = Math.min(1, 256 / Math.max(width, height))
      resize(sample, Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale)))
      sampleContext.drawImage(frame, 0, 0, sample.width, sample.height)
      let guide
      segmenter.segmentForVideo(sample, data.timestamp, result => {
        // The binary selfie model has one confidence channel, the foreground.
        guide = makeGuide(frame, result.confidenceMasks[0])
      })
      if (guide) {
        self.postMessage({ type: 'mask', guide, frame }, [guide.inside.buffer, guide.outside.buffer, guide.anchors.buffer, frame])
        transferred = true
      } else {
        self.postMessage({ type: 'empty' })
      }
    } finally {
      if (!transferred) frame.close()
    }
  } catch {
    self.postMessage({ type: 'error' })
  }
}
