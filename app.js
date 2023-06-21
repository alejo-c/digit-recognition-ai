const getElement = id => document.querySelector(id)

const scaleInput = getElement('#scale-input')
const lineWidthInput = getElement('#line-width-input')

const scaleLabel = getElement('#scale-label')
const lineWidthLabel = getElement('#line-width-label')

const multipleDrawsCheckbox = getElement('#multiple-draws-checkbox')
const convertCheckbox = getElement('#convert-checkbox')
const hotColormapCheckbox = getElement('#hot-colormap-checkbox')

const clearBtn = getElement('#clear-btn')
const predictBtn = getElement('#predict-btn')
const defaultScaleBtn = getElement('#default-scale-btn')
const defaultLineWidthBtn = getElement('#default-line-width-btn')

const canvas = getElement('#canvas')

const probabilities = getElement('#probabilities')

const ctx = canvas.getContext('2d')
const coord = { x: 0, y: 0 }
const wrongDigitLabel = 'If it\'s wrong, choose the correct digit:<br>'

const defaultScale = 24
const defaultLineWidth = 2
let scale = scaleInput.value = defaultScale
let lineWidth = lineWidthInput.value = defaultLineWidth
let isDrawing = false
let itWasDrawed = false
let allowsMultipleDraws = false
let canConvertTo28x28 = true
let bigPixelMatrix = null
let pixelMatrix = null

const clear = () => {
    canvas.width = 28 * scale
    canvas.height = 28 * scale

    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = lineWidth * scale
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'white'

    scaleLabel.innerHTML = scale
    lineWidthLabel.innerHTML = lineWidth
    probabilities.innerHTML = ''
}

const startDrawing = e => {
    isDrawing = true
    if (!allowsMultipleDraws)
        clear()
    canvas.addEventListener('mousemove', draw)
    canvas.addEventListener('touchmove', draw)
    reposition(e)
    itWasDrawed = true
}

const reposition = e => {
    coord.x = e.clientX - canvas.offsetLeft
    coord.y = e.clientY - canvas.offsetTop
}

const stopDrawing = () => {
    if (!isDrawing)
        return
    isDrawing = false
    canvas.removeEventListener('mousemove', draw)
    canvas.removeEventListener('touchmove', draw)
    updatePixelMatrix()
    if (canConvertTo28x28)
        convertTo28x28()
    if (!allowsMultipleDraws)
        predict()
}

const draw = e => {
    ctx.beginPath()
    ctx.moveTo(coord.x, coord.y)
    reposition(e)
    ctx.lineTo(coord.x, coord.y)
    ctx.stroke()
}

const normalizeData = pixelData => {
    const normalizedData = []
    // Loop through the pixel data and convert it to a matrix of values ranging from 0 to 255
    for (let i = 0; i < pixelData.length; i += 4) {
        const r = pixelData[i]
        const g = pixelData[i + 1]
        const b = pixelData[i + 2]
        const grayscaleValue = (r + g + b) / 3 // Convert to grayscale

        // Normalize the grayscale value to a range from 0 to 255
        const normalizedValue = Math.round((grayscaleValue / 255) * 255)

        // Add the normalized value to the matrix
        normalizedData.push(normalizedValue)
    }
    return normalizedData
}

const array2Matrix = (array, size) => {
    let matrix = []
    let row = []
    for (const i of array) {
        row.push(i)
        if (row.length === size) {
            matrix.push(row)
            row = []
        }
    }
    return matrix
}

const decomposeMatrix = (matrix, blockSize) => {
    const blocks = []
    const step = Math.floor(blockSize)
    for (let i = 0; i < matrix.length; i += step) { // x
        for (let j = 0; j < matrix.length; j += step) { // y
            let block = []
            for (let k = i; k < i + step; k++) {
                let row = []
                for (let l = j; l < j + step; l++) {
                    row.push(matrix[k][l])
                }
                block.push(row)
            }
            blocks.push(block)
        }
    }
    return blocks
}

const blockAverages = blocks => {
    const matrix = []
    let row = []
    for (const block of blocks) {
        let sum = 0
        for (i = 0; i < block.length; i++) {
            for (j = 0; j < block.length; j++) {
                sum += block[i][j]
            }
        }
        row.push(sum / scale ** 2)
        if (row.length === Math.sqrt(blocks.length)) {
            matrix.push(row)
            row = []
        }
    }
    return matrix
}

const fixData = matrix => {
    const fixedMatrix = []
    for (let i = 0; i < matrix.length; i++) {
        let row = []
        for (let j = 0; j < matrix.length; j++)
            row.push([matrix[i][j] / 255])
        fixedMatrix.push(row)
    }
    return fixedMatrix
}

const updatePixelMatrix = () => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const pixelData = imageData.data
    const normalizedPixels = normalizeData(pixelData)
    bigPixelMatrix = array2Matrix(normalizedPixels, canvas.width)
    const blocks = decomposeMatrix(bigPixelMatrix, scale)
    pixelMatrix = blockAverages(blocks)
}

const convertTo28x28 = () => {
    for (let i = 0; i < pixelMatrix.length; i++) {
        for (let j = 0; j < pixelMatrix.length; j++) {
            const pixelValue = Math.floor(pixelMatrix[i][j])
            ctx.fillStyle = `rgb(${pixelValue}, ${pixelValue}, ${pixelValue})`
            ctx.fillRect(j * scale, i * scale, scale, scale)
        }
    }
}

const loadModel = async () => await tf.loadLayersModel('digits_model.json')

const drawOutput = output => {
    probabilities.innerHTML = ''
    output.forEach(element => {
        if (element[1] > 0)
            probabilities.innerHTML += `<li class='item'>
                <strong class='digit'>
                    ${element[0]} 
                </strong>
                =
                <span class='probability'>
                    ${element[1]}%
                </span>                
            </li>`
    })
}

const predict = () => {
    if (!itWasDrawed)
        return

    const digit = fixData(pixelMatrix)
    const input = tf.tensor4d([digit])

    loadModel().then(model => {
        const output = model.predict(input)
        const probability = output.arraySync()[0]
        const arr = []
        probability.forEach((e, i) =>
            arr.push([i, +(e * 100).toFixed(4)])
        )
        drawOutput(arr.sort((a, b) => b[1] - a[1]))
    })
}

scaleInput.addEventListener('input', () => {
    scale = scaleInput.value
    clear()
})
lineWidthInput.addEventListener('input', () => {
    lineWidth = lineWidthInput.value
    clear()
})
multipleDrawsCheckbox.addEventListener('change', () => {
    allowsMultipleDraws = multipleDrawsCheckbox.checked
    clear()
})
convertCheckbox.addEventListener('change', () => {
    canConvertTo28x28 = convertCheckbox.checked
    clear()
})
defaultScaleBtn.addEventListener('click', () => {
    scale = scaleInput.value = defaultScale
    clear()
})
defaultLineWidthBtn.addEventListener('click', () => {
    lineWidth = lineWidthInput.value = defaultLineWidth
    clear()
})

canvas.addEventListener('mousedown', startDrawing)
canvas.addEventListener('mouseup', stopDrawing)
canvas.addEventListener('mouseout', stopDrawing)

canvas.addEventListener('touchstart', startDrawing)
canvas.addEventListener('touchend', stopDrawing)

clearBtn.addEventListener('click', clear)
predictBtn.addEventListener('click', predict)
document.addEventListener('DOMContentLoaded', clear)

scaleInput.setAttribute('max', defaultScale)