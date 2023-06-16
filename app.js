const getElement = id => document.querySelector(id)

const canvas = getElement('#canvas')
const outputArea = getElement('#output-area')
const outputLabel = 'Digit Confidence:'
const scaleInput = getElement('#scale-input')
const lineWidthInput = getElement('#line-width-input')
const multipleDrawsCheckbox = getElement('#multiple-draws-checkbox')
const convertCheckbox = getElement('#convert-checkbox')
const clearBtn = getElement('#clear-btn')
const predictBtn = getElement('#predict-btn')

const ctx = canvas.getContext('2d')
const coord = { x: 0, y: 0 }

let scale = scaleInput.value = 21
let lineWidth = lineWidthInput.value = 2
let multipleDraws = false
let convert = true
let pixelMatrix = null

const clear = () => {
    canvas.width = 28 * scale
    canvas.height = 28 * scale

    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = lineWidth * scale
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'white'
    outputArea.innerHTML = outputLabel
}

const startDrawing = e => {
    if (!multipleDraws)
        clear()
    canvas.addEventListener('mousemove', draw)
    reposition(e)
}

const reposition = e => {
    coord.x = e.clientX - canvas.offsetLeft
    coord.y = e.clientY - canvas.offsetTop
}

const stopDrawing = () => canvas.removeEventListener('mousemove', draw)

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
    const bigPixelMatrix = array2Matrix(normalizedPixels, canvas.width)
    const blocks = decomposeMatrix(bigPixelMatrix, scale)
    pixelMatrix = blockAverages(blocks)
}

const pixelize = () => {
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
    outputArea.innerHTML = outputLabel
    output.forEach(element => {
        if (element[1] > 0)
            outputArea.innerHTML += `
            <div class='item'>
                <strong class='digit'>
                    ${element[0]} 
                </strong>
                =
                <span class='probability'>
                    ${element[1]}%
                </span>                
            </div>
        `
    });
}

const predict = () => {
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
    getElement('#scale').innerHTML = scale
    clear()
})
lineWidthInput.addEventListener('input', () => {
    lineWidth = lineWidthInput.value
    getElement('#line-width').innerHTML = lineWidth
    clear()
})
multipleDrawsCheckbox.addEventListener('change', () => {
    multipleDraws = multipleDrawsCheckbox.checked
    clear()
})
convertCheckbox.addEventListener('change', () => {
    convert = convertCheckbox.checked
    clear()
})
canvas.addEventListener('mouseup', () => {
    stopDrawing()
    updatePixelMatrix()
    if (convert)
        pixelize()
    if (!multipleDraws)
        predict()
})
canvas.addEventListener('mousedown', startDrawing)
canvas.addEventListener('mouseout', stopDrawing)
clearBtn.addEventListener('click', clear)
predictBtn.addEventListener('click', predict)

clear()