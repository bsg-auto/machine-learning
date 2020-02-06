import allLabels from './labels.js'

'use strict'
const IMAGE_WIDTH = 100
const IMAGE_HEIGHT = 32
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

const DIGITS_RECTS_OFFSETS = [20, 32, 44, 56, 68]
export const NUM_DIGITS_PER_IMAGE = DIGITS_RECTS_OFFSETS.length  // 5
const DIGITS_RECTS_TOP = 6
const DIGIT_ACTUAL_WIDTH = 14
export const DIGIT_WIDTH = 20
export const DIGIT_HEIGHT = 20
const DIGIT_SIZE = DIGIT_WIDTH * DIGIT_HEIGHT

export const NUM_CLASSES = 10
export const NUM_IMAGES = allLabels.length
const NUM_TRAIN_IMAGES = 10073 //8000

export const NUM_DATASET_ELEMENTS = NUM_IMAGES * NUM_DIGITS_PER_IMAGE
export const NUM_TRAIN_ELEMENTS = NUM_TRAIN_IMAGES * NUM_DIGITS_PER_IMAGE
export const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export default class MnistData {
	constructor() {
		this.shuffledTrainIndex = 0
		this.shuffledTestIndex = 0
	}
	
	async load() {
		const {imagesDataset, labelsDataset} = await this.readStoreAndGetDatasetsFromUserData()
		// const {imagesDataset, labelsDataset} = await this.loadStoredDatasets()
		
		this.imagesDataset = imagesDataset
		this.labelsDataset = labelsDataset
		
		// Create shuffled indices into the train/test set for when we select a
		// random dataset element for training / validation.
		this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
		this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS)
		
		// Slice the the images and labels into train and test sets.
		const bound1 = NUM_TRAIN_ELEMENTS * DIGIT_SIZE
		const bound2 = NUM_TRAIN_ELEMENTS * NUM_CLASSES
		
		this.trainImages = this.imagesDataset.slice(0, bound1)
		this.testImages = this.imagesDataset.slice(bound1)
		this.trainLabels = this.labelsDataset.slice(0, bound2)
		this.testLabels = this.labelsDataset.slice(bound2)
	}
	
	async loadStoredDatasets() {
		console.log('Loading stored datasets.')
		return {
			labelsDataset: new Uint8Array(await (await
					fetch('datasets/bashgah-labels-dataset@1398-11-17@2213.bin')).arrayBuffer()
			),
			imagesDataset: new Float32Array(await (await
					fetch('datasets/bashgah-images-dataset@1398-11-17@2213.bin')).arrayBuffer()
			),
		}
	}
	
	async readStoreAndGetDatasetsFromUserData() {
		console.log("Reading datasets from user's data then storing it ...")
		
		const labelsS = allLabels.map(label => label.substr(0, NUM_DIGITS_PER_IMAGE)).join('')
		
		const labelsDataset = new Uint8Array(labelsS.length * NUM_CLASSES)
		let i = 0
		// Concatenate all one-hots into a huge linear array:
		for (const label of labelsS) {
			const oneHotLabel = new Uint8Array(NUM_CLASSES).fill(0)  // 0: 0%
			oneHotLabel[(Number.parseInt(label))] = 1  // 1: 100%
			labelsDataset.set(oneHotLabel, NUM_CLASSES * i++)
		}
		
		//console.log(labelsDataset)
		//--------------------------------------------------------/
		
		const imgRequests = new Array(NUM_IMAGES)
		const imagesRawData = new Array(NUM_IMAGES)
		
		for (let i = 0; i < NUM_IMAGES; i++) {
			imgRequests[i] = new Promise((resolve, reject) => {
						// Make a request for the MNIST sprited image.
						const img = new Image()
						const canvas = document.createElement('canvas')
						const ctx = canvas.getContext('2d')
						
						img.crossOrigin = ''
						img.onload = () => {
							const w = img.width = img.naturalWidth
							const h = img.height = img.naturalHeight
							
							//const datasetBytesBuffer = new ArrayBuffer(IMAGE_SIZE * 4)
							
							//const chunkSize = 5
							canvas.width = w
							canvas.height = h
							
							// ctx.fillStyle = 'white'
							// ctx.fillRect(0, 0, w, h)
							ctx.drawImage(img, 0, 0)
							
							// linear array:
							const imageRawData = new Float32Array(DIGIT_SIZE * NUM_DIGITS_PER_IMAGE)
							
							const rawData = ctx.getImageData(0, 0, w, h).data
							
							const top = DIGITS_RECTS_TOP
							const bottom = top + DIGIT_HEIGHT
							let index = 0
							for (const left of DIGITS_RECTS_OFFSETS) {
								const right = left + DIGIT_ACTUAL_WIDTH
								const extraPixels = DIGIT_WIDTH - (right - left)
								
								for (let y = top; y < bottom; y++) {
									for (let i = 0; i < extraPixels / 2; i++) imageRawData[index++] = 0
									
									for (let x = left; x < right; x++) {
										const redIndex = (x + y * IMAGE_WIDTH) * 4
										
										const rF = rawData[redIndex] / 255  // the Red   value of Foreground
										const gF = rawData[redIndex + 1] / 255  // the Green value of Foreground
										const bF = rawData[redIndex + 2] / 255  // the Blue  value of Foreground
										const a = rawData[redIndex + 3] / 255  // the Alpha value of Foreground
										
										// Calculate the color on a white (0xFFFFFF) background
										const r = combineColors(rF, 1, a)
										const g = combineColors(gF, 1, a)
										const b = combineColors(bF, 1, a)
										
										// Because the image is almost grayscale, we only include one channel ((r+g+b)/3):
										imageRawData[index++] = 1 - ((r + g + b) / 3)
										// if (i===0 && index < 11) {
										// 	console.log(index-1)
										// 	console.log(x)
										// 	console.log(y)
										// 	console.log(redIndex)
										// 	console.log(rawData[redIndex])
										// 	console.log(rawData[redIndex+1])
										// 	console.log(rawData[redIndex+2])
										// 	console.log(rawData[redIndex+3])
										// 	console.log('----------------------')
										// }
									}
									
									for (let i = 0; i < extraPixels / 2; i++) imageRawData[index++] = 0
								}
								
								imagesRawData[i] = imageRawData
								resolve()
							}
						}
						
						img.src = `captchas/1/${allLabels[i]}.png`
					}
			)
		}
		
		await Promise.all(imgRequests)
		
		// console.log(imagesRawData)
		const imageNumPixels = imagesRawData[0].length
		const imagesDataset = new Float32Array(imagesRawData.length * imageNumPixels)
		
		// Concatenate all pixels to a huge linear array
		for (const [i, imageRawData] of imagesRawData.entries())
			imagesDataset.set(imageRawData, i * imageNumPixels)
		// console.log(imagesDataset)
		//--------------------------------------------------------/
		
		const imagesUrl = window.URL.createObjectURL(new Blob([imagesDataset]))
		const labelsUrl = window.URL.createObjectURL(new Blob([labelsDataset]))
		
		const a = document.createElement('a')
		a.download = `bashgah-images-dataset@1398-11-17@${NUM_IMAGES}.bin`
		a.href = labelsUrl
		a.click()
		a.download = `bashgah-labels-dataset@1398-11-17@${NUM_IMAGES}.bin`
		a.href = imagesUrl
		a.click()
		return {imagesDataset, labelsDataset}
	}
	
	nextTrainBatch(batchSize) {
		return this.nextBatch(
				batchSize, this.trainImages, this.trainLabels, () => {
					this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
					return this.trainIndices[this.shuffledTrainIndex]
				})
	}
	
	nextTestBatch(batchSize) {
		return this.nextBatch(batchSize, this.testImages, this.testLabels, () => {
			this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length
			return this.testIndices[this.shuffledTestIndex]
		})
	}
	
	nextBatch(batchSize, allImages, allLabels, index) {
		const batchImagesArray = new Float32Array(batchSize * DIGIT_SIZE)
		const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)
		const batchIndices = new Uint32Array(batchSize)

		for (let i = 0; i < batchSize; i++) {
			const idx = index()
			
			const image = allImages.slice(idx * DIGIT_SIZE, idx * DIGIT_SIZE + DIGIT_SIZE)
			batchImagesArray.set(image, i * DIGIT_SIZE)
			
			const label = allLabels.slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES)
			batchLabelsArray.set(label, i * NUM_CLASSES)
			
			batchIndices.set([idx], i)
		}
		const xs = tf.tensor2d(batchImagesArray, [batchSize, DIGIT_SIZE])
		const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])
		
		return {xs, labels, indices: batchIndices}
	}
}

function combineColors(foreColor, backColor, alpha) {
	return alpha * foreColor + (1 - alpha) * backColor
}
