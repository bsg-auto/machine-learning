'use strict'

import labels from './labels.js'

const IMAGE_WIDTH = 100
const IMAGE_HEIGHT = 32
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

const DIGITS_RECTS_OFFSETS = [20, 32, 44, 56, 68]
const NUM_DIGITS_PER_IMAGE = DIGITS_RECTS_OFFSETS.length
const DIGITS_RECTS_TOP = 6
const DIGIT_ACTUAL_WIDTH = 14
export const DIGIT_WIDTH = 20
export const DIGIT_HEIGHT = 20
const DIGIT_SIZE = DIGIT_WIDTH * DIGIT_HEIGHT

export const NUM_CLASSES = 10
const NUM_IMAGES = labels.length
const NUM_TRAIN_IMAGES = 850

export const NUM_DATASET_ELEMENTS = NUM_IMAGES * NUM_DIGITS_PER_IMAGE
export const NUM_TRAIN_ELEMENTS = NUM_TRAIN_IMAGES * NUM_DIGITS_PER_IMAGE
export const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS

const MNIST_LABELS_PATH = 'mnist_labels_uint8' //'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'

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
		const imgRequests = new Array(NUM_IMAGES)
		const imagesPixels = new Array(NUM_IMAGES)
		
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
							
							ctx.fillStyle = 'white'
							ctx.fillRect(0, 0, w, h)
							ctx.drawImage(img, 0, 0)
							
							// linear array:
							const imagePixels = new Float32Array(DIGIT_SIZE * NUM_DIGITS_PER_IMAGE)
							
							const pixels = ctx.getImageData(0, 0, w, h).data
							
							const top = DIGITS_RECTS_TOP
							const bottom = top + DIGIT_HEIGHT
							let index = 0
							for (const left of DIGITS_RECTS_OFFSETS) {
								const right = left + DIGIT_ACTUAL_WIDTH
								const extraPixels = DIGIT_WIDTH - (right - left)
								
								for (let y = top; y < bottom; y++) {
									for (let i = 0; i < extraPixels / 2; i++) imagePixels[index++] = 0
									
									for (let x = left; x < right; x++) {
										// All channels hold an equal value since the image is almost grayscale, so
										// just read the blue channel (red + 2).
										const redIndex = (x + y * IMAGE_WIDTH) * 4
										imagePixels[index++] = (255 - pixels[redIndex + 2]) / 255
										// if (i===0 && index < 11) {
										// 	console.log(index-1)
										// 	console.log(x)
										// 	console.log(y)
										// 	console.log(redIndex)
										// 	console.log(pixels[redIndex])
										// 	console.log(pixels[redIndex+1])
										// 	console.log(pixels[redIndex+2])
										// 	console.log(pixels[redIndex+3])
										// 	console.log('----------------------')
										// }
									}
									
									for (let i = 0; i < extraPixels / 2; i++) imagePixels[index++] = 0
								}
								
								imagesPixels[i] = imagePixels
								resolve()
							}
						}
						
						img.src = `captchas/1/${labels[i]}.jpg`
					}
			)
		}
		
		await Promise.all(imgRequests)
		
		// console.log(imagesPixels)
		const imageNumPixels = imagesPixels[0].length
		this.datasetImages = new Float32Array(imagesPixels.length * imageNumPixels)
		
		// Concatenate all pixels to a huge linear array
		for (const [i, imagePixels] of imagesPixels.entries())
			this.datasetImages.set(imagePixels, i * imageNumPixels)
		// console.log(this.datasetImages)
		
		// const labelsOneHot = [...labels.join('')].map(digit => {
		// 	const labelOneHot = new Uint8Array(NUM_CLASSES).fill(0)  // 0: 0%
		// 	labelOneHot[(Number.parseInt(digit))] = 1  // 1: 100%
		// 	return labelOneHot
		// })
		//
		// this.datasetLabels = new Uint8Array(labelsOneHot.length * NUM_CLASSES)
		//
		// // Concatenate all probability columns to a huge linear array
		// for (const [i, probabilityColumn] of labelsOneHot.entries())
		// 	this.datasetLabels.set(probabilityColumn, i * NUM_CLASSES)
		
		const labelsS = labels.join('')
		this.datasetLabels = Uint8Array.from(
				tf.oneHot(tf.tensor1d([...labelsS], 'int32'), NUM_CLASSES)
				// Make it linear
						.reshape([labelsS.length * NUM_CLASSES])
						.dataSync()
		)
		//console.log(this.datasetLabels)
		
		// Create shuffled indices into the train/test set for when we select a
		// random dataset element for training / validation.
		this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
		this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS)
		
		// Slice the the images and labels into train and test sets.
		const bound1 = NUM_TRAIN_ELEMENTS * DIGIT_SIZE
		const bound2 =  NUM_TRAIN_ELEMENTS * NUM_CLASSES
		
		this.trainImages = this.datasetImages.slice(0, bound1)
		this.testImages = this.datasetImages.slice(bound1)
		this.trainLabels = this.datasetLabels.slice(0, bound2)
		this.testLabels = this.datasetLabels.slice(bound2)
	}
	
	nextTrainBatch(batchSize) {
		return this.nextBatch(
				batchSize, this.trainImages, this.trainLabels, () => {
					this.shuffledTrainIndex =
							(this.shuffledTrainIndex + 1) % this.trainIndices.length
					return this.trainIndices[this.shuffledTrainIndex]
				})
	}
	
	nextTestBatch(batchSize) {
		return this.nextBatch(batchSize, this.testImages, this.testLabels, () => {
			this.shuffledTestIndex =
					(this.shuffledTestIndex + 1) % this.testIndices.length
			return this.testIndices[this.shuffledTestIndex]
		})
	}
	
	nextBatch(batchSize, allImages, allLabels, index) {
		const batchImagesArray = new Float32Array(batchSize * DIGIT_SIZE)
		const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)
		
		for (let i = 0; i < batchSize; i++) {
			const idx = index()
			
			const image = allImages.slice(idx * DIGIT_SIZE, idx * DIGIT_SIZE + DIGIT_SIZE)
			batchImagesArray.set(image, i * DIGIT_SIZE)
			
			const label = allLabels.slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES)
			batchLabelsArray.set(label, i * NUM_CLASSES)
		}
		
		const xs = tf.tensor2d(batchImagesArray, [batchSize, DIGIT_SIZE])
		const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])
		
		return {xs, labels}
	}
}
