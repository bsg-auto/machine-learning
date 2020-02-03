import MnistData, {
	DIGIT_WIDTH,
	DIGIT_HEIGHT,
	NUM_CLASSES,
	NUM_DATASET_ELEMENTS,
	NUM_TEST_ELEMENTS,
	NUM_TRAIN_ELEMENTS,
} from './data.js'

async function showExamples(data) {
	// Create a container in the visor
	const surface =
			tfvis.visor().surface({name: 'Input Data Examples', tab: 'Input Data'})
	
	// Get the examples
	const examples = data.nextTestBatch(20)
	const numExamples = examples.xs.shape[0]

	// Create a canvas element to render each example
	for (let i = 0; i < numExamples; i++) {
		// Reshape the image to DIGIT_HEIGHT x DIGIT_WIDTH px
		const imageTensor = tf.tidy(() =>
				examples.xs
						.slice([i, 0], [1, examples.xs.shape[1]])
						.reshape([DIGIT_HEIGHT, DIGIT_WIDTH, 1]))
		
		const labelTensor = tf.tidy(() =>
				examples.labels
						.slice([i, 0], [1, examples.labels.shape[1]])
						.reshape([10, 1]))
		
		console.log(labelTensor.argMax().arraySync()[0])
		
		const canvas = document.createElement('canvas')
		canvas.width = DIGIT_WIDTH
		canvas.height = DIGIT_HEIGHT
		canvas.style = 'margin: 4px'
		await tf.browser.toPixels(imageTensor, canvas)
		surface.drawArea.appendChild(canvas)
		
		imageTensor.dispose()
		labelTensor.dispose()
	}
}

function getModel() {
	const model = tf.sequential()
	
	const IMAGE_WIDTH = DIGIT_WIDTH
	const IMAGE_HEIGHT = DIGIT_HEIGHT
	const IMAGE_CHANNELS = 1
	
	// In the first layer of our convolutional neural network we have
	// to specify the input shape. Then we specify some parameters for
	// the convolution operation that takes place in this layer.
	model.add(tf.layers.conv2d({
		inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
		kernelSize: 5,
		filters: 8,
		strides: 1,
		activation: 'relu',
		kernelInitializer: 'varianceScaling',
	}))
	
	// The MaxPooling layer acts as a sort of downsampling using max values
	// in a region instead of averaging.
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}))
	
	// Repeat another conv2d + maxPooling stack.
	// Note that we have more filters in the convolution.
	model.add(tf.layers.conv2d({
		kernelSize: 5,
		filters: 16,
		strides: 1,
		activation: 'relu',
		kernelInitializer: 'varianceScaling',
	}))
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}))
	
	// Now we flatten the output from the 2D filters into a 1D vector to prepare
	// it for input into our last layer. This is common practice when feeding
	// higher dimensional data to a final classification output layer.
	model.add(tf.layers.flatten())
	
	// Our last layer is a dense layer which has 10 output units, one for each
	// output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
	const NUM_OUTPUT_CLASSES = NUM_CLASSES
	model.add(tf.layers.dense({
		units: NUM_OUTPUT_CLASSES,
		kernelInitializer: 'varianceScaling',
		activation: 'softmax',
	}))
	
	
	// Choose an optimizer, loss function and accuracy metric,
	// then compile and return the model
	const optimizer = tf.train.adam()
	model.compile({
		optimizer: optimizer,
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy'],
	})
	
	return model
}

async function train(model, data) {
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc']
	const container = {
		name: 'Model Training', styles: {height: '1000px'}
	}
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)
	
	const BATCH_SIZE = 512
	const TRAIN_DATA_SIZE = NUM_TRAIN_ELEMENTS
	const TEST_DATA_SIZE = NUM_TEST_ELEMENTS
	
	const [trainXs, trainYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, DIGIT_HEIGHT, DIGIT_WIDTH, 1]),
			d.labels,
		]
	})
	
	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE)
		//console.log(d.xs)
		return [
			d.xs.reshape([TEST_DATA_SIZE, DIGIT_HEIGHT, DIGIT_WIDTH, 1]),
			d.labels,
		]
	})
	
	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,
		validationData: [testXs, testYs],
		epochs: 10,
		shuffle: true,
		callbacks: fitCallbacks,
	})
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

function doPrediction(model, data, testDataSize = 500) {
	const IMAGE_WIDTH = DIGIT_WIDTH
	const IMAGE_HEIGHT = DIGIT_HEIGHT
	const testData = data.nextTestBatch(testDataSize)
	const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
	const labels = testData.labels.argMax([-1])
	const preds = model.predict(testxs).argMax([-1])
	
	testxs.dispose()
	return [preds, labels]
}


async function showAccuracy(model, data) {
	const [preds, labels] = doPrediction(model, data)
	const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
	const container = {name: 'Accuracy', tab: 'Evaluation'}
	tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
	
	labels.dispose()
}

async function showConfusion(model, data) {
	const [preds, labels] = doPrediction(model, data)
	const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds)
	const container = {name: 'Confusion Matrix', tab: 'Evaluation'}
	tfvis.render.confusionMatrix(
			container, {values: confusionMatrix}, classNames)
	
	labels.dispose()
}

async function run() {
	const data = new MnistData()
	await data.load()
	await showExamples(data)
	
	const model = getModel()
	tfvis.show.modelSummary({name: 'Model Architecture'}, model)
	
	await train(model, data)
	
	await showAccuracy(model, data)
	await showConfusion(model, data)
}

document.addEventListener('DOMContentLoaded', run)
