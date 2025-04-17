require("math")
math.randomseed(os.time())

local inputs = 2
local hiddenNodes = 2
local outputs = 1
local training = 4

local function initWeight()
	return (math.random() * 2) - 1 -- random float b/w 1, -1
end

local function sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

local function derivative(x)
	return x * (1 - x)
end

local function shuffle(t)
	for i = #t, 2, -1 do
		local j = math.random(1, i)
		t[i], t[j] = t[j], t[i]
	end
end

local function initModel() --initializing table of model vars
	local model = {

		rate = 0.1,
		hiddenLayer = {},
		outputLayer = {},

		hiddenLayerBias = {},
		outputLayerBias = {},

		hiddenWeights = {},
		outputWeights = {},

		trainingSetOrder = { 1, 2, 3, 4 },
		numberOfEpochs = 10000,

		trainingInputs = {
			{ 0.0, 0.0 },
			{ 1.0, 0.0 },
			{ 0.0, 1.0 },
			{ 1.0, 1.0 },
		},

		trainingOutputs = {
			{ 0.0 },
			{ 1.0 },
			{ 1.0 },
			{ 0.0 },
		},
	}

	for i = 1, hiddenNodes do
		model.hiddenLayer[i] = 0
		model.hiddenLayerBias[i] = initWeight()
	end

	for i = 1, outputs do
		model.outputLayer[i] = 0
		model.outputLayerBias[i] = initWeight()
	end

	for i = 1, inputs do
		model.hiddenWeights[i] = {}
		for j = 1, hiddenNodes do
			model.hiddenWeights[i][j] = initWeight()
		end
	end

	for i = 1, hiddenNodes do
		model.outputWeights[i] = {}
		for j = 1, outputs do
			model.outputWeights[i][j] = initWeight()
		end
	end

	return model
end

local function compute()
	local model = initModel()

	for epoch = 1, model.numberOfEpochs do
		if epoch % 1000 == 0 then
			print(epoch, "simulations completed")
		end
		shuffle(model.trainingSetOrder) -- shuffling the order

		for x = 1, #model.trainingSetOrder do
			local i = model.trainingSetOrder[x]

			-- forward pass: compute hidden layer
			for j = 1, hiddenNodes do
				local activation = model.hiddenLayerBias[j]
				for k = 1, inputs do
					activation = activation + model.trainingInputs[i][k] * model.hiddenWeights[k][j]
				end
				model.hiddenLayer[j] = sigmoid(activation)
			end

			-- forward pass:
			-- compute output layer
			for j = 1, outputs do
				local activation = model.outputLayerBias[j]
				for k = 1, hiddenNodes do
					activation = activation + model.hiddenLayer[k] * model.outputWeights[k][j]
				end
				model.outputLayer[j] = sigmoid(activation)
			end

			-- backward pass:
			-- compute delta for output layer
			local deltaOutput = {}
			for j = 1, outputs do
				local error = model.trainingOutputs[i][j] - model.outputLayer[j]
				deltaOutput[j] = error * derivative(model.outputLayer[j])
			end

			-- backward pass:
			-- compute delta for hidden layer
			local deltaHidden = {}
			for j = 1, hiddenNodes do
				local error = 0.0
				for k = 1, outputs do
					error = error + deltaOutput[k] * model.outputWeights[j][k]
				end
				deltaHidden[j] = error * derivative(model.hiddenLayer[j])
			end

			for j = 1, outputs do -- update output weights and biases
				model.outputLayerBias[j] = model.outputLayerBias[j] + deltaOutput[j] * model.rate
				for k = 1, hiddenNodes do
					model.outputWeights[k][j] = model.outputWeights[k][j]
						+ model.hiddenLayer[k] * deltaOutput[j] * model.rate
				end
			end

			for j = 1, hiddenNodes do -- update hidden weights and biases
				model.hiddenLayerBias[j] = model.hiddenLayerBias[j] + deltaHidden[j] * model.rate
				for k = 1, inputs do
					model.hiddenWeights[k][j] = model.hiddenWeights[k][j]
						+ model.trainingInputs[i][k] * deltaHidden[j] * model.rate
				end
			end
		end
	end

	-- final predictions
	for i = 1, training do
		-- forward pass
		for j = 1, hiddenNodes do
			local activation = model.hiddenLayerBias[j]
			for k = 1, inputs do
				activation = activation + model.trainingInputs[i][k] * model.hiddenWeights[k][j]
			end
			model.hiddenLayer[j] = sigmoid(activation)
		end

		for j = 1, outputs do
			local activation = model.outputLayerBias[j]
			for k = 1, hiddenNodes do
				activation = activation + model.hiddenLayer[k] * model.outputWeights[k][j]
			end
			model.outputLayer[j] = sigmoid(activation)
		end
		print(
			string.format(
				"input: %f %f | target: %f | predicted: %f",
				model.trainingInputs[i][1],
				model.trainingInputs[i][2],
				model.trainingOutputs[i][1],
				model.outputLayer[1]
			)
		)
	end
end

compute()
