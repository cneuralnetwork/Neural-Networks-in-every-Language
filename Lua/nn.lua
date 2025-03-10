math.randomseed(os.time())  

function weights(rows, columns)
    local weight = {}
    for i = 1, rows do
        weight[i] = {}
        for j = 1, columns do
            weight[i][j] = (math.random() * 2 - 1) * 0.01 -- stable values
        end
    end
    return weight
end

function loss(prediction, target)
    return 0.5 * (prediction[1] - target[1]) ^ 2
end

function relu(value)
    return math.max(0, value)
end

function derivative(value)
    return value > 0 and 1 or 0
end

function mat_vec(matrix, vector)
    local result = {}
    for i = 1, #matrix do
        result[i] = 0
        for j = 1, #vector do
            result[i] = result[i] + matrix[i][j] * vector[j]
        end
    end
    return result
end

input_size = 3 -- network dimensions
hidden_size = 3  
output_size = 1
learning_rate = 0.01

weight_1 = weights(hidden_size, input_size)  
bias_1 = {0, 0, 0}  

weight_2 = weights(output_size, hidden_size)  
bias_2 = {0}  

function forward(input)
    local hidden = mat_vec(weight_1, input)
    for i = 1, #hidden do
        hidden[i] = relu(hidden[i] + bias_1[i])  
    end

    local output = mat_vec(weight_2, hidden)
    output[1] = output[1] + bias_2[1]  

    return hidden, output
end

function backward(input, hidden, output, target)
    local error = output[1] - target[1]  

    local weight_2_gradient = {}
    for i = 1, #hidden do
        weight_2_gradient[i] = error * hidden[i]
    end
    local bias_2_gradient = error  

    local weight_1_gradient = {}
    for i = 1, hidden_size do
        weight_1_gradient[i] = {}
        local hidden_gradient = weight_2[1][i] * error * derivative(hidden[i])
        for j = 1, input_size do
            weight_1_gradient[i][j] = hidden_gradient * input[j]
        end
        bias_1[i] = bias_1[i] - learning_rate * hidden_gradient  
    end

    for i = 1, output_size do
        for j = 1, hidden_size do
            weight_2[i][j] = weight_2[i][j] - learning_rate * weight_2_gradient[j]
        end
    end
    bias_2[1] = bias_2[1] - learning_rate * bias_2_gradient  

    for i = 1, hidden_size do
        for j = 1, input_size do
            weight_1[i][j] = weight_1[i][j] - learning_rate * weight_1_gradient[i][j]
        end
    end
end

input = {1.0, 2.0, 3.0}  -- random data
target = {1.0}  

for epoch = 1, 1000 do
    local hidden, output = forward(input)
    local current_loss = loss(output, target)  

    backward(input, hidden, output, target)

    if epoch % 100 == 0 then
        print("Epoch:", epoch, " Loss:", current_loss)
    end
end

local _, final_output = forward(input)
print("final: ", final_output[1])
