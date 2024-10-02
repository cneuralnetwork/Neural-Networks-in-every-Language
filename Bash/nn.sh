#!/bin/bash

NUM_INPUTS=3
NUM_HIDDEN=4
NUM_OUTPUTS=2 

for ((i = 0; i < NUM_INPUTS * NUM_HIDDEN; i++)); do
    weights_ih[$i]=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); printf("%.6f\n", rand() - 0.5)}')
done

for ((i = 0; i < NUM_HIDDEN * NUM_OUTPUTS; i++)); do
    weights_ho[$i]=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); printf("%.6f\n", rand() - 0.5)}')
done

for ((i = 0; i < NUM_HIDDEN; i++)); do
    bias_h[$i]=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); printf("%.6f\n", rand() - 0.5)}')
done

for ((i = 0; i < NUM_OUTPUTS; i++)); do
    bias_o[$i]=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); printf("%.6f\n", rand() - 0.5)}')
done

sigmoid() {
    local x=$1
    echo "scale=10; 1 / (1 + e(-($x)))" | bc -l 
}

relu() {
    local x=$1
    awk -v x="$x" 'BEGIN {print (x < 0) ? 0 : x}'
}

derivative_sigmoid() {
    local x=$1
    echo "scale=10; $x * (1 - $x)" | bc -l 
}

derivative_relu() {
    local x=$1
    awk -v x="$x" 'BEGIN {print (x < 0) ? 0 : 1}'
}

forward_pass() {
    local inputs=("$@")
    local hidden_layer=()
    local output_layer=()

    for ((i = 0; i < NUM_HIDDEN; i++)); do
        local sum=0
        for ((j = 0; j < NUM_INPUTS; j++)); do
            sum=$(echo "scale=10; $sum + ${inputs[$j]} * ${weights_ih[$((i*NUM_INPUTS+j))]}" | bc -l)
        done
        sum=$(echo "scale=10; $sum + ${bias_h[$i]}" | bc -l)
        hidden_layer[$i]=$(relu $sum)
    done

    for ((i = 0; i < NUM_OUTPUTS; i++)); do
        local sum=0
        for ((j = 0; j < NUM_HIDDEN; j++)); do
            sum=$(echo "scale=10; $sum + ${hidden_layer[$j]} * ${weights_ho[$((i*NUM_HIDDEN+j))]}" | bc -l)
        done
        sum=$(echo "scale=10; $sum + ${bias_o[$i]}" | bc -l)
        output_layer[$i]=$(sigmoid $sum)
    done

    echo "${output_layer[@]}"
}

backward_pass() {
    local inputs=("$@")
    local targets=("${@: -2}")
    local hidden_layer=()
    local output_layer=()
    local error_gradients=()
    local hidden_gradients=()

    for ((i = 0; i < NUM_HIDDEN; i++)); do
        local sum=0
        for ((j = 0; j < NUM_INPUTS; j++)); do
            sum=$(echo "scale=10; $sum + ${inputs[$j]} * ${weights_ih[$((i*NUM_INPUTS+j))]}" | bc -l)
        done
        sum=$(echo "scale=10; $sum + ${bias_h[$i]}" | bc -l)
        hidden_layer[$i]=$(relu $sum)
    done

    for ((i = 0; i < NUM_OUTPUTS; i++)); do
        local sum=0
        for ((j = 0; j < NUM_HIDDEN; j++)); do
            sum=$(echo "scale=10; $sum + ${hidden_layer[$j]} * ${weights_ho[$((i*NUM_HIDDEN+j))]}" | bc -l)
        done
        sum=$(echo "scale=10; $sum + ${bias_o[$i]}" | bc -l)
        output_layer[$i]=$(sigmoid $sum)
    done

    for ((i = 0; i < NUM_OUTPUTS; i++)); do
        local error=$(echo "scale=10; ${targets[$i]} - ${output_layer[$i]}" | bc -l)
        error_gradients[$i]=$(echo "scale=10; $error * $(derivative_sigmoid ${output_layer[$i]})" | bc -l)
    done

    for ((i = 0; i < NUM_HIDDEN; i++)); do
        local sum=0
        for ((j = 0; j < NUM_OUTPUTS; j++)); do
            sum=$(echo "scale=10; $sum + ${error_gradients[$j]} * ${weights_ho[$((j*NUM_HIDDEN+i))]}" | bc -l)
        done
        hidden_gradients[$i]=$(echo "scale=10; $sum * $(derivative_relu ${hidden_layer[$i]})" | bc -l)
    done

    for ((i = 0; i < NUM_INPUTS*NUM_HIDDEN; i++)); do
        local weight=${weights_ih[$i]}
        local input=${inputs[$(($i%NUM_INPUTS))]}
        local gradient=${hidden_gradients[$(($i/NUM_INPUTS))]}
        weights_ih[$i]=$(echo "scale=10; $weight + 0.1 * $input * $gradient" | bc -l)
    done

    for ((i = 0; i < NUM_HIDDEN*NUM_OUTPUTS; i++)); do
        local weight=${weights_ho[$i]}
        local hidden=${hidden_layer[$(($i%NUM_HIDDEN))]}
        local gradient=${error_gradients[$(($i/NUM_HIDDEN))]}
        weights_ho[$i]=$(echo "scale=10; $weight + 0.1 * $hidden * $gradient" | bc -l)
    done

    for ((i = 0; i < NUM_HIDDEN; i++)); do
        bias_h[$i]=$(echo "scale=10; ${bias_h[$i]} + 0.1 * ${hidden_gradients[$i]}" | bc -l)
    done

    for ((i = 0; i < NUM_OUTPUTS; i++)); do
        bias_o[$i]=$(echo "scale=10; ${bias_o[$i]} + 0.1 * ${error_gradients[$i]}" | bc -l)
    done
}

# Training
for ((epoch = 0; epoch < 100; epoch++)); do
    inputs=(1 2 3)
    targets=(0.5 0.8)
    backward_pass "${inputs[@]}" "${targets[@]}"
done

# Testing
inputs=(1 2 3)
outputs=($(forward_pass "${inputs[@]}"))
echo "Output: ${outputs[@]}"