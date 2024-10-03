using Flux

model = Chain(
    Dense(100, 64, relu),   
    Dropout(0.3),           
    Dense(64, 128, relu),   
    Dropout(0.4),          
    Dense(128, 64, relu),   
    Dense(64, 10),          
    softmax                
)

X = rand(Float32, 100, 1000)

y = Flux.onehotbatch(rand(1:10, 1000), 1:10)

loss(x, y) = Flux.crossentropy(model(x), y)

using Flux: train!, params


optimizer = Flux.ADAM()


for epoch in 1:100
    
    Flux.train!(loss, params(model), [(X, y)], optimizer)

    if epoch % 10 == 0
        println("Epoch $epoch - Loss: $(loss(X, y))")
    end
end
# Predictions
predictions = model(X)

predicted_labels = Flux.onecold(predictions, 1:10)

true_labels = Flux.onecold(y, 1:10)

# Accuracy
accuracy = sum(predicted_labels .== true_labels) / length(true_labels)
println("Accuracy: $accuracy")
