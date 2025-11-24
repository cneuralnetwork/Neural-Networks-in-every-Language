;;; XOR Neural Network in Common Lisp
;;; Architecture: 2 inputs, 2 hidden neurons, 1 output
;;; Run with: sbcl --script nn.lisp (or clisp nn.lisp)

(defun sigmoid (x)
  "Sigmoid activation function"
  (/ 1.0 (+ 1.0 (exp (- x)))))

(defun sigmoid-derivative (y)
  "Derivative of sigmoid (using output value)"
  (* y (- 1.0 y)))

(defun dot-product (vec1 vec2)
  "Compute dot product of two vectors"
  (reduce #'+ (mapcar #'* vec1 vec2)))

(defun matrix-vector-mult (matrix vector)
  "Multiply matrix by vector"
  (mapcar (lambda (row) (dot-product row vector)) matrix))

(defun vector-add (vec1 vec2)
  "Add two vectors element-wise"
  (mapcar #'+ vec1 vec2))

(defun vector-scalar-mult (scalar vec)
  "Multiply vector by scalar"
  (mapcar (lambda (x) (* scalar x)) vec))

(defun apply-sigmoid (vector)
  "Apply sigmoid to each element of vector"
  (mapcar #'sigmoid vector))

(defun random-weight ()
  "Generate random weight in range [-0.5, 0.5]"
  (- (random 1.0) 0.5))

(defun make-weight-matrix (rows cols)
  "Create random weight matrix"
  (loop repeat rows collect
        (loop repeat cols collect (random-weight))))

(defun make-bias-vector (size)
  "Create random bias vector"
  (loop repeat size collect (random-weight)))

(defun forward-pass (input hidden-weights hidden-biases output-weights output-biases)
  "Perform forward pass through network"
  (let* ((hidden-weighted (matrix-vector-mult hidden-weights input))
         (hidden-pre (vector-add hidden-weighted hidden-biases))
         (hidden-layer (apply-sigmoid hidden-pre))
         (output-weighted (matrix-vector-mult output-weights hidden-layer))
         (output-pre (vector-add output-weighted output-biases))
         (output-layer (apply-sigmoid output-pre)))
    (values output-layer hidden-layer)))

(defun train-network (training-inputs training-outputs num-epochs learning-rate)
  "Train neural network using backpropagation"
  (let ((num-inputs 2)
        (num-hidden 2)
        (num-outputs 1)
        (hidden-weights (make-weight-matrix num-inputs 2))
        (hidden-biases (make-bias-vector 2))
        (output-weights (make-weight-matrix 2 1))
        (output-biases (make-bias-vector 1)))
    
    ;; Training loop
    (loop for epoch from 1 to num-epochs do
          (loop for input in training-inputs
                for target in training-outputs do
                
                ;; Forward pass
                (multiple-value-bind (output-layer hidden-layer)
                    (forward-pass input hidden-weights hidden-biases 
                                  output-weights output-biases)
                  
                  ;; Compute output errors
                  (let* ((output-errors
                          (mapcar (lambda (t-val o-val)
                                    (* (- t-val o-val) 
                                       (sigmoid-derivative o-val)))
                                  target output-layer))
                         
                         ;; Compute hidden errors
                         (hidden-errors
                          (loop for j from 0 below (length hidden-layer)
                                collect
                                (* (sigmoid-derivative (nth j hidden-layer))
                                   (loop for k from 0 below (length output-errors)
                                         sum (* (nth k output-errors)
                                                (nth j (nth k output-weights))))))))
                    
                    ;; Update output weights and biases
                    (loop for j from 0 below (length output-weights) do
                          (loop for k from 0 below (length (nth j output-weights)) do
                                (incf (nth k (nth j output-weights))
                                      (* learning-rate 
                                         (nth k output-errors)
                                         (nth j hidden-layer))))
                          (incf (nth j output-biases)
                                (* learning-rate (nth j output-errors))))
                    
                    ;; Update hidden weights and biases
                    (loop for j from 0 below (length hidden-weights) do
                          (loop for k from 0 below (length (nth j hidden-weights)) do
                                (incf (nth k (nth j hidden-weights))
                                      (* learning-rate 
                                         (nth k hidden-errors)
                                         (nth j input))))
                          (incf (nth j hidden-biases)
                                (* learning-rate (nth j hidden-errors))))))))
    
    ;; Return trained weights and biases
    (values hidden-weights hidden-biases output-weights output-biases)))

(defun test-network (training-inputs training-outputs 
                     hidden-weights hidden-biases output-weights output-biases)
  "Test the trained network"
  (format t "~%Training complete! Testing network:~%~%")
  (loop for input in training-inputs
        for expected in training-outputs do
        (multiple-value-bind (output hidden)
            (forward-pass input hidden-weights hidden-biases 
                          output-weights output-biases)
          (format t "Input: ~{~,1F ~}, Output: ~,6F, Expected: ~{~,1F~}~%" 
                  input (first output) expected))))

(defun main ()
  "Main function to run the neural network"
  ;; Training data for XOR
  (let ((training-inputs '((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0)))
        (training-outputs '((0.0) (1.0) (1.0) (0.0)))
        (num-epochs 10000)
        (learning-rate 0.1))
    
    (format t "Training XOR Neural Network in Common Lisp...~%")
    
    ;; Train the network
    (multiple-value-bind (hidden-weights hidden-biases output-weights output-biases)
        (train-network training-inputs training-outputs num-epochs learning-rate)
      
      ;; Test the network
      (test-network training-inputs training-outputs
                    hidden-weights hidden-biases output-weights output-biases))))

;; Run the program
(main)
