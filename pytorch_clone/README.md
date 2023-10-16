An attempt to clone [***Pytorch***](https://github.com/pytorch/pytorch)

## TO DO
- Ops:
    - conv2d
- Layers:
    - lstm.
- Optimizer:
    - SGD with momentum
    - Adam
- Backward propagation:
    - Add matmul backward.
    - Relu
    - Softmax
    - LogSoftmax
    - CrossEntropy
    - Maximum
    - NLLLoss   
    - max/min
    - select (tensor[indices])

- Other:
    - Add grad check.
    - Add Trainer api.
    - Add datasets (mnist..).
    - Add graviz for backward viz.



## To test
--------
```bash
python -m pytest # to test 
```