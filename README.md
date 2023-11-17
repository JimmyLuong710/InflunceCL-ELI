# InfluenceCL combine to ELI for incremental learning

## Dependencies

```shell
pip install -r requirements.txt
```

## Quick Start

Train and evaluate models through `utils/main.py`. For example, to train our model on Split CIFAR-10 with a 500 fixed-size buffer, one could execute:
```shell
python utils/main.py --model soif --load_best_args --dataset seq-cifar10 --buffer_size 500
```

To compare with the result of vanilla influence functions, simply run:
```shell
python utils/main.py --model soif --load_best_args --dataset seq-cifar10 --buffer_size 500 --nu 0
```

More datasets and methods are supported. You can find the available options by running:
```shell
python utils/main.py --help
```
