import argparse
from model_utils import ModelTypes
from mnist_data import MnistDataset
from model import WN
from pathlib import Path


class ProblemTypes:
    MNIST_16_12 = 'mnist_16_12'  # use first 16 rows to free-run the last 12 rows of the mnist digit


def main(config):
    model_path = Path('models')
    if not model_path.exists():
        model_path.mkdir()

    if config.problem == ProblemTypes.MNIST_16_12:
        input_dim = 28
        num_layers = 4  # to have a receptive field of 16
        dataset = MnistDataset(config.batch_size)
    else:
        raise NotImplementedError

    if config.model_type == ModelTypes.WN:
        model = WN(input_dim=input_dim, layer_dim=config.layer_dim,
                   learning_rate=config.learning_rate, num_layers=num_layers)
    else:
        raise NotImplementedError

    if config.mode == 'train':
        model.train(dataset.train_loader, dataset.test_loader, config.epochs, config.patience)
    elif config.mode == 'evaluate':
        loss = model.evaluate(dataset.test_loader)
        print(f"The test loss is {loss}")
    else:
        raise ValueError(f'Unknown mode: {config.mode} choose train or evaluate')


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--problem', type=str, default=ProblemTypes.MNIST_16_12, required=False, help="choose 1 from ProblemTypes")

    # Model params
    parser.add_argument('--model_type', type=str, default=ModelTypes.WN, help='choose from ModelTypes')
    parser.add_argument('--layer_dim', type=int, default=128, help='Number of weights in each layer')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch to train')

    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=3, help='Patience of early stopping')

    # Misc params
    parser.add_argument('--mode', type=str, default='evaluate', help='choose train or validate')

    parsed = parser.parse_args()

    # Train the model
    main(parsed)
