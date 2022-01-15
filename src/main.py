import argparse
from model_utils import ModelTypes
from mnist_data import MnistDataset
from model import WN, SWN
from pathlib import Path


class ProblemTypes:
    MNIST_16_12 = 'mnist_16_12'  # use first 16 rows to free-run the last 12 rows of the mnist digit


def main(config):
    model_path = Path('models')
    if not model_path.exists():
        model_path.mkdir()

    dataset, input_dim, num_layers = get_dataset_and_related_properties(config)
    model = get_model(config, input_dim, num_layers)
    run(config, model, dataset)


def get_dataset_and_related_properties(config):
    if config.problem == ProblemTypes.MNIST_16_12:
        input_dim = 28
        num_layers = 4  # to have a receptive field of 16
        dataset = MnistDataset(config.batch_size)
    else:
        raise NotImplementedError
    return dataset, input_dim, num_layers


def get_model(config, input_dim, num_layers):
    if config.model_type == ModelTypes.WN:
        model = WN(input_dim=input_dim, layer_dim=config.layer_dim,
                   learning_rate=config.learning_rate, num_layers=num_layers,
                   model_name=ModelTypes.WN + '_' + config.problem)
    elif config.model_type == ModelTypes.SWN:
        model = SWN(input_dim=input_dim, layer_dim=config.layer_dim,
                    learning_rate=config.learning_rate, num_layers=num_layers,
                    model_name=ModelTypes.WN + '_' + config.problem)
    else:

        raise NotImplementedError
    return model


def run(config, model, dataset):
    if config.mode == 'train':
        model.train(dataset.train_loader, dataset.test_loader, config.epochs, config.patience)
    elif config.mode == 'evaluate':
        loss = model.evaluate(dataset.test_loader)
        print(f"The test loss is {loss}")
        if isinstance(dataset, MnistDataset):
            sample_x, _ = dataset.sample_each_digit()
        else:
            print(f'No code to sample dataset of type {type(dataset)}')
            return

        model.visualize_performance(sample_x)

    else:
        raise ValueError(f'Unknown mode: {config.mode} choose train or evaluate')


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--problem', type=str, default=ProblemTypes.MNIST_16_12, required=False,
                        help="choose 1 from ProblemTypes")

    # Model params
    parser.add_argument('--model_type', type=str, default=ModelTypes.SWN, help='choose from ModelTypes')
    parser.add_argument('--layer_dim', type=int, default=128, help='Number of weights in each layer')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch to train')

    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=30, help='Patience of early stopping')

    # Misc params
    parser.add_argument('--mode', type=str, default='train', help='choose train or validate')
    parsed = parser.parse_args()

    # Train the model
    main(parsed)
