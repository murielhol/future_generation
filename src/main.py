import argparse
from model_utils import ModelTypes
from mnist_data import MnistDataset
from models import WN, SWN
from pathlib import Path
from dataclasses import dataclass, asdict, field
import json
from s3_handling import S3Handler


@dataclass(frozen=True)
class Configuration:
    problem: str
    model_type: str
    layer_dim: int
    learning_rate: int
    learning_rate: float
    epochs: int
    patience: int
    batch_size: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_dict(self) -> str:
        return asdict(self)


class ProblemTypes:
    MNIST_16_12 = 'mnist_16_12'  # use first 16 rows to free-run the last 12 rows of the mnist digit


@dataclass
class JobRunner:

    mode: str
    s3_handler: S3Handler = field(init=False)

    def __post_init__(self):
        self.s3_handler = S3Handler()

    def get_dataset_and_related_properties(self, config: Configuration):
        if config.problem == ProblemTypes.MNIST_16_12:
            input_dim = 28
            num_layers = 4  # to have a receptive field of 16
            dataset = MnistDataset(config.batch_size, data_path='data', download=True)
        else:
            raise NotImplementedError
        return dataset, input_dim, num_layers

    def get_model(self, config: Configuration, input_dim: int, num_layers: int, model_name: str):
        if config.model_type == ModelTypes.WN:
            model = WN(input_dim=input_dim, layer_dim=config.layer_dim,
                       learning_rate=config.learning_rate, num_layers=num_layers,
                       model_name=model_name)
        elif config.model_type == ModelTypes.SWN:
            model = SWN(input_dim=input_dim, layer_dim=config.layer_dim,
                        learning_rate=config.learning_rate, num_layers=num_layers,
                        model_name=model_name)
        else:
            raise NotImplementedError
        return model

    @staticmethod
    def save_configuration(model_name: str, config: Configuration):
        with open(Path('outputs', model_name, 'configuration.json'), 'w') as fp:
            json.dump(config.to_json(), fp)

    @staticmethod
    def init_file_structure(model_name):
        model_path = Path(f'outputs')
        if not model_path.exists():
            model_path.mkdir()
        model_path = Path(f'outputs/{model_name}')
        if not model_path.exists():
            model_path.mkdir()

    def get_config(self, model_name) -> Configuration:
        if self.mode == 'train':
            # when training, get the configuration that has been uploaded to the input folder
            config_bytes = self.s3_handler.get_file_from_s3(f'inputs/configuration.json')
            config_json = json.load(config_bytes)
        else:
            # when evaluating, download the result outputs from training, and use that configuration
            assert model_name, 'specify the model name to evaluate'
            config_bytes = self.s3_handler.get_file_from_s3(f'outputs/{model_name}/configuration.json')
            config_json = json.loads(json.load(config_bytes))

        config = Configuration(**config_json)
        return config

    def run(self, model_name=None):
        config = self.get_config(model_name)
        print(f"------- configuration --------- \n {config.to_dict()}")
        model_name = f"{config.problem}_{config.model_type}"
        print(f'Model name: {model_name}')

        self.init_file_structure(model_name)
        dataset, input_dim, num_layers = self.get_dataset_and_related_properties(config)
        model = self.get_model(config, input_dim, num_layers, model_name)

        if self.mode == 'train':
            model.train(dataset.train_loader, dataset.test_loader, config.epochs, config.patience)
            self.save_configuration(model_name, config)
            self.s3_handler.upload_outputs_to_s3(model_name)

        elif self.mode == 'evaluate':
            self.s3_handler.download_file_from_s3(f'outputs/{model_name}/state.pth',
                                                  f'outputs/{model_name}/state.pth')
            loss = model.evaluate(dataset.test_loader)
            print(f"The test loss is {loss}")
            if isinstance(dataset, MnistDataset):
                sample_x, _ = dataset.sample_each_digit()
            else:
                print(f'No code to sample dataset of type {type(dataset)}')
                return

            model.visualize_performance(sample_x)

        else:
            raise ValueError(f'Unknown mode: {self.mode} choose train or evaluate')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'])
    parser.add_argument('--model_name', type=str, default='mnist_16_12_wn')
    parsed = parser.parse_args()
    jr = JobRunner(parsed.mode)
    jr.run(model_name=parsed.model_name)
