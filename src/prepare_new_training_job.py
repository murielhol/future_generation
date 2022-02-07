from s3_handling import S3Handler
from dataclasses import dataclass, field


@dataclass()
class JobPreparation:
    s3_handler: S3Handler = field(init=False)

    def __post_init__(self):
        self.s3_handler = S3Handler()

    def configuration_to_s3(self):
        self.s3_handler.upload_file_to_s3('configuration.json', f'inputs/configuration.json')


if __name__ == "__main__":
    JobPreparation().configuration_to_s3()
