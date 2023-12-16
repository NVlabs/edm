import argparse

from os.path import join

from utils import load_dataset, isotropic_score, gaussian_score, save_parameters

def configure_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("score_name", type=str, help="Name of the score to calculate")
    parser.add_argument("parameters_dir", type=str, help="Calculated score parameters directory")
    return parser

def main(dataset_name, score_name, parameters_dir):
    dataset = load_dataset(dataset_name)

    if score_name == "isotropic":
        parameters = isotropic_score(dataset)
    elif score_name == "gaussian":
        parameters = gaussian_score(dataset)
    else:
        raise ValueError(f"No {score_name}")

    save_parameters(parameters, join(parameters_dir, score_name + ".pt"))


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))

