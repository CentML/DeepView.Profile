import argparse
import pandas as pd


def percentage_error(predicted, actual):
    return (predicted - actual) / actual


def evaluate_model(model, measurements):
    df = measurements.copy()
    df['samples_per_second_predicted'] = df['batch_size'] / (
        model.run_time_ms_slope * df['batch_size'] +
        model.run_time_ms_bias
    ) * 1000
    df['memory_bytes_predicted'] = (
        model.memory_bytes_slope * df['batch_size'] +
        model.memory_bytes_bias
    )
    df['samples_per_second_error'] = percentage_error(
        df['samples_per_second_predicted'],
        df['samples_per_second'],
    )
    df['memory_usage_error'] = percentage_error(
        df['memory_bytes_predicted'],
        df['memory_usage_bytes'],
    )
    df['model_centered_batch_size'] = model.batch_size
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, required=True)
    parser.add_argument('--measurements', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    models = pd.read_csv(args.models)
    measurements = pd.read_csv(args.measurements)

    frames = [
        evaluate_model(model, measurements)
        for model in models.itertuples()
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
