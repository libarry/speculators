"""Export a speculators PARD-2 training checkpoint to PARD inference layout."""

import argparse

from speculators.models.pard2.export import convert_checkpoint_for_infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing model.safetensors from training",
    )
    args = parser.parse_args()
    result = convert_checkpoint_for_infer(args.checkpoint_dir)
    print(result)


if __name__ == "__main__":
    main()
