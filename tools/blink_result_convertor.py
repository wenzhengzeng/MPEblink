import json
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(
        description='Convert frame-level blink scores into blink intervals.')
    parser.add_argument(
        'input',
        nargs='?',
        default='results/results_instblink_r50_test.json',
        help='Input prediction JSON file.')
    parser.add_argument(
        'output',
        nargs='?',
        default='results/results_blink_converted.json',
        help='Output JSON file.')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Frame-level blink score threshold.')
    return parser.parse_args()


def scores_to_intervals(scores, threshold):
    intervals = []
    start = None
    for index, score in enumerate(scores):
        if score >= threshold and start is None:
            start = index
        if score < threshold and start is not None:
            average = sum(scores[start:index]) / (index - start)
            intervals.append([start, index - 1, average])
            start = None

    if start is not None:
        average = sum(scores[start:]) / (len(scores) - start)
        intervals.append([start, len(scores) - 1, average])
    return intervals


def main():
    args = parse_args()
    with open(args.input, encoding='utf-8') as source:
        results = json.load(source)

    for query in results:
        query['blinks_converted'] = scores_to_intervals(
            query['blink_scores'], args.threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as destination:
        json.dump(results, destination)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
