import argparse


class My_Parse:

    def get_args(self):
        parser = argparse.ArgumentParser()

        group = parser.add_mutually_exclusive_group()

        group.add_argument('--filename', type=int)
        group.add_argument('--token', type=int)
        group.add_argument('--len', type=int)
        group.add_argument('--size', type=int)
        group.add_argument('--dropout', type=float)
        group.add_argument('--batch_size', type=int)
        group.add_argument('--epochs', type=int)

        return parser.parse_args()
