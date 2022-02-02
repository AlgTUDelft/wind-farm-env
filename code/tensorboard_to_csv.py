import argparse
from glob import glob
import pandas as pd
import progressbar

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Tensorboard data to csv')
    # Common arguments
    parser.add_argument('--path', type=str, default='data/action_representations_tunnel',
                        help='the directory with tensorboard logs sorted by seeds')

    args = parser.parse_args()

    path = args.path
    in_dirs = glob(f"{path}/seed_*/results/*")
    big_df = None
    big_df_eval = None
    for in_dir in progressbar.progressbar(in_dirs):
        seed = int(next(i for i in in_dir.split('/') if i.startswith('seed')).split('_')[1])
        name = in_dir.split('/')[-1]
        x = EventAccumulator(in_dir)
        x.Reload()
        tags = x.Tags().get('scalars')
        df = None
        df_eval = None
        for tag in tags:
            col = pd.DataFrame(x.Scalars(tag)).drop(columns="wall_time").rename(columns={'value': tag})
            if tag.startswith('eval'):
                if df_eval is None:
                    df_eval = col
                else:
                    df_eval = df_eval.merge(col, left_on='step', right_on='step', how='outer')
            else:
                if df is None:
                    df = col
                else:
                    df = df.merge(col, left_on='step', right_on='step', how='outer')
        if df is not None:
            df.insert(0, 'seed', seed)
            df.insert(1, 'algorithm', name)
            if big_df is None:
                big_df = df
            else:
                big_df = big_df.append(df, ignore_index=True)
        if df_eval is not None:
            df_eval.insert(0, 'seed', seed)
            df_eval.insert(1, 'algorithm', name)
            if big_df_eval is None:
                big_df_eval = df_eval
            else:
                big_df_eval = big_df_eval.append(df_eval, ignore_index=True)

    big_df.to_csv(f'{path}/train.csv', index=False)
    big_df_eval.to_csv(f'{path}/eval.csv', index=False)
