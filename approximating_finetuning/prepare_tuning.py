import pandas as pd

from approximating_finetuning.helpers import (
    read_file,
    evenly_distributed_sampling,
    train_test_split,
)
from approximating_finetuning.global_params import SHARE_TEST


def text_to_jan_style_tuning_csv(
    text: str,
    out_path_incl_filename: str,
    max_tune_examples: int = None,
    n_samples: int = 5000,  # 5000 => Each samples becomes 525 words long
    sample_n_words: int = None,  # Becomes full step size if None which is generally good
):
    samples = evenly_distributed_sampling(text, n_samples=n_samples, sample_n_words=sample_n_words)
    df = pd.DataFrame(data=samples, columns=['completion'])

    if max_tune_examples is not None:
        print("Limiting tuning examples by taking first", max_tune_examples, "examples")
        df = df.iloc[0:max_tune_examples]

    # Add leading space to completion to make it a valid for tuning
    df['completion'] = ' ' + df['completion']

    # Use empty prompt
    df['prompt'] = ''
    df.to_csv(out_path_incl_filename, index=False)




if __name__ == '__main__':
    text = read_file('data/fairy_tales.txt')
    train, text = train_test_split(
        text=text,
        share_test=SHARE_TEST,
    )
    text_to_jan_style_tuning_csv(
        text=train,
        out_path_incl_filename='data/jan_style_tuning_raw.csv',
        max_tune_examples=None,
    )
