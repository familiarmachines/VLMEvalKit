from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils.vqa_eval import process_line, hit_calculate
from vlmeval.smp import load, dump, d2df, np, osp, LMUDataRoot

class EmoSet118K(ImageBaseDataset):
    TYPE = 'VQA'

    @classmethod
    def supported_datasets(cls):
        return ['EmoSet118K']

    def load_data(self, dataset):
        # Load from ~/LMUData/EmoSet118K.tsv
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        data = load(data_path)
        data['question'] = [(
            'Describe the image with one of the following emotions: '
            'contentment, '
            'disgust, '
            'anger, '
            'sadness, '
            'amusement, '
            'awe, '
            'fear, '
            'excitement.'
        )] * len(data)
        return data

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        # Last chance to add more to the question.
        for m in msgs:
            if m['type'] == 'text':
                m['value'] += '\nRespond with exactly ONE word. No punctuation.'
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data, 'Need answer & prediction columns'

        # Convert to strings to be safe
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # Remove the trailing full stop from predictions if present
        data['prediction'] = [
            p.strip()[:-1] if p.strip().endswith('.') else p
            for p in data['prediction']
        ]

        # Per-item matching using VLMEvalKit’s VQA helper in 'accuracy' mode.
        # Returns line-level 0/1 comparisons with lower/strip normalization. 
        items = [process_line(data.iloc[i], method='accuracy') for i in range(len(data))]

        # Returns a list of per-item scores; default branch is mean of matches.
        per_item = hit_calculate(items, dataset_name='EmoSet118K')
        overall = float(np.mean(per_item) * 100)
        ret = {'Overall': overall}

        # Save and return a DataFrame (VLMEvalKit convention)
        out = d2df(ret).round(2)
        suffix = eval_file.split('.')[-1]
        dump(out, eval_file.replace(f'.{suffix}', '_acc.csv'))
        cols = data[['answer', 'prediction']]
        dump(cols, eval_file.replace(f'.{suffix}', '_cols.csv'))
        return out


if __name__ == '__main__':
    import base64, random, os
    from typing import Dict, Any, Iterable
    import pandas as pd
    from datasets import load_dataset

    # use os to get HOME environment variable
    HOME_DIR = os.environ.get("HOME", os.path.expanduser("~"))
    TSV_FILE = os.path.join(HOME_DIR, 'LMUData', 'EmoSet118K.tsv')

    ds = load_dataset("Woleek/EmoSet-118K", split="test", streaming=True)
    ds = ds.decode(False)

    def iter_candidates(ds) -> Iterable[Dict[str, Any]]:
        for ex in ds:
            if ex.get("emotion") is not None:
                yield ex

    emotions = set()
    records = []

    cand_stream = iter_candidates(ds)
    for idx, item in enumerate(cand_stream):
        print(idx, item['image_id'])
        e = item['emotion']
        b64 = base64.b64encode(item['image']['bytes']).decode('utf-8')
        answer = e
        r = dict(index=idx, image=b64, answer=answer)
        records.append(r)
        emotions.add(e)

    print("Emotions found:", emotions)
    pd.DataFrame(records).to_csv(TSV_FILE, sep='\t', index=False)
    print(f"wrote {TSV_FILE} with {len(records)} rows")