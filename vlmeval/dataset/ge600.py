from .image_base import ImageBaseDataset
from .utils.vqa_eval import process_line, hit_calculate
from ..smp import load, dump, d2df, np, osp, LMUDataRoot

class GE600(ImageBaseDataset):
    TYPE = 'VQA'

    @classmethod
    def supported_datasets(cls):
        return ['GE600']

    def load_data(self, dataset):
        # Load from ~/LMUData/GE600.tsv
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        data = load(data_path)
        data['question'] = [(
            'Describe the image with one of the following emotions or gestures: '
            'Arms opened, '
            "Palm-out 'Stop', "
            'Facial expression - Happy, '
            'Facial expression - Neutral, '
            'Facial expression - Sad.'
        )] * len(data)
        return data

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        # Last chance to add more to the question.
        """
        for m in msgs:
            if m['type'] == 'text':
                m['value'] += '\nRespond with exactly ONE word. No punctuation.'
        """
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data, 'Need answer & prediction columns'

        # Convert to strings to be safe
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # Per-item matching using VLMEvalKit’s VQA helper in 'accuracy' mode.
        # Returns line-level 0/1 comparisons with lower/strip normalization. 
        items = [process_line(data.iloc[i], method='accuracy') for i in range(len(data))]

        # Returns a list of per-item scores; default branch is mean of matches.
        per_item = hit_calculate(items, dataset_name='GE600')
        overall = float(np.mean(per_item) * 100)
        ret = {'Overall': overall}

        # Save and return a DataFrame (VLMEvalKit convention)
        out = d2df(ret).round(2)
        suffix = eval_file.split('.')[-1]
        dump(out, eval_file.replace(f'.{suffix}', '_acc.csv'))
        return out

