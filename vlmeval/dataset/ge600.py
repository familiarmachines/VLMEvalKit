from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils.vqa_eval import process_line, hit_calculate
from vlmeval.smp import load, dump, d2df, np, osp, LMUDataRoot


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


def build_label_list(root_dir):
    """
    Walks the directory structure starting at `root_dir` and returns
    a list of dicts with keys: "filename", "label1", "label2".

    Assumes structure:
        root_dir/
            subject_0/
                jpeg/
                    img_0000.jpg ... img_0059.jpg
            ...
            subject_9/
                jpeg/
                    img_0000.jpg ... img_0059.jpg
    """
    # Order of labels for each group of 6 images
    group_labels = [
        ("Arms opened", "Facial expression - Happy"),
        ("Arms opened", "Facial expression - Neutral"),
        ("Arms opened", "Facial expression - Sad"),
        ("Palm-out 'Stop'", "Facial expression - Happy"),
        ("Palm-out 'Stop'", "Facial expression - Neutral"),
        ("Palm-out 'Stop'", "Facial expression - Sad"),
    ]

    all_entries = []

    # Loop over subject directories
    for subject_name in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject_name)
        jpeg_dir = os.path.join(subject_path, "jpeg")

        if not os.path.isdir(jpeg_dir):
            continue  # skip anything that isn't a subject/jpeg folder

        # Collect (index_number, relative_filename) for even-numbered images
        numbered_files = []
        for fname in os.listdir(jpeg_dir):
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue

            # Extract number from filename (e.g. img_0004.jpg -> 4)
            match = re.search(r"(\d+)", fname)
            if not match:
                continue

            num = int(match.group(1))

            # Discard odd-numbered images
            if num % 2 == 1:
                continue

            # Store relative path from root for convenience
            rel_path = os.path.join(subject_name, "jpeg", fname)
            numbered_files.append((num, rel_path))

        # Sort files based on extracted number
        numbered_files.sort(key=lambda x: x[0])

        # Group into chunks of 6 and assign labels
        for i in range(0, len(numbered_files), 6):
            group = numbered_files[i:i + 6]

            # Only process full groups of 6
            if len(group) < 6:
                break

            for idx_in_group, (_, rel_path) in enumerate(group):
                label1, label2 = group_labels[idx_in_group]
                all_entries.append({
                    "filename": rel_path,
                    "label1": label1,
                    "label2": label2,
                })

    return all_entries


if __name__ == "__main__":
    import base64
    import io
    import os
    import re
    import pandas as pd
    from PIL import Image

    root_directory = "data"
    labels = build_label_list(root_directory)
    print(f"Total entries: {len(labels)}")
    records = []

    HOME_DIR = os.environ.get("HOME", os.path.expanduser("~"))
    TSV_FILE = os.path.join(HOME_DIR, 'LMUData', 'GE600.tsv')

    for i, entry in enumerate(labels):
        path = "data/" + entry['filename']
        label1 = entry['label1']
        label2 = entry['label2']
        img = Image.open(path).convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        buffer = buffered.getvalue()
        b64 = base64.b64encode(buffer).decode('utf-8')
        r1 = dict(index=i, image=b64, answer=label1)
        r2 = dict(index=i, image=b64, answer=label2)
        records.append(r1)
        records.append(r2)

    pd.DataFrame(records).to_csv(TSV_FILE, sep='\t', index=False)
    print(f"wrote {TSV_FILE} with {len(records)} rows")