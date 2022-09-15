import pandas as pd

from tokenizers import ByteLevelBPETokenizer

IU_XRAY_REPORTS = '../../IU_XRAY/indiana_reports.csv'

reports = pd.read_csv(IU_XRAY_REPORTS)
reports = reports.dropna(subset=['Reports'])  # delete empty reports
reports = list(reports['Reports'].values)

with open('/tmp/mimic.txt', 'w') as f:
    for item in reports:
        f.write("%s\n" % item)

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files='/tmp/mimic.txt', vocab_size=20000, min_frequency=2, special_tokens=[
    '<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',
])

tokenizer.save('.', 'mimic')
