import glob
import os
import ujson
import jsbeautifier
from utils import read_json, compute_rouge, get_file_name


def get_name_and_version(filename):
    filename = get_file_name(filename).replace('.jsonl', '')
    splitted_name = filename.split('_')
    return splitted_name[0], splitted_name[1]

def compute_scores(filename):

    n_elems = 0.
    rouge1 = 0.
    rouge2 = 0.
    rougeL = 0.

    for data in read_json(filename):
        target = data['abstract'].strip()
        reference = data['llm_response'].strip()
        r1, r2, rl = compute_rouge(target, reference)
        rouge1 += r1
        rouge2 += r2
        rougeL += rl
        n_elems += 1

    return (rouge1 / n_elems), (rouge2 / n_elems), (rougeL / n_elems)


def main(main_folder, output_folder):

    outputs = {}
    for filename in glob.glob(os.path.join(main_folder, '**/*.jsonl'), recursive=True):
        persona, version = get_name_and_version(filename)
        outputs.setdefault(persona, {})
        outputs[persona].setdefault(version, {})

        r1, r2, rl = compute_scores(filename)
        outputs[persona][version]['r1'] = r1
        outputs[persona][version]['r2'] = r2
        outputs[persona][version]['rl'] = rl

    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    with open(os.path.join(output_folder, 'cnn_dm.json'), 'w') as writer:
        writer.write(jsbeautifier.beautify(ujson.dumps(outputs), opts))


if __name__ == '__main__':

    main('/Users/giovanni/Downloads/Persona/results/gemma/cnn_dm_v2-output', '/Users/giovanni/Downloads/Persona/results/gemma/scores')
