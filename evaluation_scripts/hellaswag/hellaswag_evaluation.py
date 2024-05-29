import ujson
import glob
import os
import jsbeautifier
import regex as re

from pathlib import Path
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, FactorRange


def get_file_name(filename):

    name = Path(filename).stem.split('_')
    persona = name[0]
    version = name[1].replace('.jsonl', '')

    return persona, version


def extract_label(text):
    try:
        num_regex = re.compile(r'[0-9]+')
        numbers = [int(x) for x in num_regex.findall(text)]
        return numbers[0]
    except Exception:
        return None


def read_json(filename):

    with open(filename, 'r') as reader:
        for row in reader:
            data = ujson.loads(row.strip())
            yield data


def compute_accuracy(filename):

    n_elems = 0.
    accuracy = 0.

    for data in read_json(filename):
        correct_label = int(data['correct_label'])
        selected_label = extract_label(data['llm_response'])
        n_elems += 1

        if selected_label:
            if selected_label == correct_label:
                accuracy += 1

    return accuracy / n_elems


def plot_scores(filename):

    with open(filename, 'r') as reader:
        json_dict = ujson.load(reader)

    no_persona = json_dict['no persona']['v0']

    plot_data = {
        'persona': [],
        'v0': [],
        'v1': []
    }

    for persona in json_dict.keys():
        if persona != 'no persona':
            plot_data['persona'].append(persona)
            plot_data['v0'].append(json_dict[persona]['v0'])
            plot_data['v1'].append(json_dict[persona]['v1'])

    x = [(persona, version) for persona in plot_data['persona'] for version in ['v0', 'v1']]
    counts = sum(zip(plot_data['v0'], plot_data['v1']), ())
    colors = ["#c9d9d3", "#718dbf"]

    source = ColumnDataSource(data=dict(x=x, counts=counts))

    plot = figure(x_range=FactorRange(*x), height=350)
    plot.vbar(x='x', top='counts', width=0.9, source=source)

    plot.line(range(len(x)), no_persona)

    plot.y_range.start = 0
    plot.x_range.range_padding = 0.1
    plot.xgrid.grid_line_color = None
    plot.axis.minor_tick_line_color = None
    plot.xaxis.major_label_orientation = "vertical"
    plot.outline_line_color = None

    show(plot)


def main(main_folder, output_path):

    outputs = {}
    for filename in glob.glob(os.path.join(main_folder, '**/*.jsonl'), recursive=True):
        persona, version = get_file_name(filename)
        outputs.setdefault(persona, {})

        accuracy = compute_accuracy(filename)
        outputs[persona][version] = accuracy


    opts = jsbeautifier.default_options()
    opts.indent_size = 2

    if not os.path.exists(output_path): os.makedirs(output_path)
    with open(os.path.join(output_path, 'hellaswag.json'), 'w') as writer:
        writer.write(jsbeautifier.beautify(ujson.dumps(outputs), opts))


if __name__ == '__main__':

    main('/Users/giovanni/Downloads/Persona/results/mixtral/hellaswag_v2-output', '/Users/giovanni/Downloads/Persona/results/mixtral/scores')

    #plot_scores('scores/hellaswag_v1/scores/hellaswag.json')