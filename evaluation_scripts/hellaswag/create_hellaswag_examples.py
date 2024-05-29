import ujson
import random


def read_file(filename):
    datas = []
    with open(filename, 'r') as reader:
        for line in reader:
            data = ujson.loads(line.strip())
            datas.append(data)

        return datas


def select_examples(examples):
    random.shuffle(examples)
    return random.choices(examples, k=10)


def write_selected_examples(selected_examples):

    with open('../../hellaswag/selected_val_examples.jsonl', 'w') as writer:
        for example in selected_examples:
            d = {'ind': example['ind'], 'ctx': example['ctx'], 'endings': example['endings'], 'label': example['label']}
            writer.write(f"{ujson.dumps(d)}\n")


def main():

    examples = read_file('../../hellaswag/hellaswag_val.jsonl')
    selected_examples = select_examples(examples)
    write_selected_examples(selected_examples)


if __name__ == '__main__':

    main()