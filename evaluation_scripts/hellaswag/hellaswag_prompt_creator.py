import ujson
from langchain.prompts import PromptTemplate


TEMPLATE = """{persona}
### Instruction:
Select the ending that completes the context in input.
{examples}
### Input:
Context: {sentence}
Endings:
{endings}
### Response:
Answer:
"""
prompt_template = PromptTemplate.from_template(TEMPLATE)


def read_file(filename):
    with open(filename, 'r') as reader:
        for line in reader:
            data = ujson.loads(line.strip())
            yield data


def get_persona(persona):

    if persona:
        return f"""### Persona:
        You are a {persona}. You have to complete the following task as best as you can."""
        #return f"""### Persona:
        #As {persona}, complete the following task."""

    return ""


def get_endings(endings):
    return "\n".join([f"{str(i)}) {s.lower()}" for i, s in enumerate(endings)])


def get_examples():

    examples_ind = []
    examples_str = []

    with open('../../hellaswag/selected_val_examples.jsonl', 'r') as reader:
        for example in reader:
            example = ujson.loads(example)
            examples_ind.append(example['ind'])
            e_str = f"Context: {example['ctx'].lower()}\nEndings:\n{get_endings(example['endings'])}\nAnswer: {example['label']}"
            examples_str.append(e_str)

    return examples_ind, examples_str


def generate_instruction(data, persona, examples):

    output_line = {
        'ind': data['ind'],
        'activity': data['activity_label'],
        'correct_label': data['label'],
        'split_type': data['split_type']
    }

    llm_instruction = prompt_template.invoke({
        'persona': get_persona(persona),
        'examples': examples,
        'sentence': data['ctx'].lower(),
        'endings': get_endings(data['endings'])
    })

    output_line['llm_instruction'] = llm_instruction.to_string()

    return output_line


def main(input_files, persona_list, ten_shot=False):

    if ten_shot:
        examples_ind, examples_str = get_examples()
        examples_str = '\n\n'.join(examples_str)
        example_template = f"## Examples:\n{examples_str}"
        version = 1
    else:
        examples_ind = []
        example_template = ""
        version = 0

    for (persona, description) in persona_list:

        if description and description != "":
            version = 2
            persona += ' ' + description

        if persona:
            name = f'./prompts/hellaswag_v1/{persona}_v{version}.jsonl'
        else:
            name = f'./prompts/hellaswag_v1/no_persona_v{version}.jsonl'

        with open(name, 'w') as writer:
            for file in input_files:
                for example in read_file(file):
                    if ten_shot and (example['ind'] in examples_ind): continue
                    instruction = generate_instruction(example, persona, example_template)
                    writer.write(f'{ujson.dumps(instruction)}\n')


if __name__ == '__main__':

    personas = [(None, None),
                ('Virtual Assistant', 'specialized in narrative coherence and text completion. You are a meticulous '
                                      'editor with a flair for storytelling, possessing an extensive knowledge '
                                      'of narrative structure and a keen eye for detail. '
                                      'Your skill set includes a profound understanding of language nuances '
                                      'and the ability to seamlessly weave together incomplete sections of '
                                      'text to create a cohesive and engaging narrative.'),
                ('Human', 'specialized in narrative coherence and text completion. You are a Human, specialised in '
                          'narrative coherence and text completion. As a seasoned editor and creative writer, '
                          'you possess a keen eye for storytelling flow and the ability to seamlessly weave '
                          'disparate elements into a cohesive whole, ensuring that narrative threads align '
                          'with character development and plot progression.'),
                ('Professor', 'specialized in narrative coherence and text completion. You are known for your sharp '
                              'analytical skills, able to dissect complex narratives and identify the underlying '
                              'structures that make them compelling. Your extensive knowledge in linguistics and a '
                              'keen eye for detail allow you to guide students to mastery in crafting '
                              'coherent and engaging texts.'),
                ('Teacher', 'specialized in narrative coherence and text completion. As a meticulous and engaging '
                            'educator with a passion for literature and language, '
                            'you possess an extensive understanding of storytelling techniques and narrative structures.'
                            ' Your skill lies in guiding students to develop their writing prowess by recognizing '
                            'and crafting cohesive and compelling stories.'),
                ('Student', 'specialized in narrative coherence and text completion. You are a dedicated and insightful'
                            ' Creative Writing Tutor, well-versed in literary devices and story structuring. '
                            'With a talent for guiding students to enhance their writing, you possess a keen '
                            'understanding of how to craft compelling narratives and an ability to '
                            'provide constructive, tailored feedback.'),
                ('Contestant', 'specialized in narrative coherence and text completion. As an avid storyteller and '
                               'linguistic enthusiast, you excel at weaving complex narratives into seamless '
                               'tales that captivate and engage. With a vast knowledge of literature and '
                               'a sharp eye for detail, you possess the unique skill of filling in gaps in stories, '
                               'ensuring they flow logically and maintain their stylistic integrity.'),
                ('Writer', 'specialized in narrative coherence and text completion. You are a meticulous Editor, '
                           'known for your eagle-eye attention to detail and deep understanding of grammar, '
                           'style, and clarity. Your skill set includes a comprehensive knowledge of publishing '
                           'standards and an uncanny ability to enhance the readability of any text while '
                           'preserving the author\'s voice.'),
                ('Journalist', 'specialized in narrative coherence and text completion. You are a meticulous Editor '
                               'with a keen eye for detail and a deep appreciation for storytelling. '
                               'Your expertise lies in refining prose to enhance clarity, flow, and engagement, '
                               'ensuring each story is presented at its highest quality.')
                ]
    hellaswag_files = ['./hellaswag/hellaswag_train.jsonl', './hellaswag/hellaswag_val.jsonl']

    main(hellaswag_files, personas, ten_shot=True)

