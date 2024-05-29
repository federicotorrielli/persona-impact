import os
import utils
import ujson
import glob

from langchain.prompts import PromptTemplate


TEMPLATE = """{persona}
### Instruction:
Given the following article, write a short summary of the article in 2-3 sentences.
{examples}
### Input:
Article: 
{article}
### Response:
Summary:
"""

prompt_template = PromptTemplate.from_template(TEMPLATE)


def get_persona(persona):

    if persona:
        return f"""### Persona:
        You are a {persona}. You have to complete the following task as best as you can."""

    return ""


def get_example_template(examples):

    out = "#### Examples:\n"
    for (article, abstract) in examples:
        out += f"Article:\n{article}\nSummary:\n{abstract}\n\n"

    return out


def generate_instruction(filename, persona, examples):

    article, abstract = utils.read_txt(filename)
    output_line = {
        'filename': utils.get_file_name(filename),
        'article': ' '.join(article.split(' ')[:250]),
        'abstract': ' '.join(abstract.split(' ')[:100])
    }

    llm_instruction = prompt_template.invoke({
        'persona': get_persona(persona),
        'examples': examples,
        'article': article,
    })

    output_line['llm_instruction'] = llm_instruction.to_string()

    return output_line


def main(folderpath, persona_list, ten_shot=False):

    if ten_shot:
        examples = utils.get_examples(num_examples=1)
        example_template = get_example_template(examples)
        version = 1
    else:
        example_template = ""
        version = 0

    for (persona, description) in persona_list:

        if description and description != "":
            version = 2
            persona += ', ' + description

        if persona:
            name = f'../../prompts/cnn_dm_v{version}/{persona}_v{version}.jsonl'
        else:
            name = f'../../prompts/cnn_dm_v{version}/nopersona_v{version}.jsonl'

        with open(name, 'w') as writer:
            for filename in glob.glob(os.path.join(folderpath, '**/*.txt'), recursive=True):
                output_line = generate_instruction(filename, persona, example_template)
                json_line = ujson.dumps(output_line)
                writer.write(f'{json_line}\n')


if __name__ == '__main__':

    folder = '/Users/giovanni/Desktop/r3s_cnn_dm/test'

    personas = [
        (None, None),
        ('Virtual Assistant', 'tasked with providing concise summaries of articles'),
        ('Human', 'tasked with summarizing articles'),
        ('Professor', 'specializing in literature and critical analysis. You have a keen eye for detail and a deep understanding of various literary forms and techniques. You excel in synthesizing complex information into concise summaries'),
        ('Teacher', 'tasked with guiding students in understanding complex topics and improving their comprehension skills'),
        ('Student', 'tasked with summarizing the article provided below'),
        ('Contestant', 'tasked with summarizing the following article accurately and succinctly'),
        ('Writer', 'tasked with summarizing articles accurately and concisely'),
        ('Journalist', 'specializing in technology and innovation. You have a keen eye for detail and a knack for distilling complex information into concise and engaging summaries')
    ]

    main(folder, personas, False)