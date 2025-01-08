import json
from .openai_api import openai_completion
from tqdm import tqdm
import random

categories = ['object', 'human', 'animal', 'food', 'activity', 'attribute', 'counting', 'color', 'material', 'spatial', 'location', 'shape', 'other']

prompt = """
Given a image descriptions, generate one or two multiple-choice questions that verifies if the image description is correct.
Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other), and then generate a question for each type.

Description: A man posing for a selfie in a jacket and bow tie.
Entities: man, selfie, jacket, bow tie
Activities: posing
Colors:
Counting:
Other attributes:
Questions and answers are below:
About man (human):
Q: is this a man?
Choices: yes, no
A: yes
Q: who is posing for a selfie?
Choices: man, woman, boy, girl
A: man
About selfie (activity):
Q: is the man taking a selfie?
Choices: yes, no
A: yes
Q: what type of photo is the person taking?
Choices: selfie, landscape, sports, portrait
A: selfie
About jacket (object):
Q: is the man wearing a jacket?
Choices: yes, no
A: yes
Q: what is the man wearing?
Choices:jacket, t-shirt, tuxedo, swearter
A: jacket
About bow tie (object):
Q: is the man wearing a bow tie?
Choices: yes, no
A: yes
Q: is the man wearing a bow tie or a neck tie?
Choices: bow tie, neck tie, cravat, bolo tie
A: bow tie
About posing (activity):
Q: is the man posing for the selfie?
Choices: yes, no
A: yes
Q: what is the man doing besides taking the selfie?
Choices: posing, waving, nothing, shaking
A: posing

Description: """

def parse_resp(resp):

    resp = resp.split('\n')
    print("Here1, ", resp)
    
    question_instances = []
    
    this_entity = None
    this_type = None
    this_question = None
    this_choices = None
    this_answer = None
    
    for line_number in range(6, len(resp)):
        line = resp[line_number]
        print(line)
        if line.startswith('About '):
            whole_line = line[len('About '):-1]
            this_entity = whole_line.split(' (')[0]
            this_type = whole_line.split(' (')[1].split(')')[0]
            
        elif line.startswith('Q: '):
            this_question = line[3:]
        elif line.startswith('Choices: '):
            this_choices = line[9:].split(', ')
        elif line.startswith('A: '):
            this_answer = line[3:]
            
            if this_entity and this_question and this_choices:
                question_instances.append((this_entity, this_question, this_choices, this_answer, this_type))
            this_question = None
            this_choices = None
            this_answer = None
            
    return question_instances


## Generate questions for a caption with GPT-3

def get_question_and_answers(caption):
    this_prompt = prompt + caption+ "\nEntities:"
    resp = openai_completion(this_prompt)
    
    with open('resp.json', 'w') as f:
        json.dump(resp, f)

    print()
    print(resp)
    print()
    
    question_instances = parse_resp(resp)

    print(question_instances)
    # exit()
    
    this_caption_qas = []
    
    for question_instance in question_instances:
        this_qa = {}
        this_qa['caption'] = caption
        this_qa['element'] = question_instance[0]
        this_qa['question'] = question_instance[1]
        this_qa['choices'] = question_instance[2]
        this_qa['answer'] = question_instance[3]
        this_qa['element_type'] = question_instance[4]
        
        if question_instance[4] not in categories:
            continue
            
        if this_qa['element_type'] in ['animal', 'human']:
            this_qa['element_type'] = 'animal/human'
            
        this_caption_qas.append(this_qa)
        
    return this_caption_qas
