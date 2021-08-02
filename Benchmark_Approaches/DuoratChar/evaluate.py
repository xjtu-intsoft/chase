import collections
import json
import _jsonnet

from duorat import datasets
from duorat.utils import registry

config = json.loads(_jsonnet.evaluate_file("configs/duorat/duorat-good-no-bert.jsonnet"))
data = registry.construct("dataset", config["data"]["val"])
with open('./inference_output.eval', 'r') as f:
    eval = json.load(f)

res = list()
for example, sql in zip(data.examples, eval['per_item']):
    example = example.orig
    item = {'interaction_id': example['interaction_id'], 'turn_id': example['turn_id'], 'question': example['question'],
            'groundtruth': sql['gold'], 'sql': sql['predicted'], 'exact': sql['exact']}
    res.append(item)
res = sorted(res, key=lambda x: x['interaction_id'])
with open('./duorat_sparc.json', 'w') as f:
    json.dump(res, f, indent=4)

interaction = collections.defaultdict(list)
for r in res:
    interaction['ex_' + str(r['interaction_id'])].append(r)
exact_interaction = []
for ex in interaction.values():
    exact = True
    for turn in ex:
        if not turn['exact']:
            exact = False
            break
    if exact:
        exact_interaction.append(ex)
eval['total_scores']['joint_all'] = {}
eval['total_scores']['joint_all']['count'] = len(interaction)
eval['total_scores']['joint_all']['exact'] = len(exact_interaction) / len(interaction)
turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn >4']
levels = ['easy', 'medium', 'hard', 'extra', 'all', "joint_all"]
partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                 'group', 'order', 'and/or', 'IUEN', 'keywords']
f = open('./eval_duorat_sparc.txt', 'w')
f.write("{:20} {:20} {:20} {:20} {:20} {:20} {:20}\n".format("", *levels))
counts = [eval['total_scores'][level]['count'] for level in levels]
f.write("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}\n\n".format("count", *counts))
f.write("====================== EXACT MATCHING ACCURACY =====================\n")
exact_scores = [eval['total_scores'][level]['exact'] for level in levels]
f.write("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n\n".format("exact match", *exact_scores))
eval['total_scores']['joint_all']['partial'] = collections.defaultdict(dict)
for type_ in partial_types:
    eval['total_scores']['joint_all']['partial'][type_]['acc'] = 0
    eval['total_scores']['joint_all']['partial'][type_]['rec'] = 0
    eval['total_scores']['joint_all']['partial'][type_]['f1'] = 1
f.write("---------------------PARTIAL MATCHING ACCURACY----------------------\n")
for type_ in partial_types:
    this_scores = [eval['total_scores'][level]['partial'][type_]['acc'] for level in levels]
    f.write("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n".format(type_, *this_scores))
f.write("---------------------- PARTIAL MATCHING RECALL ----------------------\n")
for type_ in partial_types:
    this_scores = [eval['total_scores'][level]['partial'][type_]['rec'] for level in levels]
    f.write("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n".format(type_, *this_scores))
f.write("---------------------- PARTIAL MATCHING F1 --------------------------\n")
for type_ in partial_types:
    this_scores = [eval['total_scores'][level]['partial'][type_]['f1'] for level in levels]
    f.write("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n".format(type_, *this_scores))
f.write("\n\n{:20} {:20} {:20} {:20} {:20} {:20}\n".format("", *turns))
for i, turn in enumerate(turns):
    eval['total_scores'][turn] = {'count': 0, 'exact_count': 0}
    for ex in interaction.values():
        if len(ex) > i:
            eval['total_scores'][turn]['count'] += 1
            if ex[i]['exact']:
                assert i + 1 == ex[i]['turn_id']
                eval['total_scores'][turn]['exact_count'] += 1
    eval['total_scores'][turn]['exact'] = eval['total_scores'][turn]['exact_count'] / eval['total_scores'][turn]['count']
counts = [eval['total_scores'][turn]['count'] for turn in turns]
f.write("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}\n\n".format("count", *counts))
f.write("====================== TURN EXACT MATCHING ACCURACY =====================\n")
exact_scores = [eval['total_scores'][turn]['exact'] for turn in turns]
f.write("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n".format("exact match", *exact_scores))
f.close()
