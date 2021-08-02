import json
import pickle
import re

with open('./data/chase/dev_gold.txt', 'r') as f:
    glist = []
    gseq_one = []
    for l in f.readlines():
        if len(l.strip()) == 0:
            glist.append(gseq_one)
            gseq_one = []
        else:
            lstrip = l.strip().split('\t')
            gseq_one.append(lstrip)
with open('./logs_chase_editsql/valid_use_gold_queries_predictions.json.eval', 'r') as f:
    res = f.readlines()
with open('./data/chase_data_removefrom/dev.pkl', 'rb') as f:
    dev = pickle.load(f)
data = []
databases = []
for g_seq in glist:
    if g_seq:
        if g_seq[0][1] not in databases:
            databases.append(g_seq[0][1])
rearranged = []
for database in databases:
    for d in dev:
        if d['database_id'] == database:
            rearranged.append(d)
pos = 0
for interaction_id, g_seq in enumerate(glist):
    assert len(g_seq) == len(rearranged[interaction_id]['interaction'])
    for turn_id, g in enumerate(g_seq):
        data.append({'interaction_id': interaction_id + 1, 'turn_id': turn_id + 1,
                     'question': rearranged[interaction_id]['interaction'][turn_id]['utterance'],
                     'ground_truth': g[0]})
        for i in range(5):
            if res[pos + i] == '\n':
                break
        pos += i + 1
        assert re.findall(r'gold: (.*)\n', res[pos - 3])[0] == g[0]
        data[-1]['sql'] = re.findall(r'pred: (.*)\n', res[pos - 4])[0]
        data[-1]['exact'] = True if re.findall(r'exact: (.*)\n', res[pos - 2])[0] == 'True' else 0
print(len(data))
print(len([d for d in data if d['exact']])/len(data))
with open('./igsql_chase.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
