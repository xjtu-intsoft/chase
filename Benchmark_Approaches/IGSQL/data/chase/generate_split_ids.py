import json


def generate_database_ids(split):
    with open('./' + split + '.json') as f:
        data = json.load(f)
    database_id = set()
    for interaction in data:
        database_id.add(interaction['database_id'])
    with open('./' + split + '_db_ids.txt', 'w') as f:
        f.write('\n'.join(database_id))


for split in ['train', 'dev']:
    generate_database_ids(split)
