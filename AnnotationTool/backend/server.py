# coding=utf8

import os
import json
import copy
import pathlib
import tempfile
from typing import Dict, List, Union
from pprint import pprint
from datetime import datetime
from overrides import overrides
from flask import Flask, Response, request, render_template, redirect, url_for, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from bson import json_util
from bson.objectid import ObjectId
from pymongo import MongoClient
from sql_units import get_sql_entity, get_db_entity_by_id, get_db_entity, get_db_entity_graph
from encryption import make_md5
from sql_parser import parse_sql, execute_sql


app = Flask(__name__,
            static_url_path='',
            template_folder=os.path.join("..", "annotate", "dist"), 
            static_folder=os.path.join("..", "annotate", "dist"))
app.secret_key = "AnnotationSecretKey"
# Login
login_manager = LoginManager()
login_manager.init_app(app)

# MongoDB
client = MongoClient()
db = client.contextual_semparse
user_collection = db.users
collection = db.raw_data
annotation_collection = db.annotations
conversation_collection = db.conversations
db_collection = db.databases
translated_db_collection = db.translated_databases
refine_sparc_collection = db.refine_sparc


# SparC & DuSQL Database
SPARC_DB_PATH = os.path.join("data", "database")
TRANSLATED_SPARC_DB_PATH = os.path.join("data", "translated_database")
DUSQL_DB_PATH = os.path.join("data", "DuSQL", "database")


def get_db_path_by_id(database_id: str, source: str, is_translated: bool) -> str:
    if source == 'sparc':
        if is_translated:
            return os.path.join(TRANSLATED_SPARC_DB_PATH, database_id, database_id + ".sqlite")
        else:
            return os.path.join(SPARC_DB_PATH, database_id, database_id + ".sqlite")
    else:
        return os.path.join(DUSQL_DB_PATH, database_id + ".sqlite")


class User(UserMixin):

    def __init__(self, user_id: str, username: str, permission: list):
        self._user_id = user_id
        self._username = username
        self._permission = permission

    @property
    def username(self):
        return self._username

    @property
    def permission(self):
        return self._permission
    
    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    @overrides
    def get_id(self):
        return self._user_id

    @classmethod
    def get_user_by_id(cls, user_id: str):
        object_id = ObjectId(user_id)
        query_key = {'_id': ObjectId(object_id)}
        doc = user_collection.find_one(query_key)
        if doc is None:
            return None
        return cls(user_id=user_id, username=doc['username'], permission=doc['permission'])


@login_manager.user_loader
def load_user(uid: str) -> Union[None, User]:
    return User.get_user_by_id(uid)


@app.route('/api/login', methods=['POST'])
def login_post() -> str:
    payload = request.json
    username = payload['username']
    password = make_md5(payload['password']) # md5
    doc = user_collection.find_one({"username": username, "password": password})
    print({"username": username, "password": password})
    if doc is not None:
        user = User(user_id=str(doc['_id']), username=doc['username'], permission=doc['permission'])
        login_user(user)
        return_message = {"status": "success"}
    else:
        return_message = {"status": "fail"}
    return json.dumps(return_message, default=json_util.default)


@app.route('/api/logout', methods=['GET'])
@login_required
def logout() -> str:
    logout_user()
    return redirect(url_for('root'))


def formate_unannotated_example(doc) -> Dict:
    # Query database
    database = db_collection.find_one({"database_id": doc['database_id']})
    database_entities = get_db_entity(database)
    formatted_result = {
        "databaseId": doc['database_id'],
        'split': doc['split'],
        'exampleId': doc['example_id'],
        'problematic': False,
        'problematicNote': ""
    }
    formatted_result.update({
        "database": database_entities
    })
    formatted_interactions = list()
    for turn in doc['interaction']:
        processed_turn = {
            'questionId': turn['question_id'],
            'question': turn['english_question'],
            'googleTranslation': turn['google_chinese_question'],
            'baiduTranslation': turn['baidu_chinese_question'],
            'sql': turn['sql']
        }
        sql_entity_ids = get_sql_entity(turn['sql_dict'])
        sql_entities = get_db_entity_by_id(database, sql_entity_ids)
        processed_turn.update(sql_entities)
        formatted_interactions.append(processed_turn)
    formatted_result['interaction'] = formatted_interactions
    return formatted_result


@app.route('/api/meta', methods=['GET'])
def get_meta_info() -> str:
    remaining_conversation = collection.count_documents({"is_annotated": False})
    new_conversation = conversation_collection.count_documents({})
    remaining_database = db_collection.count({"source": "sparc"}) - translated_db_collection.count_documents({})
    remaining_refine_sparc = refine_sparc_collection.count({"revised_by": {"$exists": False}})
    result = {
        "remainingConversation": remaining_conversation,
        "newConversation": new_conversation,
        "remainingDatabase": remaining_database,
        "remainingSparctoRefine": remaining_refine_sparc
    }
    return json.dumps(result, default=json_util.default)


@app.route('/api/annotation_list', methods=['GET'])
@login_required
def get_annotation_list() -> str:
    docs = annotation_collection.find({})
    total = 0
    results = {"conversations": list()}
    user_count = dict()
    if docs is not None:
        for doc in list(docs):
            total += 1
            results['conversations'].append({
                'problematic': False if 'is_problematic' not in doc else doc['is_problematic'],
                'problematicNote': "" if "problematic_note" not in doc else doc['problematic_note'],
                "conversationId": doc['example_id'],
                "isAnnotated": True,
                "databaseId": doc['database_id'],
                "createdBy": "" if "created_by" not in doc else doc['created_by'],
                "createDate": "" if "create_date" not in doc else doc['create_date'],
                "revisedBy": "" if "revised_by" not in doc else doc['revised_by'],
                "reviseDate": "" if "revise_date" not in doc else doc['revise_date'],
            })
            if "created_by" in doc and len(doc["created_by"]) > 0:
                un = doc["created_by"]
                if un not in user_count:
                    user_count[un] = 0
                user_count[un] += 1
    results['total'] = total
    results['users'] = [{"username": un, "total": c} for (un, c) in user_count.items()]
    results['users'].sort(key=lambda x: x['total'], reverse=True)
    return json.dumps(results, default=json_util.default)


@app.route('/api/annotate/<example_id>', methods=['GET'])
@login_required
def get_annotation_by_example_id(example_id: str) -> str:
    doc = collection.find_one({"example_id": example_id})
    if doc is None:
        return json.dumps({}, default=json_util.default)
    # Query database
    database = db_collection.find_one({"database_id": doc['database_id']})
    database_entities = get_db_entity(database)
    formatted_result = {
        "databaseId": doc['database_id'],
        'split': doc['split'],
        'exampleId': doc['example_id'],
        'isAnnotated': doc['is_annotated']
    }
    formatted_result.update({
        "database": database_entities
    })

    formatted_interactions = list()
    if doc['is_annotated']:
        annotation = annotation_collection.find_one({"example_id": example_id})
        assert annotation is not None
        formatted_result.update({
            "objectId": str(annotation['_id']),
            'problematic': False if 'is_problematic' not in annotation else annotation['is_problematic'],
            'problematicNote': "" if 'problematic_note' not in annotation else annotation['problematic_note']
        })
        for turn, annotation_turn in zip(doc['interaction'], annotation['interaction']):
            processed_turn = {
                'questionId': turn['question_id'],
                'question': turn['english_question'],
                'googleTranslation': turn['google_chinese_question'],
                'baiduTranslation': turn['baidu_chinese_question'],
                'sql': turn['sql'],
                'annotatedQuestion': annotation_turn['annoated_chinese_question'],
                'tokenizedAnnotatedQuestion': annotation_turn['tokenized_annotated_chinese_question'],
                'linking': annotation_turn['schema_linking'],
                'contextualPhenomena': annotation_turn['contextual_phenomena']
            }
            sql_entity_ids = get_sql_entity(turn['sql_dict'])
            sql_entities = get_db_entity_by_id(database, sql_entity_ids)
            processed_turn.update(sql_entities)
            formatted_interactions.append(processed_turn)
    else:
        # Turn
        formatted_result.update({
            'problematic': False,
            'problematicNote': ""
        })
        formatted_interactions = list()
        for turn in doc['interaction']:
            processed_turn = {
                'questionId': turn['question_id'],
                'question': turn['english_question'],
                'googleTranslation': turn['google_chinese_question'],
                'baiduTranslation': turn['baidu_chinese_question'],
                'sql': turn['sql']
            }
            sql_entity_ids = get_sql_entity(turn['sql_dict'])
            sql_entities = get_db_entity_by_id(database, sql_entity_ids)
            processed_turn.update(sql_entities)
            formatted_interactions.append(processed_turn)
    formatted_result['interaction'] = formatted_interactions

    return json.dumps(formatted_result, default=json_util.default)


@app.route('/api/annotate/db/<db_id>', methods=['GET'])
@login_required
def get_annotation_by_db_id(db_id) -> str:
    doc = collection.aggregate([
        {"$match": {"is_annotated": False, "database_id": db_id}},
        {"$sample": {"size": 1}}
    ])
    doc = list(doc)
    if doc is None or len(doc) == 0:
        return json.dumps({}, default=json_util.default)
    else:
        formatted_result = formate_unannotated_example(doc[0])
        return json.dumps(formatted_result, default=json_util.default)


@app.route('/api/annotate', methods=['GET'])
@login_required
def get_annotation() -> str:
    doc = collection.aggregate([
        {"$match": {"is_annotated": False}},
        {"$sample": {"size": 1}}
    ])
    doc = list(doc)
    if doc is None or len(doc) == 0:
        return json.dumps({}, default=json_util.default)
    else:
        formatted_result = formate_unannotated_example(doc[0])
        return json.dumps(formatted_result, default=json_util.default)


@app.route('/api/db/<db_id>', methods=['GET'])
@login_required
def get_database_by_id(db_id) -> str:
    database = db_collection.find_one({"database_id": db_id})
    database_entities = get_db_entity(database)
    db_graph = get_db_entity_graph(database)
    result = {
        "entities": database_entities,
        "graph": db_graph
    }
    return json.dumps(result, default=json_util.default)


@app.route('/api/db_list/', methods=['GET'])
@login_required
def get_database_list() -> str:
    user = current_user
    results = dict()
    for db_doc in db_collection.find({"labelConversation": True}, {"database_id": 1}):
        db_id = db_doc['database_id']
        assert db_id not in results
        results[db_id] = {
            "count": 0,
            "user_count": 0
        }
    # Count total
    for record in conversation_collection.aggregate([{"$group":{"_id":"$database_id", "count":{"$sum":1}}}]):
        db_id = record['_id']
        if db_id not in results:
            continue
        results[db_id]["count"] = record['count']
    # Count for each user
    for record in conversation_collection.aggregate([{"$match": {"created_by": user.username}}, {"$group":{"_id":"$database_id", "count":{"$sum":1}}}]):
        db_id = record['_id']
        if db_id not in results:
            continue
        results[db_id]["user_count"] = record['count']
    results = sorted([{"databaseId": db_id, "conversationNum": value["count"], "userConversationNum": value["user_count"]} for db_id, value in results.items()], 
                     key=lambda x: x['userConversationNum'], reverse=True)
    return json.dumps({"results": results, "username": user.username}, default=json_util.default)


@app.route('/api/sql/', methods=['POST'])
@login_required
def parse_sql_entities() -> None:
    payload = request.json
    database_id = payload['databaseId']
    sql = payload['sql']
    database = db_collection.find_one({"database_id": database_id})
    translated_database = translated_db_collection.find_one({"database_id": database_id})
    if database['source'] == 'sparc' and translated_database is not None:
        use_translated = True
    else:
        use_translated = False
    if database is None:
        return json.dumps({"msg": "Database does not exist %s." % database_id}, default=json_util.default)
    else:
        try:
            db_path = get_db_path_by_id(database_id, database['source'], use_translated)
            execution_results = execute_sql(db_path, sql)
        except Exception as e:
            error = str(e)
            return json.dumps({"msg": "Execution error: %s." % error}, default=json_util.default)
        else:
            try:
                if use_translated:
                    sql_dict = parse_sql(sql, translated_database)
                else:
                    sql_dict = parse_sql(sql, database)
            except Exception as e:
                print(e)
                return json.dumps({"msg": "Parsing error: %s." % str(e)}, default=json_util.default)
            else:
                results = {"msg": "success",}
                sql_entity_ids = get_sql_entity(sql_dict)
                if use_translated:
                    sql_entities = get_db_entity_by_id(translated_database, sql_entity_ids)
                else:
                    sql_entities = get_db_entity_by_id(database, sql_entity_ids)
                results.update(sql_entities)
                results.update(execution_results)
                return json.dumps(results, default=json_util.default)


@app.route('/api/annotate', methods=['POST'])
@login_required
def submit_annotation() -> str:
    user = current_user
    if "allowSubmit" not in user.permission:
        return Response("{'msg': 'Permission denied'}", status=403, mimetype='applicatioin/json')
    payload = request.json
    example_id = payload['conversationId']
    database_id = payload['databaseId']
    split = payload['split']

    object_id = payload['objectId'] if "objectId" in payload else None
    doc = collection.find_one({"example_id": example_id})
    result = {
        "example_id": example_id,
        "database_id": database_id,
        "split": split,
        "is_problematic": payload['problematic'],
        "problematic_note": '' if 'problematicNote' not in payload else payload['problematicNote']
    }
    if doc is not None:
        # Save it
        save_result = copy.deepcopy(result)
        processed_interactions = list()
        for turn in payload['annotation']:
            processed_turn = {
                "question_id": turn['questionId'],
                "english_question": turn['englishQuestion'],
                'google_chinese_question': turn['googleTranslation'],
                'baidu_chinese_question': turn['baiduTranslation'],
                'sql': turn['sql'],
                'annoated_chinese_question': turn['annotatedQuestion'],
                'tokenized_annotated_chinese_question': turn['tokenizedAnnotatedQuestion'],
                'schema_linking': turn['linking'],
                'contextual_phenomena': turn['contextualPhenomena']
            }
            processed_interactions.append(processed_turn)
        save_result.update({
            "interaction": processed_interactions
        })
        now = datetime.now()
        if object_id is None:
            # insert
            save_result.update({
                "created_by": user.username,
                "create_date": now
            })
            annotation_collection.insert_one(save_result)
        else:
            # update
            query_key = {'_id': ObjectId(object_id)}
            original = annotation_collection.find_one(query_key)
            if user.username != original.get('created_by', '') and "allowRevise" not in user.permission:
                return Response("{'msg': 'Permission denied'}", status=403, mimetype='applicatioin/json')
            save_result.update({
                "revised_by": user.username,
                "revise_date": now,
                "created_by": "" if "created_by" not in original else original['created_by'],
                "create_date": "" if "create_date" not in original else original['create_date'],
            })
            annotation_collection.update(query_key, save_result)

        result.update({
            "status": "AnnotaionSuccess"
        })

        # Mark the example as annotated
        doc['is_annotated'] = True
        collection.save(doc)
    else:
        result.update({
            "status": "AnnotaionFail"
        })
    return json.dumps(result, default=json_util.default)


@app.route('/api/conversation/', methods=['POST'])
@login_required
def submit_conversation() -> None:
    """
    insert or update conversation
    """
    user = current_user
    if "allowSubmit" not in user.permission:
        return Response("{'msg': 'Permission denied'}", status=403, mimetype='applicatioin/json')
    payload = request.json
    database_id = payload['databaseId']
    result = {
        "database_id": database_id,
        "dataset": "dusql",
    }
    processed_interactions = list()
    for turn in payload['annotation']:
        processed_turn = {
            'question_id': turn['questionId'],
            'question': turn['question'],
            'sql': turn['sql'],
            'tokenized_question': turn['tokenizedQuestion'],
            'schema_linking': turn['linking'],
            'contextual_phenomena': turn['contextualPhenomena'],
            'question_style': turn['questionStyle'],
            'pragmatics': turn['pragmatics'],
            'intent': turn['intent']
        }
        processed_interactions.append(processed_turn)
    result['interaction'] = processed_interactions

    now = datetime.now()
    object_id = payload['conversationId'] if "conversationId" in payload else None
    if object_id is None:
        # insert
        result.update({
            "created_by": user.username,
            "create_date": now
        })
        conversation_collection.insert_one(result)
    else:
        # update
        print("Update")
        query_key = {'_id': ObjectId(object_id)}
        original = conversation_collection.find_one(query_key)
        if user.username != original.get('created_by', '') and "allowRevise" not in user.permission:
            return Response("{'msg': 'Permission denied'}", status=403, mimetype='applicatioin/json')
        result.update({
            "revised_by": user.username,
            "revise_date": now,
            "created_by": "" if "created_by" not in original else original['created_by'],
            "create_date": "" if "create_date" not in original else original['create_date'],
        })
        conversation_collection.update(query_key, result)
    return json.dumps({"msg": "success"}, default=json_util.default)


@app.route('/api/new_annotation_list', methods=['GET'])
@login_required
def get_new_annotation_list() -> str:
    docs = conversation_collection.find({})
    total = 0
    results = {"conversations": list()}
    user_count = dict()
    if docs is not None:
        for doc in list(docs):
            total += 1
            results['conversations'].append({
                "conversationId": str(doc['_id']),
                "databaseId": doc['database_id'],
                "createdBy": "" if "created_by" not in doc else doc['created_by'],
                "createDate": "" if "create_date" not in doc else doc['create_date'],
                "revisedBy": "" if "revised_by" not in doc else doc['revised_by'],
                "reviseDate": "" if "revise_date" not in doc else doc['revise_date'],
            })
            if "created_by" in doc and len(doc["created_by"]) > 0:
                un = doc["created_by"]
                if un not in user_count:
                    user_count[un] = 0
                user_count[un] += 1
    results['total'] = total
    results['users'] = [{"username": un, "total": c} for (un, c) in user_count.items()]
    results['users'].sort(key=lambda x: x['total'], reverse=True)
    return json.dumps(results, default=json_util.default)


@app.route('/api/conversation/<conversation_id>', methods=['GET'])
@login_required
def get_conversation_by_id(conversation_id: str) -> str:
    query_key = {'_id': ObjectId(conversation_id)}
    conversation = conversation_collection.find_one(query_key)
    
    if conversation is None:
        return Response("{'msg': 'Conversation not Found'}", status=404, mimetype='applicatioin/json')
    else:
        result = {
            "databaseId": conversation['database_id'],
            "conversationId": conversation_id,
        }
        database = db_collection.find_one({"database_id": conversation['database_id']})
        processed_interactions = list()
        for turn in conversation['interaction']:
            processed_turn = {
                'questionId': turn['question_id'],
                'question': turn['question'],
                'sql': turn['sql'],
                'tokenizedQuestion': turn['tokenized_question'],
                'linking': turn['schema_linking'],
                'contextualPhenomena': turn['contextual_phenomena'],
                'questionStyle': turn['question_style'],
                'pragmatics': turn['pragmatics'],
                'intent': turn['intent']
            }
            try:
                sql_dict = parse_sql(processed_turn['sql'], database)
            except Exception as e:
                print(e)
            else:
                sql_entity_ids = get_sql_entity(sql_dict)
                sql_entities = get_db_entity_by_id(database, sql_entity_ids)
                processed_turn.update(sql_entities)
            processed_interactions.append(processed_turn)
        result['conversations'] = processed_interactions
        return json.dumps(result, default=json_util.default)


"""
Translate SparC Database
"""
@app.route('/api/translate/database/', methods=['GET'])
@login_required
def get_sparc_database_list():
    results = list()
    db_ids = set()
    for doc in translated_db_collection.find():
        results.append({
            "databaseId": doc['database_id'],
            "translator": doc['translator']
        })
        db_ids.add(doc['database_id'])
    for doc in db_collection.find({"source": "sparc"}):
        if doc['database_id'] in db_ids:
            continue
        else:
            results.append({
                "databaseId": doc['database_id'],
                "translator": ""
            })
    return json.dumps(results, default=json_util.default)


@app.route('/api/translate/database/', methods=['POST'])
@login_required
def submit_database_translation():
    user = current_user
    if "allowSubmit" not in user.permission:
        return Response("{'msg': 'Permission denied'}", status=403, mimetype='applicatioin/json')
    payload = request.json
    db_id = payload['databaseId']
    doc = translated_db_collection.find_one({"database_id": db_id, "source": "sparc"})
    db_doc = db_collection.find_one({"database_id": db_id, "source": "sparc"})

    save_result = {
        "database_id": db_id, "source": db_doc["source"],
        "column_types": db_doc["column_types"],
        "foreign_keys": db_doc["foreign_keys"],
        "primary_keys": db_doc["primary_keys"],
    }
    table_names, column_names = list(), [(-1, "*")]
    raw_columns = list()
    translation_results = payload['tables']
    for tid, table in enumerate(translation_results):
        table_names.append(table['translation'])
        assert tid == table["tableId"]
        for column in table['columns']:
            raw_columns.append({
                "column_id": column['columnId'],
                "column_name": column['translation'],
                "table_id": table["tableId"]
            })
    raw_columns.sort(key=lambda x: x['column_id'])
    column_names += [[c['table_id'], c['column_name']] for c in raw_columns]

    save_result.update({
        "column_names": column_names,
        "column_names_original": column_names,
        "table_names": table_names,
        "table_names_original": table_names
    })
    assert len(column_names) == len(db_doc['column_names'])
    assert len(table_names) == len(db_doc['table_names'])

    now = datetime.now()
    if doc is None:
        # insert
        save_result.update({
            "translator": user.username,
            "translate_date": now,
        })
        translated_db_collection.insert_one(save_result)
    else:
        # update
        save_result.update({
            "translator": doc['translator'],
            "translate_date": now,
        })
        translated_db_collection.update({"database_id": db_id, "source": "sparc"}, save_result)

    return json.dumps({"msg": "success"}, default=json_util.default)


def find_translation(database):
    db_id = database['database_id']
    db_annotations = annotation_collection.find({"database_id": db_id})
    table_map, column_map = dict(), dict()
    for doc in db_annotations:
        example_id = doc['example_id']
        interaction = doc['interaction']
        for turn in interaction:
            tokens = turn['tokenized_annotated_chinese_question']
            linkings = turn['schema_linking']
            for link in linkings:
                beg, end = link['beg'], link['end']
                link_type = link['type']
                if link_type == 'column':
                    column_id = link['entity']['columnId']
                    key = "".join(tokens[beg:end+1])
                    if column_id not in column_map:
                        column_map[column_id] = dict()
                    if key not in column_map[column_id]:
                        column_map[column_id][key] = set()
                    column_map[column_id][key].add(example_id)
                elif link_type == 'table':
                    table_id = link['entity']['tableId']
                    key = "".join(tokens[beg:end+1])
                    if table_id not in table_map:
                        table_map[table_id] = dict()
                    if key not in table_map[table_id]:
                        table_map[table_id][key] = set()
                    table_map[table_id][key].add(example_id)
    return table_map, column_map


@app.route('/api/translate/database/<db_id>', methods=['GET'])
@login_required
def get_sparc_database_by_id(db_id: str):
    doc = translated_db_collection.find_one({"database_id": db_id})
    result = {"is_translated": doc is not None}

    db_doc = db_collection.find_one({"source": "sparc", "database_id": db_id})
    assert db_doc is not None
    table_map, column_map = find_translation(db_doc)

    tables = list()
    table_names_original, table_names = db_doc['table_names_original'], db_doc['table_names']
    for tid, (tno, tn,) in enumerate(zip(table_names_original, table_names)):
        tables.append({
            "tableId": tid,
            "tableName": tn,
            "tableNameOriginal": tno,
            "columns": list(),
            "translation": "",
            "candidates": dict()
        })
    for tid, translations in table_map.items():
        for t in tables:
            if t['tableId'] == tid:
                t['candidates'].update(
                    {key: list(values) for key, values in translations.items()}
                )
                break
    column_types = db_doc['column_types']
    column_names_original, column_names = db_doc['column_names_original'], db_doc['column_names']
    for cid, ((tid, cno), (_, cn,), ct,) in enumerate(zip(column_names_original, column_names, column_types)):
        if cid == 0:
            # Skip *
            continue
        target_table = tables[tid]
        target_table['columns'].append({
            "columnId": cid,
            "columnType": ct,
            "columnName": cn,
            "columnNameOriginal": cno,
            "candidates": dict() if cid not in column_map else {key: list(values) for key, values in column_map[cid].items()},
            "translation": ""
        })

    # Foreign Keys, Primary Keys
    foreign_keys, primary_keys = list(), list()
    for pk_col_id in db_doc['primary_keys']:
        pk_tid, pk_column_name = column_names_original[pk_col_id]
        primary_keys.append("%s.%s" % (tables[pk_tid]["tableName"], pk_column_name))

    for pk, fk in db_doc['foreign_keys']:
        pk_tid, pk_column_name = column_names_original[pk]
        fk_tid, fk_column_name = column_names_original[fk]
        foreign_keys.append("%s.%s = %s.%s" % (tables[pk_tid]["tableName"], pk_column_name,
                                               tables[fk_tid]["tableName"], fk_column_name))

    if doc is not None:
        # Translated
        for tid, translated_tn in enumerate(doc['table_names']):
            tables[tid]["translation"] = translated_tn
        for cid, (tid, translated_cn,) in enumerate(doc['column_names']):
            target_table = tables[tid]
            for column in target_table['columns']:
                if column['columnId'] == cid:
                    column['translation'] = translated_cn
    result.update({
        "tables": tables,
        "foreignKeys": foreign_keys,
        "primaryKeys": primary_keys
    })
    return json.dumps(result, default=json_util.default)


"""
Refine SparC Database
"""
@app.route('/api/translate/database/<db_id>/conversations/', methods=['GET'])
@login_required
def get_translate_sparc_conversations_by_database_id(db_id: str):
    conversations = refine_sparc_collection.find({"database_id": db_id})
    results = list()
    for doc in conversations:
        results.append({
            "conversationId": doc['example_id'],
            'problematic': False if 'is_problematic' not in doc else doc['is_problematic'],
            'problematicNote': "" if "problematic_note" not in doc else doc['problematic_note'],
            "createdBy": "" if "created_by" not in doc else doc['created_by'],
            "createDate": "" if "create_date" not in doc else doc['create_date'],
            "revisedBy": "" if "revised_by" not in doc else doc['revised_by'],
            "reviseDate": "" if "revise_date" not in doc else doc['revise_date'],
            "isDeleted": doc['is_delete']
        })
    results = {
        "conversations": results,
        "total": len(results)
    }
    return json.dumps(results, default=json_util.default)


@app.route('/api/translated/database/<db_id>', methods=['GET'])
@login_required
def get_translated_database_by_id(db_id: str) -> str:
    database = translated_db_collection.find_one({"database_id": db_id})
    database_entities = get_db_entity(database)
    db_graph = get_db_entity_graph(database)
    result = {
        "entities": database_entities,
        "graph": db_graph
    }
    return json.dumps(result, default=json_util.default)


@app.route('/api/translate/conversation/<conversation_id>', methods=['GET'])
@login_required
def get_translate_conversation_by_id(conversation_id: str) -> str:
    query_key = {'example_id': conversation_id}
    conversation = refine_sparc_collection.find_one(query_key)

    if conversation is None:
        return Response("{'msg': 'Conversation not Found'}", status=404, mimetype='applicatioin/json')
    else:
        result = {
            "databaseId": conversation['database_id'],
            "conversationId": conversation_id,
            "problematic": conversation['is_problematic'],
            "problematicNote": conversation['problematic_note'],
        }
        database = db_collection.find_one({"database_id": conversation['database_id']})
        translated_database = translated_db_collection.find_one({"database_id": conversation['database_id']})
        processed_interactions = list()
        for turn in conversation['interaction']:
            processed_turn = {
                'questionId': turn['question_id'],
                'question': turn['question'],
                'sql': turn['sql'],
                'tokenizedQuestion': turn['tokenized_question'],
                'linking': turn['schema_linking'],
                'contextualPhenomena': turn['contextual_phenomena'],
            }
            try:
                sql_dict = parse_sql(processed_turn['sql'], translated_database)
            except Exception as e:
                print(e)
            else:
                sql_entity_ids = get_sql_entity(sql_dict)
                sql_entities = get_db_entity_by_id(translated_database, sql_entity_ids)
                processed_turn.update(sql_entities)
            processed_interactions.append(processed_turn)
        result['conversations'] = processed_interactions
        return json.dumps(result, default=json_util.default)


@app.route('/api/translate/conversation/<conversation_id>', methods=['DELETE'])
@login_required
def delete_translate_conversation_by_id(conversation_id: str) -> str:
    query_key = {'example_id': conversation_id}
    conversation = refine_sparc_collection.find_one(query_key)

    if conversation is None:
        return Response("{'msg': 'Conversation not Found'}", status=404, mimetype='applicatioin/json')
    else:
        conversation['is_delete'] = True
        refine_sparc_collection.update(query_key, conversation)
        return json.dumps({"msg": "success"}, default=json_util.default)


@app.route('/api/translate/conversation/', methods=['POST'])
@login_required
def submit_translate_conversation() -> None:
    """
    update translate conversation
    """
    user = current_user
    if "allowSubmit" not in user.permission:
        return Response("{'msg': 'Permission denied'}", status=403, mimetype='applicatioin/json')
    payload = request.json
    example_id = payload['conversationId']
    database_id = payload['databaseId']
    result = {
        "database_id": database_id,
        "example_id": example_id,
        "dataset": "sparc",
        "is_problematic": payload['problematic'],
        'problematic_note': payload['problematicNote'],
    }
    processed_interactions = list()
    for turn in payload['annotation']:
        processed_turn = {
            'question_id': turn['questionId'],
            'question': turn['question'],
            'sql': turn['sql'],
            'tokenized_question': turn['tokenizedQuestion'],
            'schema_linking': turn['linking'],
            'contextual_phenomena': turn['contextualPhenomena']
        }
        processed_interactions.append(processed_turn)
    result['interaction'] = processed_interactions

    now = datetime.now()

    query_key = {'example_id': example_id}
    original = refine_sparc_collection.find_one(query_key)
    if original is None:
        return Response("{'msg': 'Conversation not Found'}", status=404, mimetype='applicatioin/json')
    if user.username != original.get('created_by', '') and "allowRevise" not in user.permission:
        return Response("{'msg': 'Permission denied'}", status=403, mimetype='applicatioin/json')
    result.update({
        "split": original['split'],
        "revised_by": user.username,
        "revise_date": now,
        "created_by": "" if "created_by" not in original else original['created_by'],
        "create_date": "" if "create_date" not in original else original['create_date'],
        "is_delete": False
    })
    refine_sparc_collection.update(query_key, result)
    return json.dumps({"msg": "success"}, default=json_util.default)


"""
Download Extra Resource
"""
@app.route('/api/sparc_refined/', methods=['GET'])
@login_required
def get_refined_sparc_data():
    tmp_path = "%s.json" % tempfile.mktemp(dir=os.path.join(pathlib.Path().absolute(), "tmp"))
    data = list()
    for doc in refine_sparc_collection.find({}):
        data.append(doc)
    with open(tmp_path, "w") as f:
        f.write(json.dumps(data, default=json_util.default))
    return send_file(filename_or_fp=tmp_path, mimetype="application/json", as_attachment=True, attachment_filename="refined_sparc.json")


@app.route('/api/sparc_tables/', methods=['GET'])
@login_required
def get_refined_sparc_table_data():
    tmp_path = "%s.json" % tempfile.mktemp(dir=os.path.join(pathlib.Path().absolute(), "tmp"))
    data = list()
    for doc in translated_db_collection.find({}):
        data.append(doc)
    with open(tmp_path, "w") as f:
        f.write(json.dumps(data, default=json_util.default))
    return send_file(filename_or_fp=tmp_path, mimetype="application/json", as_attachment=True, attachment_filename="translated_sparc_tables.json")


@app.route('/api/dusql/', methods=['GET'])
@login_required
def get_dusql_data():
    tmp_path = "%s.json" % tempfile.mktemp(dir=os.path.join(pathlib.Path().absolute(), "tmp"))
    data = list()
    for doc in conversation_collection.find({}):
        data.append(doc)
    with open(tmp_path, "w") as f:
        f.write(json.dumps(data, default=json_util.default))
    return send_file(filename_or_fp=tmp_path, mimetype="application/json", as_attachment=True, attachment_filename="dusql.json")


@app.route('/api/dusql_tables/', methods=['GET'])
@login_required
def get_dusql_table_data():
    tmp_path = "%s.json" % tempfile.mktemp(dir=os.path.join(pathlib.Path().absolute(), "tmp"))
    data = list()
    for doc in db_collection.find({"source": "dusql", "labelConversation": True}):
        data.append(doc)
    with open(tmp_path, "w") as f:
        f.write(json.dumps(data, default=json_util.default))
    return send_file(filename_or_fp=tmp_path, mimetype="application/json", as_attachment=True, attachment_filename="dusql_tables.json")


@app.route('/')
def root():
    return app.send_static_file('index.html')