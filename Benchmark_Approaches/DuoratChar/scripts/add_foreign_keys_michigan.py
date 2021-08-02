"""Set foreign keys in the tables.json file.

It would have been better to add foreign key constraints to
sqlite file directly, but sqlite does not support the SQL needed
for this purpose: https://www.sqlite.org/omitted.html

"""
from collections import namedtuple
import json
import sys


def our_curated_primary_keys_for_michigan(dataset_name):
  primary_keys = {
    'restaurants': [
        ('restaurant', 'id'),
        ('geographic', 'city_name')
    ],
    'geo': [
        ('state', 'state_name')
        # all other tables have composite primary keys
    ],
    'yelp': [
        ('business', 'bid'),
        ('category', 'id'),
        ('user', 'uid'),
        ('checkin', 'cid'),
        ('neighborhood', 'id'),
        ('review', 'rid'),
        ('tip', 'tip_id'),
    ],
    'imdb': [
        ('actor', 'aid'),
        ('cast2', 'id'),
        ('classification', 'id'),
        ('company', 'id'),
        ('copyright', 'id'),
        ('directed_by', 'id'),
        ('director', 'did'),
        ('genre', 'gid'),
        ('keyword', 'id'),
        ('made_by', 'id'),
        ('movie', 'mid'),
        ('producer', 'pid'),
        ('tags', 'id'),
        ('tv_series', 'sid'),
        ('writer', 'wid'),
        ('written_by', 'id'),
        # the last one was forgotten in SPIDER
    ],
    'academic': [
        ('author', 'aid'),
        ('conference', 'cid'),
        ('domain', 'did'),
        ('journal', 'jid'),
        ('keyword', 'kid'),
        ('publication', 'pid'),
        ('organization', 'oid'),
    ],
    'advising': [
        ('course', 'course_id'),
        ('course_offering', 'offering_id'),
        ('instructor', 'instructor_id'),
        ('program', 'program_id'),
        ('semester', 'semester_id'),
        ('student', 'student_id'),
    ]
  }
  return primary_keys[dataset_name]


def our_curated_foreign_keys_for_michigan(dataset_name):
  # some "faux" foreign keys are added to connect otherwise disconnected schemas
  foreign_keys = {
    'restaurants': [
        # note that this foreign key relationship is missing in other
        # version of this database, e.g. the SPIDER one
        ('location', 'restaurant_id', 'restaurant', 'id'),
        ('restaurant', 'city_name', 'geographic', 'city_name'),
        ('location', 'city_name', 'geographic', 'city_name'),
    ],
    'geo': [
        ('city', 'state_name', 'state', 'state_name'),
        ('border_info', 'border', 'state', 'state_name'),
        ('border_info', 'state_name', 'state', 'state_name'),
        ('highlow', 'state_name', 'state', 'state_name'),
        ('mountain', 'state_name', 'state', 'state_name'),
        ('lake', 'state_name', 'state', 'state_name'),
        ('river', 'traverse', 'state', 'state_name'),
    ],
    'yelp': [
        ('category', 'business_id', 'business', 'business_id'),
        ('checkin', 'business_id', 'business', 'business_id'),
        ('neighborhood', 'business_id', 'business', 'business_id'),
        ('review', 'user_id', 'user', 'user_id'),
        ('review', 'business_id', 'business', 'business_id'),
        ('tip', 'user_id', 'user', 'user_id'),
        ('tip', 'business_id', 'business', 'business_id'),
    ],
    'imdb': [
        ('copyright', 'msid', 'movie', 'mid'), # faux
        ('classification', 'msid', 'movie', 'mid'), # faux
        ('directed_by', 'msid', 'movie', 'mid'), # faux
        ('written_by', 'msid', 'movie', 'mid'), # faux
        ('made_by', 'msid', 'movie', 'mid'), # faux
        ('copyright', 'msid', 'movie', 'mid'), # faux
        ('tags', 'msid', 'movie', 'mid'), # faux
        ('cast2', 'msid', 'movie', 'mid'), # faux
        ('copyright', 'msid', 'tv_series', 'sid'), # faux
        ('classification', 'msid', 'tv_series', 'sid'), # faux
        ('directed_by', 'msid', 'tv_series', 'sid'), # faux
        ('written_by', 'msid', 'tv_series', 'sid'), # faux
        ('made_by', 'msid', 'tv_series', 'sid'), # faux
        ('copyright', 'msid', 'tv_series', 'sid'), # faux
        ('tags', 'msid', 'tv_series', 'sid'), # faux
        ('cast2', 'msid', 'tv_series', 'sid'), # faux
        ('cast2', 'aid', 'actor', 'aid'),
        ('written_by', 'wid', 'writer', 'wid'),
        ('made_by', 'pid', 'producer', 'pid'),
        ('directed_by', 'did', 'director', 'did'),
        ('classification', 'gid', 'genre', 'gid'),
        ('tags', 'kid', 'keyword', 'id'),
        ('copyright', 'cid', 'company', 'id'),
    ],
    'academic': [
        ('author', 'oid', 'organization', 'oid'),
        ('domain_author', 'did', 'domain', 'did'),
        ('domain_author', 'aid', 'author', 'aid'),
        ('domain_conference', 'did', 'domain', 'did'),
        ('domain_conference', 'cid', 'conference', 'cid'),
        ('domain_journal', 'did', 'domain', 'did'),
        ('domain_journal', 'jid', 'journal', 'jid'),
        ('domain_keyword', 'did', 'domain', 'did'),
        ('domain_keyword', 'kid', 'keyword', 'kid'),
        ('publication', 'cid', 'conference', 'cid'),
        ('publication', 'jid', 'journal', 'jid'),
        ('domain_publication', 'did', 'domain', 'did'),
        ('domain_publication', 'pid', 'publication', 'pid'),
        ('publication_keyword', 'kid', 'keyword', 'kid'),
        ('publication_keyword', 'pid', 'publication', 'pid'),
        ('writes', 'aid', 'author', 'aid'),
        ('writes', 'pid', 'publication', 'pid'),
        ('cite', 'citing', 'publication', 'pid'),
        ('cite', 'cited', 'publication', 'pid'),
    ],
    'advising': [
      ('course_offering', 'course_id', 'course', 'course_id'),
      ('course_prerequisite', 'course_id', 'course', 'course_id'),
      ('course_prerequisite', 'pre_course_id', 'course', 'course_id'),
      ('area', 'course_id', 'course', 'course_id'),
      ('student_record', 'course_id', 'course', 'course_id'),
      ('course_tags_count', 'course_id', 'course', 'course_id'),
      ('program_course', 'course_id', 'course', 'course_id'),
      ('student_record', 'course_id', 'course', 'course_id'),

      ('course_offering', 'semester', 'semester', 'semester_id'),
      ('student_record', 'semester', 'semester', 'semester_id'),

      ('program_requirement', 'program_id', 'program', 'program_id'),
      ('program_course', 'program_id', 'program', 'program_id'),

      ('offering_instructor', 'instructor_id', 'instructor', 'instructor_id'),
      ('comment_instructor', 'instructor_id', 'instructor', 'instructor_id'),

      ('offering_instructor', 'offering_id', 'course_offering', 'offering_id'),
      ('gsi', 'course_offering_id', 'course_offering', 'offering_id'),
      ('student_record', 'offering_id', 'course_offering', 'offering_id'),

      ('comment_instructor', 'student_id', 'student', 'student_id'),
      ('gsi', 'student_id', 'student', 'student_id'),
      ('student_record', 'student_id', 'student', 'student_id'),
    ]
  }
  return foreign_keys[dataset_name]


def find_table(schema, name):
    for i, table_name in enumerate(schema['table_names_original']):
        if table_name.lower() == name:
            return i
    raise ValueError()

def find_column(schema, num, name):
    for i, (table_num, column_name) in enumerate(schema['column_names_original']):
        if table_num == num and column_name.lower() == name:
            return i
    raise ValueError()


if __name__ == '__main__':
    db = sys.argv[1]
    schema_path = sys.argv[2]

    with open(schema_path) as src:
        schema = json.load(src)[0]

    if db == 'imdb':
        # add an alias for the cast2 table
        idx = find_table(schema, 'cast')
        schema['table_names_original'][idx] = 'cast2'

    foreign_keys = []
    for from_table, from_column, to_table, to_column in our_curated_foreign_keys_for_michigan(db):
        from_ = find_column(schema, find_table(schema, from_table), from_column)
        to = find_column(schema, find_table(schema, to_table), to_column)
        foreign_keys.append([from_, to])
    schema['foreign_keys'] = foreign_keys

    primary_keys = []
    for table, column in our_curated_primary_keys_for_michigan(db):
        primary_keys.append(find_column(schema, find_table(schema, table), column))
    schema['primary_keys'] = primary_keys

    with open(schema_path, 'wt') as dst:
        json.dump([schema], dst, indent=2)
