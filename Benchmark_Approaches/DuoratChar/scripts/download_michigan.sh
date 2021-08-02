# coding=utf-8
#~/usr/bin/env bash

CODE=$PWD
MICHIGAN_GITHUB=https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master
DATABASE_DIR=data/database

function gdrive_download {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}

function create_sqlite_db { 
    DB=$1
    SQL_FILE=$2

    bash -x $CODE/scripts/mysql2sqlite.sh ${DB}_test/$SQL_FILE ${DB}_test/${DB}_test.sqlite
    create_schema_file $DB
}

function create_schema_file {
    DB=$1 
    docker exec my_duorat python /app/scripts/get_tables_new.py\
        /app/data/database/${DB}_test/${DB}_test.sqlite\
        /app/data/database/${DB}_test/tables.json
    python $CODE/scripts/add_foreign_keys_michigan.py ${DB} ${DB}_test/tables.json
}

function go_geo {
    mkdir geo_test && cd geo_test
    wget $MICHIGAN_GITHUB/data/geography.json || exit
    wget $MICHIGAN_GITHUB/data/geography-db.sql || exit
    python $CODE/scripts/convert_from_michigan.py --input geography.json --db-id geo_test --output examples.json\
        --split train --split dev
    cd -

    create_sqlite_db geo geography-db.sql
    python $CODE/scripts/create_cache.py --dataset_name geo --split train,dev
}

function go_atis {
    mkdir atis_test && cd atis_test
    wget $MICHIGAN_GITHUB/data/atis.json || exit
    wget $MICHIGAN_GITHUB/data/atis-db.sql || exit
    python $CODE/scripts/convert_from_michigan.py --input atis.json --db-id atis_test --output examples.json --split dev
    cd -

    create_sqlite_db atis atis-db.sql
}

function go_academic {
    mkdir academic_test && cd academic_test
    wget $MICHIGAN_GITHUB/data/academic.json || exit
    gdrive_download 0B-2uoWxAwJGKbnBCdkJGcHc5dDQ academic.sql
    python $CODE/scripts/convert_from_michigan.py --input academic.json --db-id academic_test --output examples.json\
        --split 0 --split 1 --split 2 --split 3 --split 4\
        --split 5 --split 6 --split 7 --split 8 --split 9
    cd -

    create_sqlite_db academic academic.sql

    echo "Building Academic indices"
    sqlite3 academic_test/academic_test.sqlite <<HEREDOC_EOF
CREATE INDEX IF NOT EXISTS "author_oid" ON "author" ("oid");
CREATE INDEX IF NOT EXISTS "cite_cited" ON "cite" ("cited");
CREATE INDEX IF NOT EXISTS "publication_title" ON "publication" ("title");
CREATE INDEX IF NOT EXISTS "publication_year" ON "publication" ("year");
HEREDOC_EOF
}

# Restaurants
function go_restaurants { 
    echo "Downloading Restaurants annotations."
    mkdir restaurants_test && cd restaurants_test
    wget $MICHIGAN_GITHUB/data/restaurants.json || exit
    wget $MICHIGAN_GITHUB/data/restaurants-db.txt || exit
    python $CODE/scripts/convert_from_michigan.py --input restaurants.json --db-id restaurants_test --output examples.json\
        --split 0 --split 1 --split 2 --split 3 --split 4\
        --split 5 --split 6 --split 7 --split 8 --split 9
    python $CODE/scripts/create_restaurant_database.py restaurants-db.txt restaurants_test.sqlite
    cd -

    create_schema_file restaurants
    python $CODE/scripts/create_cache.py --dataset_name restaurants --split 0,1,2,3,4,5,6,7,8,9
}

# Yelp
function go_yelp {
    mkdir yelp_test
    cd yelp_test
    wget $MICHIGAN_GITHUB/data/yelp.json || exit
    gdrive_download 0B-2uoWxAwJGKX09Ld1RhVW5NbVk yelp.sql || exit
    python $CODE/scripts/convert_from_michigan.py --input yelp.json --db-id yelp_test --output examples.json\
        --split 0 --split 1 --split 2 --split 3 --split 4\
        --split 5 --split 6 --split 7 --split 8 --split 9
    cd -

    create_sqlite_db yelp yelp.sql
    echo "Building YELP indices"
    sqlite3 yelp_test/yelp_test.sqlite << HEREDOC_EOF 
CREATE INDEX IF NOT EXISTS "category_on_categoryname" ON "category" ("category_name");
CREATE INDEX IF NOT EXISTS "category_on_businessid" ON "category" ("business_id");
CREATE INDEX IF NOT EXISTS "category_on_categoryname" ON "category" ("category_name");
CREATE INDEX IF NOT EXISTS "review_on_businessid" ON "review" ("business_id");
CREATE INDEX IF NOT EXISTS "review_on_userid" ON "review" ("user_id");
CREATE INDEX IF NOT EXISTS "review_on_both" ON "review" ("business_id", "user_id");
CREATE INDEX IF NOT EXISTS "review_on_userid" ON "review" ("user_id");
CREATE INDEX IF NOT EXISTS "review_on_rid" ON "review" ("rid");
CREATE INDEX IF NOT EXISTS "business_on_bid" ON "business" ("bid");
CREATE INDEX IF NOT EXISTS "business_on_businessid" ON "business" ("business_id");
CREATE INDEX IF NOT EXISTS "user_on_userid" ON "user" ("user_id");
CREATE INDEX IF NOT EXISTS "user_on_uid" ON "user" ("uid");
CREATE INDEX IF NOT EXISTS "tip_on_businessid" ON "tip" ("business_id");
CREATE INDEX IF NOT EXISTS "checkin_on_businessid" ON "checkin" ("business_id");
HEREDOC_EOF
}

# IMDB
function go_imdb {
    mkdir imdb_test && cd imdb_test
    wget $MICHIGAN_GITHUB/data/imdb.json || exit
    sed -i 's/CAST AS/CAST2 AS/g' imdb.json
    gdrive_download 0B-2uoWxAwJGKNkFGZHFCQ29yNWM imdb.sql
    python $CODE/scripts/convert_from_michigan.py --input imdb.json --db-id imdb_test --output examples.json\
        --split 0 --split 1 --split 2 --split 3 --split 4\
        --split 5 --split 6 --split 7 --split 8 --split 9
    cd -

    create_sqlite_db imdb imdb.sql

    echo "Building IMDB indices"
    sqlite3 imdb_test/imdb_test.sqlite << HEREDOC_EOF 
CREATE INDEX IF NOT EXISTS "cast_msid" ON "cast" ("msid");
CREATE INDEX IF NOT EXISTS "directed_by_did" ON "directed_by" ("did");
CREATE INDEX IF NOT EXISTS "directed_by_msid" ON "directed_by" ("msid");
CREATE INDEX IF NOT EXISTS "made_by_msid" ON "made_by" ("msid");
CREATE INDEX IF NOT EXISTS "made_by_pid" ON "made_by" ("pid");
CREATE INDEX IF NOT EXISTS "cast_aid" ON "cast" ("aid");
CREATE INDEX IF NOT EXISTS "actor_aid" ON "actor" ("aid");
CREATE INDEX IF NOT EXISTS "actor_gender" ON "actor" ("gender");
CREATE INDEX IF NOT EXISTS "actor_name" ON "actor" ("name");
CREATE INDEX IF NOT EXISTS "movie_mid" ON "movie" ("mid");
CREATE INDEX IF NOT EXISTS "movie_title" ON "movie" ("title");
CREATE INDEX IF NOT EXISTS "director_did" ON "director" ("did");
HEREDOC_EOF

    echo "ALTER TABLE cast RENAME TO cast2" | sqlite3 imdb_test/imdb_test.sqlite
}

function go_scholar {
    mkdir scholar_test && cd scholar_test
    wget $MICHIGAN_GITHUB/data/scholar.json || exit
    gdrive_download 0Bw5kFkY8RRXYRXdYYlhfdXRlTVk scholar.zip
    unzip scholar.zip
    python $CODE/scripts/convert_from_michigan.py --input scholar.json --db-id scholar_test --output examples.json\
        --split train --split dev
    cd -

    create_sqlite_db scholar scholar_mysql_dump.db
echo "Building Scholar indices"
sqlite3 scholar_test/scholar_test.sqlite <<HEREDOC_EOF
CREATE INDEX IF NOT EXISTS "writes_authorId" ON "writes" ("authorId");
CREATE INDEX IF NOT EXISTS "writes_paperId" ON "writes" ("paperId");
CREATE INDEX IF NOT EXISTS "author_authorName" ON "author" ("authorName" collate nocase);
CREATE INDEX IF NOT EXISTS "dataset_datasetName" ON "dataset" ("datasetName" collate nocase);
CREATE INDEX IF NOT EXISTS "journal_journalName" ON "journal" ("journalName" collate nocase);
CREATE INDEX IF NOT EXISTS "keyphrase_keyphraseName" ON "keyphrase" ("keyphraseName" collate nocase);
CREATE INDEX IF NOT EXISTS "paper_title" ON "paper" ("title" collate nocase);
CREATE INDEX IF NOT EXISTS "venue_venueName" ON "venue" ("venueName" collate nocase);
HEREDOC_EOF
}

function go_advising {
    mkdir advising_test && cd advising_test
    wget $MICHIGAN_GITHUB/data/advising.json || exit
    wget $MICHIGAN_GITHUB/data/advising-db.sql || exit
    python $CODE/scripts/convert_from_michigan.py --input advising.json --db-id advising_test --output examples.json\
        --split train --split dev
    cd -
    
    create_sqlite_db advising advising-db.sql
}


cd $DATABASE_DIR

# Check that MySQL docker container is running
docker exec duorat-mysql echo 
if [ $? != 0 ]
then    
    echo "Run scripts/mysql_docker_build_and_run.sh"
    exit
fi    

for DB in $@
do
    if [ ! -d ${DB}_test ] 
    then
        echo "Download and construct ${DB}"
        go_${DB}
    fi
done     
