set -e
# input: a file with SQL command in my SQL format
MYSQL_INPUT=$1; shift
# output: an SQLITE database
SQLITE_OUTPUT=$1; shift
set +e

docker cp $MYSQL_INPUT duorat-mysql:/workspace
docker exec duorat-mysql bash -c "echo DROP DATABASE DB | mysql -uroot -pp"
docker exec duorat-mysql bash -c "echo CREATE DATABASE DB | mysql -uroot -pp"
docker exec duorat-mysql bash -c "cat <( echo USE DB ) `basename $MYSQL_INPUT` | mysql -uroot -pp"
docker exec duorat-mysql bash -c "[ -f  /workspace/DB.sqlite ] && rm /workspace/DB.sqlite"
docker exec duorat-mysql bash -c "bash mysql2sqlite.sh $@ -uroot -pp DB | sqlite3 /workspace/DB.sqlite"
docker cp duorat-mysql:/workspace/DB.sqlite  $SQLITE_OUTPUT
