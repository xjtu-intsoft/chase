docker build -t duorat-mysql-image -f Dockerfile_Mysql .
docker run --rm --name duorat-mysql -e MYSQL_ROOT_PASSWORD=p -d duorat-mysql-image
