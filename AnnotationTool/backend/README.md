# Annotation Backend

### Project setup
```
// Setup Python Environment
conda create -n annotate python=3.7
conda activate annotate
pip install -r requirements.txt

// Setup MongoDB
Install mongodb follow the guidelines in https://www.mongodb.com/try/download/community
python initialize_mongodb.py
python config_users.py

// Setup SparC & DuSQL Database
1. Run python ./parse_sqlite_db.py in ./data/DuSQL, which parses all .sqlite to .json
2. Configure ./data/conversation_databases.txt
2. Run python ./update_databases.py in ./, which stores .json into mongodb
```

### Development

```
// Linux
export FLASK_APP=server.py
python -m flask run

// Windows
set FLASK_APP=server.py
python -m flask run

browse http://127.0.0.1:5000/
```

### Deployment
```
gunicorn -w 8 -b 127.0.0.1:5000 server:app --access-logfile [PATH]
```