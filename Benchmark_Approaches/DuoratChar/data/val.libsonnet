function(prefix) {
  local databases = [
    "学术比赛",
    "软件排行",
    "world_1",
    "employee_hire_evaluation",
    "中国民族",
    "course_teach",
    "battle_death",
    "flight_2",
    "voter_1",
    "student_transcripts_tracking",
    "医院",
    "museum_visit",
    "网易云阅读",
    "network_1",
    "世界篮球国家队",
    "poker_player",
    "wta_1",
    "dog_kennels",
    "real_estate_properties",
    "各城市一二三产业经济",
    "中国卫视频道",
    "羽毛球",
    "智能音箱",
    "平台自制节目",
    "城市地铁",
    "tvshow",
    "购书平台",
    "动物天敌和朋友",
    "car_1",
    "NBA奖项",
    "orchestra",
    "中国餐饮公司",
    "concert_singer",
    "singer",
    "外卖预定",
    "电视机",
    "中国旅行社",
    "cre_Doc_Template_Mgt",
    "台风",
    "pets_1"
  ],

  name: 'spider',
  paths: [
    prefix + 'database/%s/examples.json' % [db]
    for db in databases
  ],
  tables_paths: [
    prefix + 'database/%s/tables.json' % [db]
    for db in databases
  ],
  db_path: prefix + 'database',
}