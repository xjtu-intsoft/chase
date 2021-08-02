<template>
    <div id="aside">
        <div class="card">
            <div class="header">
                <h2>What is Chase?</h2>
            </div>
            <p>
                Chase is to date the largest Chinese dataset for the cross-database context-dependent Text-to-SQL problem.
                It consists of 5,459 question sequences (17,940 questions) over 280 databases.
                Each question in Chase has rich semantic annotations, including its SQL query, contextual dependency, and schema linking.
            </p>
            <row container :gutter="10">
                <column :xs="12" :md="12" :lg="4">
                     <el-button type="success" round><a href="https://aclanthology.org/2021.acl-long.180/" class="btn-link">Paper</a></el-button>
                </column>
                <column :xs="12" :md="12" :lg="4">
                    <el-button type="success" round><a href="https://github.com/xjtu-intsoft/chase/tree/page/data" class="btn-link">Dataset</a></el-button>
                </column>
                <column :xs="12" :md="12" :lg="4">
                    <el-button type="success" round><a href="https://github.com/xjtu-intsoft/chase" class="btn-link">Github</a></el-button>
                </column>
            </row>
        </div>
        <div class="card">
            <div class="header">
                <h2>News</h2>
            </div>
            <p>
                We have released the data of Chase. 
                Please feel free to examine the data and contact us if there are any problem.
                We are constructing Japanese version of Chase. 
            </p>
        </div>
        <div class="card">
            <div class="header">
                <h2>How to Construct Chase?</h2>
            </div>
            <p>
                Data in Chase are collected from two sources.
                First, we collect 120 databases from <a href="https://www.aclweb.org/anthology/2020.emnlp-main.562.pdf"><b>DuSQL</b></a>, which is a context-independent Chinese Text-to-SQL dataset.
                To collect question sequences, we recruit 12 Chinese college students who are skilled at SQL to create question sequences against databases from scratch.
                Second, to enlarge dour dataset, we translate all the data in <a href="https://yale-lily.github.io/sparc"><b>SParC</b></a>, inlcuding questions, SQL queries, and databases, from English to Chinese.
                11 students are recruited to perform the translation.
                Please refer to our <a href="https://aclanthology.org/2021.acl-long.180/"><b>ACL'21 paper</b></a> for more details of the dataset construction.
            </p>
             <el-table :data="tableData" style="width: 100%">
                <el-table-column label="Dataset" prop="name"></el-table-column>
                <el-table-column label="# DB" prop="db"></el-table-column>
                <el-table-column label="# Seq." prop="sequence"></el-table-column>
                <el-table-column label="# Pair" prop="pair"></el-table-column>
                <el-table-column label="# Avg. Turn" prop="turn"></el-table-column>
             </el-table>
        </div>
        <div class="card">
            <div class="header">
                <h2>Have Questions or Want to Contribute?</h2>
            </div>
            <p>Please ask us questions at our <a href="https://github.com/xjtu-intsoft/chase/issues">Github issue pages</a> or contact Jiaqi Guo (JasperGuo AT outlook.com).</p>
            <p>We plan to support more languages in Chase. Please contact us if you are interested in building a context-dependent Text-to-SQL dataset for your native language.</p>
        </div>
        <div class="card">
            <div class="header">
                <h2>Acknowledgements</h2>
            </div>
            <p> 
            We thank Weixu Zhang, Jiawei Lin, Xiaotong Zheng, Nan Hu, Tingting Zhang, Zekun Qi, Chengzu Li, Junjie Tao, Jinghan He, and Yu Ma for participating in the construction of Chase. 
            Ming Fan was partially supported by NSFC (61902306), China Postdoctoral Science Foundation (2019TQ0251, 2020M673439), Youth Talent Support Plan of Xi’an Association for Science and Technology (095920201303).
            </p>
        </div>
        <div class="card">
            <div class="header">
                <h2>Citation</h2>
            </div>
            <pre>
                <code>
@inproceedings{guo-etal-2021-chase,
  title = {Chase: A Large-Scale and Pragmatic Chinese Dataset for 
              Cross-Database Context-Dependent Text-to-SQL},
  author = {Jiaqi Guo and Ziliang Si and Yu Wang and Qian Liu 
                  and Ming Fan and Jian-Guang Lou and Zijiang Yang 
                  and Ting Liu}
  booktitle = "Proceedings of the 59th Annual Meeting of 
                      the Association for Computational Linguistics and 
                      the 11th International Joint Conference on 
                      Natural Language Processing",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  year = 2021,
  month = aug,
  pages = 2316--2331,
  url = "https://aclanthology.org/2021.acl-long.180",
}
                </code>
            </pre>
        </div>
    </div>
</template>

<script>
export default {
  name: 'Aside',
  data() {
      return {
          tableData: [
              {
                  name: "Chase",
                  db: 280,
                  table: 1280,
                  sequence: 5459,
                  pair: 17940,
                  turn: 3.3
              },
              {
                  name: "Chase-C",
                  db: 120,
                  table: 462,
                  sequence: 2003,
                  pair: 7694,
                  turn: 3.8
              },
              {
                  name: "Chase-T",
                  db: 160,
                  table: 818,
                  sequence: 3456,
                  pair: 10246,
                  turn: 3.0
              }
          ]
      }
  },
}
</script>

<style>
#aside {
    width: 100%;
    background-color: #fff;
    font-family: 'Avenir, Helvetica, Arial, sans-serif';
    padding-top:30px;
    padding-bottom: 10px;
}

#aside a{
    text-decoration: none;
}

.btn-link {
    color: #fff;
}

.card {
    margin: 0 auto 30px auto;
    width: 90%;
}

.header > h2 {
    margin: 0;
    font-size: 1.4em;
    padding-bottom: 2px;
    border-bottom: 2px solid #ddd;
    text-align: left;
}

.card > p {
    text-align: left;
    font-family: 'Lato, Arial, sans-serif';
}

p.indent {
    text-indent: 20px;
}

.btn {
    display: block;
    margin: 10px auto;
    max-width: 40%;
    line-height: 2.0;
    width: 100%;
    background-color: #73c44a;
    color: #fff !important;
    text-align: center;
    font-size: 1.11em;
    font-weight: 400;
    border: 2px solid #73c44a;
    border-radius: 20px;
    transition: all 0.5s;
    white-space: normal;
    user-select: none;
    touch-action: manipulation;
    cursor: pointer;
    text-decoration: none;
}

pre code {
  background-color: #eee;
  border: 0px solid #999;
  display: block;
  padding: 0 10px 0 10px;
  text-align: left;
  font-family: 'Lato, Arial, sans-serif';
  font-size: 14px;
  width: 95%;
  margin: 0 auto 0 auto;
  overflow: auto;
}

</style>