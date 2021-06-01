<template>
    <div id="chase-c-leaderboard">
        <div class="card">
            <div class="header">
                <h2>Chase Leaderboard - Exact Set Match without Values</h2>
            </div>
            <p> Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.</p>
            <AccuracyTable :tableData=tableData />
        </div>
    </div>
</template>

<script>

import AccuracyTable from './AccuracyTable.vue'

export default {
  name: 'ChaseTLeaderboard',
  components: {
      AccuracyTable
  },
  data() {
      return {
        tableData: [
              {
                rank: 1,
                date: "2021-06-01",
                model: "IGSQL",
                pretrainLM: "Chinese-BERT-base",
                ensemble: "false",
                QM: 40.1,
                IM: 14.2,
                code: "https://github.com/headacheboy/IGSQL",
                paper: "https://www.aclweb.org/anthology/2020.emnlp-main.560.pdf",
                affliation: "Peking University"
              },
              {
                rank: 2,
                date: "2021-06-01",
                model: "EditSQL",
                pretrainLM: "Chinese-BERT-base",
                ensemble: "false",
                QM: 40.1,
                IM: 14.2,
                code: "https://github.com/headacheboy/IGSQL",
                paper: "https://www.aclweb.org/anthology/2020.emnlp-main.560.pdf",
                affliation: "Peking University"
              }
          ]
      }
  },
}
</script>

<style>
#main-leaderboard {
    width: 100%;
    background-color: #fff;
    font-family: 'Avenir, Helvetica, Arial, sans-serif';
    padding-top:30px;
    padding-bottom: 10px;
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
</style>