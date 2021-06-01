<template>
    <div id="main-leaderboard">
        <div class="card">
            <div class="header">
                <h2>Chase Leaderboard - Exact Set Match without Values</h2>
            </div>
            <AccuracyTable :tableData=chaseTableData />
        </div>
        <div class="card">
            <div class="header">
                <h2>Chase-C Leaderboard - Exact Set Match without Values</h2>
            </div>
            <AccuracyTable :tableData=chaseCTableData />
        </div>
        <div class="card">
            <div class="header">
                <h2>Chase-T Leaderboard - Exact Set Match without Values</h2>
            </div>
            <AccuracyTable :tableData=chaseTTableData />
        </div>
    </div>
</template>

<script>

import AccuracyTable from './AccuracyTable.vue'

export default {
  name: 'MainLeaderboard',
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
                code: "https://github.com/headacheboy/IGSQL",
                paper: "https://www.aclweb.org/anthology/2020.emnlp-main.560.pdf",
                affliation: "Peking University",
                chase: {
                    QM: 40.4,
                    IM: 15.6,
                    devQM: 41.4,
                    devIM: 20.0,
                    devHardness: {
                        easy: 65.6, medium: 41.1, hard: 27.3, extra: 16.6
                    },
                    devContext: {
                        independent: 56.3, dependent: 33.3, coreference: 33.1, ellipsis: 33.2
                    },
                    hardness: {
                        easy: 65.9, medium: 38.5, hard: 27.1, extra: 10.2
                    },
                    context: {
                        independent: 55.7, dependent: 31.2, coreference: 35.1, ellipsis: 25.8
                    }
                },
                chaseC: {
                    QM: 32.6,
                    IM: 9.3,
                    devQM: 31.4,
                    devIM: 10.8,
                    devHardness: {
                        easy: 52.3, medium: 36.6, hard: 23.5, extra: 11.4
                    },
                    devContext: {
                        independent: 45.4, dependent: 25.7, coreference: 25.8, ellipsis: 25.2
                    },
                    hardness: {
                        easy: 61.2, medium: 33.8, hard: 22.9, extra: 9.0
                    },
                    context: {
                        independent: 46.7, dependent: 26.5, coreference: 29.8, ellipsis: 21.6
                    }
                },
                chaseT: {
                    QM: 43.3,
                    IM: 26.3,
                    devQM: 43.3,
                    devIM: 26.3,
                    devHardness: {
                        easy: 66.2, medium: 36.9, hard: 24.0, extra: 12.5
                    },
                    devContext: {
                        independent: 56.2, dependent: 33.9, coreference: 35.1, ellipsis: 31.1
                    },
                    hardness: {
                        easy: 66.2, medium: 36.9, hard: 24.0, extra: 12.5
                    },
                    context: {
                        independent: 56.2, dependent: 33.9, coreference: 35.1, ellipsis: 31.1
                    }
                }
              },
              {
                rank: 2,
                date: "2021-06-01",
                model: "EditSQL",
                pretrainLM: "Chinese-BERT-base",
                ensemble: "false",
                code: "https://github.com/ryanzhumich/editsql",
                paper: "https://www.aclweb.org/anthology/D19-1537.pdf",
                affliation: "Yale University, Salesforce Research, and Cornell University",
                chase: {
                    QM: 37.8,
                    IM: 14.7,
                    devQM: 37.7,
                    devIM: 17.4,
                    devHardness: {
                        easy: 63.2, medium: 37.0, hard: 22.0, extra: 13.8
                    },
                    devContext: {
                        independent: 52.8, dependent: 29.5, coreference: 29.4, ellipsis: 29.2
                    },
                    hardness: {
                        easy: 63.0, medium: 34.3, hard: 26.1, extra: 10.8
                    },
                    context: {
                        independent: 54.1, dependent: 28.0, coreference: 30.1, ellipsis: 24.5
                    },
                },
                chaseC: {
                    QM: 32.6,
                    IM: 8.7,
                    devQM: 33.6,
                    devIM: 8.4,
                    devHardness: {
                        easy: 58.5, medium: 36.8, hard: 24.4, extra: 15.4
                    },
                    devContext: {
                        independent: 51.1, dependent: 26.6, coreference: 27.1, ellipsis: 25.4
                    },
                    hardness: {
                        easy: 59.1, medium: 33.6, hard: 25.6, extra: 8.0
                    },
                    context: {
                        independent: 49.3, dependent: 25.3, coreference: 28.6, ellipsis: 20.2
                    }
                },
                chaseT: {
                    QM: 41.6,
                    IM: 21.6,
                    devHardness: {
                        easy: 65.7, medium: 34.7, hard: 20.8, extra: 14.6
                    },
                    devContext: {
                        independent: 55.8, dependent: 30.2, coreference: 29.4, ellipsis: 31.1
                    },
                    hardness: {
                        easy: 65.7, medium: 34.7, hard: 20.8, extra: 14.6
                    },
                    context: {
                        independent: 55.8, dependent: 30.2, coreference: 29.4, ellipsis: 31.1
                    },
                    devQM: 41.6,
                    devIM: 21.6,
                }
              },
              {
                rank: 3,
                date: "2021-06-01",
                model: "RatSQL + Concat",
                pretrainLM: "Chinese-BERT-base",
                ensemble: "false",
                code: "https://github.com/microsoft/rat-sql",
                paper: "https://www.aclweb.org/anthology/2020.acl-main.677.pdf",
                affliation: "University of Edinburgh, UC Berkeley, and Microsoft Research",
                chase: {
                    QM: 32.5,
                    IM: 9.8,
                    devQM: 35.1,
                    devIM: 14.6,
                    devHardness: {
                        easy: 60.9, medium: 35.4, hard: 16.4, extra: 11.6
                    },
                    devContext: {
                        independent: 49.3, dependent: 27.3, coreference: 27.1, ellipsis: 28.1
                    },
                    hardness: {
                        easy: 56.0, medium: 29.1, hard: 18.4, extra: 11.7
                    },
                    context: {
                        independent: 45.7, dependent: 24.5, coreference: 25.0, ellipsis: 23.8
                    },
                },
                chaseC: {
                    QM: 23.9,
                    IM: 4.5,
                    devQM: 24.6,
                    devIM: 5.4,
                    devHardness: {
                        easy: 46.5, medium: 29.5, hard: 13.3, extra: 8.3
                    },
                    devContext: {
                        independent: 35.8, dependent: 20.0, coreference: 19.8, ellipsis: 20.2
                    },
                    hardness: {
                        easy: 48.3, medium: 24.9, hard: 14.3, extra: 5.2
                    },
                    context: {
                        independent: 34.7, dependent: 19.2, coreference: 21.5, ellipsis: 15.6
                    }
                },
                chaseT: {
                    QM: 43.7,
                    IM: 21.6,
                    devQM: 43.7,
                    devIM: 21.6,
                    devHardness: {
                        easy: 68.0, medium: 37.4, hard: 18.2, extra: 15.3
                    },
                    devContext: {
                        independent: 56.4, dependent: 34.5, coreference: 33.2, ellipsis: 36.8
                    },
                    hardness: {
                        easy: 68.0, medium: 37.4, hard: 18.2, extra: 15.3
                    },
                    context: {
                        independent: 56.4, dependent: 34.5, coreference: 33.2, ellipsis: 36.8
                    },
                }
              }
          ]
      }
  },
  computed: {
    chaseTableData: function() {
        var data = new Array
        for (let i = 0; i < this.tableData.length; i++) {
            let inst = Object.assign({}, this.tableData[i])
            inst = Object.assign(inst, inst.chase)
            inst = Object.assign(inst, inst.chase.hardness)
            inst = Object.assign(inst, inst.chase.context)
            data.push(inst)
        }
        return data
    },
    chaseCTableData: function() {
        var data = new Array
        for (let i = 0; i < this.tableData.length; i++) {
            let inst = Object.assign({}, this.tableData[i])
            inst = Object.assign(inst, inst.chaseC)
            inst = Object.assign(inst, inst.chaseC.hardness)
            inst = Object.assign(inst, inst.chaseC.context)
            data.push(inst)
        }
        return data
    },
    chaseTTableData: function() {
        var data = new Array
        for (let i = 0; i < this.tableData.length; i++) {
            let inst = Object.assign({}, this.tableData[i])
            inst = Object.assign(inst, inst.chaseT)
            inst = Object.assign(inst, inst.chaseT.hardness)
            inst = Object.assign(inst, inst.chaseT.context)
            data.push(inst)
        }
        return data
    }
  }
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