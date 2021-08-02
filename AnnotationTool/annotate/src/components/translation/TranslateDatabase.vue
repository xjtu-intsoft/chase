<template>
    <div class="translate-database">
        <h2>Database: <a :href="'http://202.117.43.245:8888/' + databaseId" target="_blank">{{ databaseId }}</a> <span @click="toConversationList">To its Conversations</span> </h2>
        <TranslateTable v-for="(table, index) in tables" v-bind:key="index" :tableId="index" :tableName="table.tableName" :tableNameOriginal="table.tableNameOriginal"
            :columns="table.columns" :candidates="table.candidates" :initTranslation="table.translation" @updateTranslation="updateTranslation" />
        <div class="operation-block">
            <button type="button" class="submit" @click="submitTranslation">Submit</button>
        </div>
    </div>
</template>

<script>
import TranslateTable from './TranslateTable.vue'
import Network from '../../network.js'

export default({
    name: "TranslateDatabase",
    components: {
        TranslateTable
    },
    data() {
        return {
            isSubmitting: false,
            databaseId: "flight_2",
            tables: new Array,
            primaryKeys: new Array,
            foreignKeys: new Array
        }
    },
    mounted() {
        this.databaseId = this.$route.params.id
        this.getDatabase(this.databaseId)
    },
    watch: {
        '$route.params.id': function() {
            this.databaseId = this.$route.params.id
            this.getDatabase(this.databaseId)
        }
    },
    methods: {
        getDatabase: function(databaseId){
            Network.getTranslateDatabaseById(databaseId).then(response => {
                var data = response.data
                var status = response.status
                if(status == 200){
                    this.tables = data.tables
                    this.primaryKeys = data.primaryKeys
                    this.foreignKeys = data.foreignKeys
                }
            }).catch(error => {
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }
            })
        },
        toConversationList: function(){
            this.$router.push({name: "translateConversations", params: {id: this.databaseId}})
        },
        updateTranslation: function(tableId, tableNameTranslation, columnNameTranslations){
            var targetTable = this.tables[tableId]
            targetTable.translation = tableNameTranslation
            for(var i = 0; i < targetTable.columns.length; i++){
                targetTable.columns[i].translation = columnNameTranslations[i]
            }
        },
        submitTranslation: function(){
            if(this.isSubmitting){
                return
            }
            // Check
            var isValid = true
            for(var i = 0; i < this.tables.length; i++){
                let currTable = this.tables[i]
                if((typeof currTable.translation !== 'string') || currTable.translation.trim() === ""){
                    isValid = false
                }
                for(let j = 0; j < currTable.columns.length; j++){
                    let currColumn = currTable.columns[j]
                    if((typeof currColumn.translation !== 'string') || currColumn.translation.trim() === ""){
                        isValid = false
                        break
                    }
                }
                if(!isValid){
                    break
                }
            }
            if(isValid){
                var payload = {
                    tables: this.tables,
                    databaseId: this.databaseId
                }
                this.isSubmitting = true
                Network.submitTranslation(payload).then(response => {
                    var data = response.data
                    this.isSubmitting = false
                    if(response.status == 200 && data.msg === "success"){
                        this.$router.push({name: "translateDatabases"})
                    }
                }).catch(error => {
                    this.isSubmitting = false
                    if(error.response.status == 401){
                        this.$router.replace({name: "login"})
                    }
                })
            }
        }
    },
})
</script>

<style scoped>
.translate-database {
    width: 98%;
    margin-left: auto;
    margin-right: auto;
}
h2 {
    width: 700px;
    text-align: center;
    margin: 30px auto 30px auto;
}
.submit {
    width: 100px;
    height: 40px;
    margin-left: 20px;
    margin-top: 5px;
    margin-bottom: 5px;
}
span {
    margin-left: 15px;
    cursor: pointer;
    text-decoration: underline;
}
span:hover {
    color: blue;
}
</style>