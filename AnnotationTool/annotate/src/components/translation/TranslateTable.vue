<template>
    <div class="translate-table">
        <table>
            <thead>
                <tr>
                    <th></th>
                    <th>Name</th>
                    <th>Original Name</th>
                    <th class="type-cell">Data Type</th>
                    <th>Translation</th>
                    <th>Candidates</th>
                    <th>Conversation Ids</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="item-name">Table: </td>
                    <td><strong> {{ tableName }} </strong></td>
                    <td><strong> {{ tableNameOriginal }} </strong></td>
                    <td class="type-cell"></td>
                    <td>
                        <input type="text" v-model="tableNameTranslation" />
                    </td>
                    <td>
                        <span v-for="(value, name) in candidates" v-bind:key="name" @click="fillInTable(name, value)" >{{name}}</span>
                    </td>
                    <td class="conversation-cell">
                        <a :href="'#/' + cid" target="_blank" v-for="(cid, cidIndex) in tableNameCandidateConversationIds" v-bind:key="cidIndex" >{{cid}}</a>
                    </td>
                </tr>
                <tr v-for="(column, cindex) in columns" v-bind:key="cindex">
                    <td class="item-name">C{{cindex}}: </td>
                    <td><strong>{{ column.columnName }}</strong></td>
                    <td><strong>{{ column.columnNameOriginal }}</strong></td>
                    <td class="type-cell">{{ column.columnType }}</td>
                    <td>
                        <input type="text" v-model="columnNameTranslations[cindex]" />
                    </td>
                    <td>
                        <span v-for="(cvalue, cname) in column.candidates" v-bind:key="cname" @click="fillInColumn(cindex, cname, cvalue)">{{cname}}</span>
                    </td>
                    <td class="conversation-cell">
                        <a :href="'#/' + cid" target="_blank" v-for="(cid, cidIndex) in columnNameCandidateConversationIds[cindex]" v-bind:key="cidIndex" >{{cid}}</a>
                    </td>
                </tr>
            </tbody>
        </table>
        <hr />
    </div>
</template>

<script>
import Vue from 'vue'
import Utils from '../../utils.js'

export default {
    name: 'TranslateTable',
    props: {
        tableId: Number,
        tableName: String,
        tableNameOriginal: String,
        initTranslation: String,
        candidates: Object,
        columns: Array,
    },
    data() {
        return {
            tableNameTranslation: this.initTranslation,
            columnNameTranslations: this.copyTranslation(this.columns),
            tableNameCandidateConversationIds: new Array,
            columnNameCandidateConversationIds: Utils.newEmptyArray(this.columns.length)
        }
    },
    watch: {
        tableNameTranslation: function(){
            if(!(this.tableNameTranslation in this.candidates)){
                this.tableNameCandidateConversationIds = new Array
            }else{
                this.tableNameCandidateConversationIds = this.candidates[this.tableNameTranslation]
            }
            this.updateTranslation()
        },
        columnNameTranslations: {
            deep: true,
            handler: function(){
                for(var i = 0; i < this.columns.length; i++){
                    let currColumn = this.columns[i]
                    let currColumnTranslation = this.columnNameTranslations[i]
                    if(!(currColumnTranslation in currColumn.candidates)){
                        this.columnNameCandidateConversationIds[i] = new Array
                    }else{
                        this.columnNameCandidateConversationIds[i] = currColumn.candidates[currColumnTranslation]
                    }
                }
                this.updateTranslation()
            }
        }
    },
    methods: {
        copyTranslation: function(columns){
            console.log("Run Copy")
            var translations = new Array
            for(var i = 0; i < columns.length; i++){
                translations.push(columns[i].translation)
            }
            return translations
        },
        fillInTable: function(candidate, conversationIds){
            this.tableNameTranslation = candidate
            this.tableNameCandidateConversationIds = conversationIds
        },
        fillInColumn: function(columnIndex, candidate, conversationIds){
            // Hack
            Vue.set(this.columnNameTranslations, columnIndex, candidate)
            Vue.set(this.columnNameCandidateConversationIds, columnIndex, conversationIds)
        },
        updateTranslation: function(){
            this.$emit("updateTranslation", this.tableId, this.tableNameTranslation, this.columnNameTranslations)
        }
    },
}
</script>

<style scoped>
table {
    border-spacing: 10px 15px;
    margin: 0 0 0 20px;
    font-size: 18px;
}
tr {
    margin: 0 0 13px 0;
}
td {
    min-width: 200px;
    padding: 5px 5px 5px 5px;
}
th {
    font-weight: normal;
    text-align: left;
    padding: 5px 5px 5px 5px;
}
span {
    margin-right: 10px;
    cursor: pointer;
}
span:hover {
    color: blue;
}
.item-name {
    min-width: 50px;
    text-align: right;
    padding-right: 10px;
}
.type-cell {
    min-width: 120px;
}
.conversation-cell {
    width: 300px;
}
input {
    width: 90%;
    font-size: 20px;
    padding: 8px 8px 8px 8px;
    box-sizing: border-box;
}
a {
    margin-right: 10px;
    cursor: pointer;
}
</style>