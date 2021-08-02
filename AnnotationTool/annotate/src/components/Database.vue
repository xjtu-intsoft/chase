<template>
    <div class="database scrollable-block" v-bind:class="{overflow: showInputBlock }">
        <h2>Database: <a :href="'http://202.117.43.245:8888/' + databaseId" target="_blank">{{ databaseId }}</a></h2>
        <div v-if="showInputBlock" class="input-block">
            <label for="selected-tables">Selected Tables: </label>
            <input type="text" name="selected-tables" :value="formattedTables" readonly>
            <button type="button" class="button" @click="clear">Clear</button>
        </div>
        <div v-if="showInputBlock" class="input-block">
            <label for="from-clause">Synthesized From Clause: </label>
            <textarea type="text" name="from-clause" ref="fromClause" cols="50" rows="5" charswidth="17" v-model="fromClause"></textarea>
            <button type="button" class="button copy" @click="copy">Copy</button>
        </div>
        <hr v-if="showInputBlock" />
        <Table v-for="(table, index) in database" @selectTable="selectTable" v-bind:key="index" :tableName="table.tableName" :columns="table.columns" />
    </div>
</template>

<script>
import Utils from '../utils.js'
import Table from './Table.vue'

export default {
    name: 'Database',
    components: {
        Table
    },
    props: {
        databaseId: String,
        database: Array,
        databaseGraph: Object,
        showInputBlock: Boolean
    },
    data() {
        return {
            selectedTables: new Array,
            formattedTables: "",
            fromClause: "",
        }
    },
    watch: {
        selectedTables: function(){
            // format
            this.formattedTables = ""
            for(var i = 0; i < this.selectedTables.length; i++){
                var tn = this.selectedTables[i]
                if(i == 0){
                    this.formattedTables = tn
                }else{
                    this.formattedTables = this.formattedTables + " , " + tn
                }
            }
            this.fromClause = Utils.searchJoinPath(this.databaseGraph.nodes, this.databaseGraph.edges, this.selectedTables)
        }
    },
    methods: {
        selectTable: function(tableName){
            var index = -1
            for(var i = 0; i < this.selectedTables.length; i++){
                if(this.selectedTables[i] === tableName){
                    index = i
                    break
                }
            }
            if(index == -1){
                // Add
                this.selectedTables.push(tableName)
            }else{
                // Remove
                this.selectedTables.splice(index, 1)
            }
        },
        clear: function(){
            this.selectedTables = new Array
        },
        copy: function(){
            var el = this.$refs.fromClause
            el.select();
            document.execCommand('copy');
        }
    },
}
</script>

<style scoped>
.database {
    width: 50%;
    margin-left: 3%;
}
.overflow {
    height: 940px;
    overflow-y: auto;
    overflow-x: visible;
}
h2 {
    margin: 30px 0 15px 0;
    text-align: center;
}
.input-block {
    margin-top: 30px;
    margin-left: 20px;
    width: 100%;
}
.input-block > label {
    display: block;
    margin-bottom: 5px;
    font-size: 17px;
    font-weight: bold;
}
.input-block input {
    width: 70%;
    height: 30px;
    padding: 5px 5px 5px 5px;
    font-size: 17px;
}
.button {
    width: 10%;
    height: 40px;
    margin-left: 20px;
    margin-top: 5px;
}
textarea {
    display: block;
    padding: 5px 5px 5px 5px;
    font-size: 17px;
    resize: vertical;
    font-family: Avenir, Helvetica, Arial, sans-serif;
}
.copy {
    margin-left: 0;
    margin-top: 10px;
}
</style>
