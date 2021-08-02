<template>
    <div class='translate-conversation-block'>
        <table>
            <tr>
                <td class='item-name'>Question:</td>
                <td>
                    <form onsubmit="return false">
                        <input type="text" class="main-input-block" v-model="question">
                    </form>
                </td>
            </tr>
            <tr>
                <td class='item-name'>Contextual Phenomena:</td>
                <span class='phenomenon' v-for="(p, index) in contextualPhenomenaStyles" v-bind:key="index" 
                        v-on:click="togglePhenomenon(p.name)" v-bind:style="p.style" >
                    {{p.name}}
                </span>
            </tr>
            <tr>
                <td class='item-name'>SQL:</td>
                <td>
                    <form onsubmit="return false" @keyup.enter="parseSQL">
                        <!-- <input type="text" class="main-input-block" v-model="sql"> -->
                        <textarea class="main-input-block" type="text" cols="47" rows="5" charswidth="17" v-model="sql"></textarea>
                    </form>
                    <div class="error-message" v-if="showErrorMessage">
                        {{errorMessage}}
                    </div>
                    <button type="button" class="parse-button" @click="parseSQL">Execute & Parse</button>
                </td>
            </tr>
            <tr>
                <td class='item-name'>Results:</td>
                <td>
                    <span v-if="data_details.length==0">
                        Empty result set.
                    </span>
                    <div class="table-div scrollable-block" v-if="data_details.length!=0">
                        <div class="table-scroll-wrapper">
                            <table>
                                <thead>
                                    <tr>
                                        <th v-for="(col_desc, index) in data_description" v-bind:key="index">
                                            {{ col_desc[0] }}
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr v-for="(row, index) in data_details" v-bind:key="index">
                                        <td v-for="(value, index) in row" v-bind:key="index">
                                            <span v-if="!value">
                                                NULL
                                            </span>
                                            <span v-else>
                                                {{ value }}
                                            </span>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </td>
            </tr>
            <tr>
                <td class='item-name'>Tokens:</td>
                <span class='token' v-for="(token, index) in tokenizedChineseQuestion" v-bind:key="index" v-bind:style="tokenStyles[index]" >
                    {{index}}. {{ token }}
                </span>
            </tr>
            <tr>
                <td class='item-name'>SQL Tables:</td>
                <td>
                    <span class='entity table-entity' v-for="(table, index) in tables" v-bind:class="{sql: !table.inSQL}"  v-bind:key="index" v-on:click="selectEntity('table', index)">
                        <strong>{{ table.name }}</strong>
                    </span>
                </td>
            </tr>
            <tr>
                <td class='item-name'>SQL Columns:</td>
                <td>
                    <span class='entity' v-for="(column, index) in columns" v-bind:key="index" v-on:click="selectEntity('column', index)">
                        <strong>{{ column.name }}</strong><br />{{ column.tableName }}
                    </span>
                </td>
            </tr>
            <tr>
                <td class='item-name'>SQL Values:</td>
                <td>
                    <span class='entity' v-for="(value, index) in values" v-bind:key="index" v-on:click="selectEntity('value', index)">
                        <strong>{{ value.value }}</strong><br />{{ value.column }}
                    </span>
                </td>
            </tr>
            <tr>
                <td class='item-name'>Linking:</td>
                <div v-for="(sc, index) in linkings" v-bind:key="index">
                    <input class='input-index input-beg-index' type="text" v-model="sc.beg" /> - <input class='input-index input-end-index' type="text" v-model="sc.end" /> - <input class='input-entity' type="text" :data-index="index" :value="displayLinkings[index].text" readonly v-on:focus="linkEntity(index)" />
                    <svg class="icon" aria-hidden="true" v-if="displayLinkings[index].showClear" @click="clearEntity(index)"><use xlink:href="#icon-49shurushanchu-2"></use></svg>
                    <svg class="icon" aria-hidden="true" v-if="displayLinkings[index].showAdd" @click="addSlot"><use xlink:href="#icon-add_3"></use></svg>
                </div>
            </tr>
            <tr>
                <svg class="icon turn-action" aria-hidden="true" @click="removeTurn"><use xlink:href="#icon-49shurushanchu-2"></use></svg>
                <svg class="icon turn-action" aria-hidden="true" @click="addTurn"><use xlink:href="#icon-add_3"></use></svg>
            </tr>
        </table>
    </div>
</template>

<script>
import Vue from 'vue'
import Network from '../../network.js'
import '../../iconfont.js'

export default {
    name: "TranslateConversationBlock",
    props: {
        databaseId: String,
        questionId: String,
        initQuestion: String,
        initSql: String,
        initLinking: Array,
        initTokenizedQuestion: Array,
        initContextualPhenomena: Array,
        initTables: Array,
        initColumns: Array,
        initValues: Array
    },
    data() {
        return {
            focusedIndex: null,
            showErrorMessage: false,
            errorMessage: "",
            question: this.initQuestion,
            sql: this.initSql,
            linkings: Vue.util.extend([], this.initLinking),
            tokenizedChineseQuestion: Vue.util.extend([], this.initTokenizedQuestion),
            selectedContextualPhenomena: Vue.util.extend([], this.initContextualPhenomena),
            columns: Vue.util.extend([], this.initColumns),
            values: Vue.util.extend([], this.initValues),
            tables: Vue.util.extend([], this.initTables),
            data_details: new Array,
            data_description: new Array,
        }
    },
    computed: {
        displayLinkings: function(){
            // For styles
            var displayLinkings = new Array
            for(var i = 0; i < this.linkings.length; i++){
                if(this.linkings[i].entity && this.linkings[i].entity != null){
                    displayLinkings.push({text: "(" + this.linkings[i].type + ", " + this.linkings[i].entity.value + ")", showClear: true, showAdd: false})
                }else{
                    displayLinkings.push({text: "", showClear: false, showAdd: false})
                }
            }
            console.log(displayLinkings)
            if(displayLinkings.length > 0){
                displayLinkings[displayLinkings.length - 1].showAdd = true
            }
            return displayLinkings
        },
        tokenStyles: function() {
            // For token styles
            var styles = new Array
            for(var j = 0; j < this.tokenizedChineseQuestion.length; j++){
                styles.push({backgroundColor: 'white'})
            }
            for(var i = 0; i < this.linkings.length; i++){
                let begIndex = Number.parseInt(this.linkings[i].beg)
                let endIndex = Number.parseInt(this.linkings[i].end)
                if(!isNaN(begIndex) && begIndex >= 0 && begIndex < this.tokenizedChineseQuestion.length){
                    // BegIndex is a number
                    if(isNaN(endIndex) || endIndex < begIndex || endIndex >= this.tokenizedChineseQuestion.length){
                        // endIndex is not a number
                        // Show begIndex
                        styles[begIndex].backgroundColor = "yellow"
                    }else{
                        // Both begIndex and endIndex are a valid number
                        // Show begIndex to endIndex
                        for(let n = begIndex; n <= endIndex; n++){
                            styles[n].backgroundColor = "yellow"
                        }
                    }
                }else{
                    // BegIndex is not a number
                    if(!isNaN(endIndex) && endIndex >= 0 && endIndex < this.tokenizedChineseQuestion.length){
                        // Show endIndex
                        styles[endIndex].backgroundColor = "yellow"
                    }
                }
            }
            return styles
        },
        contextualPhenomenaStyles: function(){
            var options = ["Context Independent", "Coreference", "Ellipsis Continuation", "Ellipsis Substitution", "Far Side"]
            var styles = new Array
            for(var z = 0; z < options.length; z++){
                styles.push({
                    name: options[z],
                    selected: false,
                    style: {
                        color: "black",
                        backgroundColor: "white"
                    }
                })
            }
            for(var i = 0; i < this.selectedContextualPhenomena.length; i++){
                for(var j = 0; j < styles.length; j++){
                    if(styles[j].name === this.selectedContextualPhenomena[i] || (styles[j].name === "Coreference" && this.selectedContextualPhenomena[i].startsWith("Coreference"))){
                        styles[j].selected = true
                        styles[j].style.backgroundColor = "grey"
                        styles[j].style.color = "white"
                        break
                    }
                }
            }
            return styles
        },
        questionStyles: function(){
            var options = this.questionStyleOptions
            var styles = new Array
            for(var i = 0; i < options.length; i++){
                styles.push({
                    name: options[i],
                    selected: false,
                    style: {
                        color: "black",
                        backgroundColor: "white"
                    }
                })
            }
            for(var j = 0; j < styles.length; j++){
                if(styles[j].name === this.selectedQuestionStyle){
                    styles[j].selected = true
                    styles[j].style.backgroundColor = "grey"
                    styles[j].style.color = "white"
                    break
                }
            }
            return styles
        },
        pragmaticsStyles: function(){
            var options = this.pragmaticsOptions
            var styles = new Array
            for(var i = 0; i < options.length; i++){
                styles.push({
                    name: options[i],
                    selected: false,
                    style: {
                        color: "black",
                        backgroundColor: "white"
                    }
                })
            }
            for(var j = 0; j < styles.length; j++){
                if(styles[j].name === this.selectedPragmatics){
                    styles[j].selected = true
                    styles[j].style.backgroundColor = "grey"
                    styles[j].style.color = "white"
                    break
                }
            }
            return styles
        },
        intentStyles: function(){
            var options = this.intentOptions
            var styles = new Array
            for(var i = 0; i < options.length; i++){
                styles.push({
                    name: options[i],
                    selected: false,
                    style: {
                        color: "black",
                        backgroundColor: "white"
                    }
                })
            }
            for(var j = 0; j < styles.length; j++){
                if(styles[j].name === this.intent){
                    styles[j].selected = true
                    styles[j].style.backgroundColor = "grey"
                    styles[j].style.color = "white"
                    break
                }
            }
            return styles
        },
    },
    watch: {
        question: function(){
            var tokens = this.question.split('')
            this.tokenizedChineseQuestion = new Array
            var i = 0
            // tokenize chinese question
            while(i < tokens.length){
                if(escape(tokens[i]).indexOf("%u") === 0){
                    this.tokenizedChineseQuestion.push(tokens[i])
                    i += 1
                }else{
                    let newToken = ''
                    while(i < tokens.length){
                        if(escape(tokens[i]).indexOf("%u") === 0){
                            break
                        }
                        newToken += tokens[i]
                        i += 1
                    }
                    this.tokenizedChineseQuestion.push(newToken)
                }
            }
            if(this.linkings.length == 0){
                // Initialize linkings
                for(let i = 0; i < 5; i++){
                    this.linkings.push({beg: "", end: "", entity: null})
                }
            }
            this.$emit("updateAnnotation", "question", this.questionId, {question: this.question, tokenizedQuestion: this.tokenizedChineseQuestion})
        },
        sql: function(){
            // Parse SQL entities
            this.$emit("updateAnnotation", "sql", this.questionId, this.sql)
        },
        linkings: {
            handler: function(){
                var linkingResults = new Array;
                for(var i = 0; i < this.linkings.length; i++){
                    let begIndex = Number.parseInt(this.linkings[i].beg)
                    let endIndex = Number.parseInt(this.linkings[i].end)
                    let entityType = this.linkings[i].type
                    let entityValue = this.linkings[i].entity
                    if(!isNaN(begIndex) && begIndex >= 0 && begIndex < this.tokenizedChineseQuestion.length){
                        // BegIndex is a number
                        if(!isNaN(endIndex) && endIndex >= begIndex && endIndex < this.tokenizedChineseQuestion.length){
                            // EndIndex is a valid number
                            // Check whether entity value is valid
                            if(entityValue && entityValue != null){
                                // Has Value
                                linkingResults.push({
                                    beg: begIndex, end: endIndex,
                                    type: entityType, entity: entityValue
                                })
                            }
                        }
                    }
                }
                console.log("Watch Linkings: ", linkingResults)
                // Emit event
                if(linkingResults.length >= 0){
                    this.$emit("updateAnnotation", "linking", this.questionId, linkingResults)
                }
            },
            deep: true
        },
        selectedContextualPhenomena: function(){
            this.$emit("updateAnnotation", "phenomena", this.questionId, this.selectedContextualPhenomena)
        },
        selectedQuestionStyle: function(){
            this.$emit("updateAnnotation", "questionStyle", this.questionId, this.selectedQuestionStyle)
        },
        selectedPragmatics: function(){
            this.$emit("updateAnnotation", "pragmatics", this.questionId, this.selectedPragmatics)
        },
        intent: function(){
            this.$emit("updateAnnotation", "intent", this.questionId, this.intent)
        }
    },
    methods: {
        fill: function(event){
            event.preventDefault()
            var value = event.target.nextElementSibling.innerHTML
            // Fill in value to input
            this.value = value
        },
        linkEntity: function(index){
            this.focusedIndex = index
        },
        isValidSpan: function(){
            if(this.focusedIndex != null){
                // Has Valid Index
                var begIndex = Number.parseInt(this.linkings[this.focusedIndex].beg)
                var endIndex = Number.parseInt(this.linkings[this.focusedIndex].end)
                if(begIndex >= 0 && endIndex >= begIndex && endIndex < this.tokenizedChineseQuestion.length){
                    return true
                }else {
                    return false
                }
            }
            return false
        },
        togglePhenomenon: function(phenomenon){
            var isExists = false
            var index = 0
            for(var i = 0; i < this.selectedContextualPhenomena.length; i++){
                if(this.selectedContextualPhenomena[i] === phenomenon){
                    index = i
                    isExists = true
                    break
                }
            }
            if(isExists){
                // Remove
                this.selectedContextualPhenomena.splice(index, 1)
            }else{
                // Add it
                this.selectedContextualPhenomena.push(phenomenon)
            }
        },
        selectQuestionStyle: function(style){
            this.selectedQuestionStyle = style
        },
        selectPragmatics: function(style){
            this.selectedPragmatics = style
        },
        selectIntent: function(intent){
            this.intent = intent
        },
        clearEntity: function(index){
            this.linkings[index].entity = null
            this.linkings[index].type = ""
        },
        addSlot: function(){
            this.linkings.push({beg: "", end: "", entity: null})
        },
        selectEntity: function(entityType, index){
            if(this.isValidSpan()){
                var content = null;
                if(entityType === 'table'){
                    console.log("Select Table")
                    var tableInfo = this.tables[index]
                    content = {
                        tableId: tableInfo.tableId,
                        value: tableInfo.name
                    }
                }else if(entityType === 'column'){
                    console.log("Select Column")
                    var columnInfo = this.columns[index]
                    content = {
                        value: columnInfo.name,
                        columnId: columnInfo.columnId,
                    }
                }else if(entityType === 'value'){
                    console.log("Select Value")
                    var valueInfo = this.values[index]
                    content = {
                        value: valueInfo.value,
                        column: valueInfo.column,
                        columnId: valueInfo.columnId,
                    }
                }
                console.log(content)
                if(content != null){
                    // Save linking result
                    this.linkings[this.focusedIndex].type = entityType
                    this.linkings[this.focusedIndex].entity = content
                    this.focusedIndex = null
                }
            }
        },
        addTurn: function(){
            this.$emit("addTurn", this.questionId)
        },
        removeTurn: function(){
            this.$emit("removeTurn", this.questionId)
        },
        parseSQL: function(){
            if(this.sql !== ""){
                var payload = {
                    "sql": this.sql, "databaseId": this.databaseId
                }
                Network.parseSql(payload).then(response => {
                    var data = response.data
                    var status = response.status
                    if(status == 200){
                        if(data.msg === "success"){
                            this.showErrorMessage = false
                            this.data_details = data.data_details
                            this.data_description = data.data_description
                            this.tables = data.tables
                            this.columns = data.columns
                            this.values = data.values
                        }else{
                            this.showErrorMessage = true
                            this.errorMessage = data.msg
                            this.data_description = new Array
                            this.data_details = new Array
                        }
                    }
                })
            }
        }
    }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.translate-conversation-block {
    text-align: left;
}
table {
    border-spacing: 0 15px;
    margin: 0 0 0 20px;
    border-bottom: 1px solid;
}
td.item-name {
    text-align: right;
    /* font-weight: bold; */
    padding-right: 10px;
}
tr {
    font-size: 17px;
    margin: 0 0 13px 0;
}
.table-div {
    overflow: auto;
    max-height: 300px;
    max-width: 450px;
    display: inline-block;
}
.table-div table {
    border-collapse: collapse;
    border: 1px solid;
    table-layout: fixed;
    margin: 0 0 0 0;
}
.table-div td {
    border: 1px solid;
}
.table-div th {
    border: 1px solid;
    background-color: rgb(216, 216, 216);
}
.main-input-block {
    width: 100%;
    padding: 10px 10px 10px 10px;
    font-size: 17px;
    margin: 8px 0;
    box-sizing: border-box;
}
textarea.main-input-block {
    resize: vertical;
    font-family: Avenir, Helvetica, Arial, sans-serif;
}
.fill-in {
    cursor: pointer;
}
.fill-in:hover {
    color: blue;
}
span.phenomenon {
    display: inline-block;
    margin-top: 10px;
    margin-right: 10px;
    min-width: 50px;
    height: 40px;
    line-height: 40px;
    text-align: center;
    border-width: 1px 1px 1px 1px;
    border-style: solid;
    padding: 5px 5px 5px 5px;
    cursor: pointer;
}
span.phenomenon:hover {
    color: red;
}
span.entity {
    display: inline-block;
    margin-right: 10px;
    min-width: 50px;
    height: 40px;
    line-height: 40px;
    text-align: center;
    border-width: 1px 1px 1px 1px;
    border-style: solid;
    padding: 5px 5px 5px 5px;
    cursor: pointer;
}
span.entity:hover {
    color: blue;
}
.highlight {
    font-weight: bold;
    font-size: 20px;
}
span.token {
    display: inline-block;
    margin-right: 10px;
    width: 94.8px;
    min-height: 40px;
    line-height: 40px;
    text-align: center;
    border-width: 1px 1px 1px 1px;
    border-style: solid;
    padding: 5px 5px 5px 5px;
    margin-top: 5px;
}
span.token>input {
    border-width: 0px;
    border: none;
    box-shadow: none;
    margin: 0;
}
span.token>input:focus {
    outline:none!important;
}
.input-index {
    font-size: 17px;
    margin: 8px 0;
    padding: 2px 2px 2px 2px;
    width: 30px;
    height: 30px;
}
.input-entity {
    width: 200px;
    height: 30px;
    font-size: 17px;
    padding: 2px 2px 2px 2px;
}
.icon {
  width: 1.3em;
  height: 1.3em;
  vertical-align: -0.4em;
  fill: currentColor;
  overflow: hidden;
  cursor: pointer;
  margin-left: 20px;
}
.sql {
    background-color: #dadada;
}
.table-entity {
    margin-top: 5px;
}
.turn-action {
    fill: green;
}
.parse-button {
    width: 120px;
    height: 40px;
    margin-top: 5px;
}
.error-message {
    color: red;
}
</style>
