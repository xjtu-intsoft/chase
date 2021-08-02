<template>
    <div class="scrollable-block translate-conversation">
        <h2>Conversation: {{conversationId}}</h2>
        <h3>Minimum Turns: {{minimumTurn}}</h3>
        <TranslateConversationBlock v-for="c in annotationResults" v-bind:key="c.questionId" :databaseId="databaseId"
            :questionId="c.questionId" :initQuestion="c.question" :initSql="c.sql" :initLinking="c.linking" 
            :initTokenizedQuestion="c.tokenizedQuestion" :initContextualPhenomena="c.contextualPhenomena"
            :initTables="c.tables" :initColumns="c.columns" :initValues="c.values"
            @updateAnnotation='updateAnnotation' @addTurn='addTurnById' @removeTurn='removeTurnById' />
        <div class="conversation-operation">
            <span class="conversation-tag" @click="toggleProblematic" :style="problematicStyle">Problematic</span>
            <input class="conversation-note" type="text" name="note" :value="problematicNote" @change="updateNote" />
            <button type="button" class="submit" @click="submitAnnotation">Submit</button>
            <button type="button" class="submit" @click="deleteAnnotation">Delete</button>
        </div>
    </div>
</template>

<script>
import Vue from 'vue'
import Utils from '../../utils.js'
import TranslateConversationBlock from './TranslateConversationBlock.vue'

export default {
    name: 'TranslateConversation',
    components: {
        TranslateConversationBlock
    },
    props: {
        databaseId: String,
        conversationId: String,
        conversations: Array,
        problematic: Boolean,
        problematicNote: String,
    },
    data() {
        return {
            annotationResults: new Array,
            minimumTurn: this.sampleMinimumTurn(this.conversations)
        }
    },
    computed: {
        problematicStyle: function(){
            if(this.problematic){
                return {
                    color: "white",
                    backgroundColor: "grey"
                }
            }else{
                return {
                    color: "black",
                    backgroundColor: "white"
                }
            }
        },
        note: function(){
            return this.problematicNote
        }
    },
    watch: {
        conversations: {
            handler: function(){
                this.annotationResults = new Array
                for(let i = 0; i < this.conversations.length; i++){
                    let turn = this.conversations[i];
                    this.annotationResults.push({
                        questionId: turn.questionId,
                        question: turn.question,
                        sql: turn.sql,
                        tokenizedQuestion: Vue.util.extend([], turn.tokenizedQuestion),
                        linking: Utils.copy(turn.linking),
                        contextualPhenomena: Vue.util.extend([], turn.contextualPhenomena),
                        tables: Vue.util.extend([], turn.tables),
                        columns: Vue.util.extend([], turn.columns),
                        values: Vue.util.extend([], turn.values),
                    })
                }
                if(this.annotationResults.length === 0){
                    this.addTurn(0)
                }
                this.minimumTurn = this.sampleMinimumTurn(this.conversations)
            },
            deep: true
        }
    },
    methods: {
        toggleProblematic: function(){
            this.$emit("toggleProblematic")
        },
        updateNote: function(event){
            this.$emit("updateNote", event.target.value)
            if(event.target.value.length > 0 && !this.problematic){
                this.toggleProblematic()
            }else if(event.target.value.length == 0 && this.problematic){
                this.toggleProblematic()
            }
        },
        sampleMinimumTurn: function(conversations){
            if(conversations.length > 0){
                return conversations.length
            }
            return 3 + Math.floor(Math.random() * Math.floor(3));
        },
        addTurn: function(index){
            var template = {
                questionId: Math.random().toString(36).substr(2, 5), // Get a random String
                question: "",
                tokenizedQuestion: new Array,
                linking: new Array,
                sql: "",
                contextualPhenomena: new Array,
                tables: new Array,
                columns: new Array,
                values: new Array,
            }
            this.annotationResults.splice(index, 0, Vue.util.extend({}, template))
        },
        removeTurn: function(index){
            this.annotationResults.splice(index, 1)
        },
        addTurnById: function(questionId){
            var targetIndex = -1
            for(var i = 0; i < this.annotationResults.length; i++){
                if(this.annotationResults[i].questionId === questionId){
                    targetIndex = i
                    break
                }
            }
            if(targetIndex >= 0){
                this.addTurn(targetIndex+1)
            }
        },
        removeTurnById: function(questionId){
            if(this.annotationResults.length <= 1){
                return
            }
            var targetIndex = -1
            for(var i = 0; i < this.annotationResults.length; i++){
                if(this.annotationResults[i].questionId === questionId){
                    targetIndex = i
                    break
                }
            }
            if(targetIndex >= 0){
                this.removeTurn(targetIndex)
            }
        },
        updateAnnotation: function(type, questionId, results){
            var target = null
            for(var i = 0; i < this.annotationResults.length; i++){
                if(this.annotationResults[i].questionId === questionId){
                    target = this.annotationResults[i]
                    break
                }
            }
            if(target != null){
                if(type === 'question'){
                    target.question = results.question
                    target.tokenizedQuestion = results.tokenizedQuestion
                }else if(type === 'linking'){
                    target.linking = results
                }else if(type === 'phenomena'){
                    target.contextualPhenomena = results
                }else if(type === 'sql'){
                    target.sql = results
                }
            }
        },
        deleteAnnotation: function(){
            this.$emit("deleteAnnotation")
        },
        submitAnnotation: function(){
            console.log("Submit")
            if(this.annotationResults.length <= 1 || this.annotationResults.length < this.minimumTurn){
                return
            }
            var processedAnnotationResults = new Array
            var isValid = true
            for(let i = 0; i < this.annotationResults.length; i++){
                let turn = this.annotationResults[i];
                let tokens = new Array;
                if(turn.tokenizedQuestion.length == 0){
                    isValid = false
                    break
                }
                for(let j = 0; j < turn.tokenizedQuestion.length; j++){
                    tokens.push(turn.tokenizedQuestion[j])
                }
                let turnAnnotation = {
                    questionId: turn.questionId,
                    question: turn.question,
                    sql: turn.sql,
                    linking: turn.linking,
                    contextualPhenomena: turn.contextualPhenomena,
                    tokenizedQuestion: tokens
                }
                // Check is valid
                if(turnAnnotation.question.length == 0 || turnAnnotation.sql.length == 0 || !turnAnnotation.contextualPhenomena || turnAnnotation.contextualPhenomena.length == 0){
                    isValid = false
                    break
                }
                if(!turnAnnotation.linking){
                    turnAnnotation.linking = new Array
                }
                processedAnnotationResults.push(turnAnnotation)
            }
            console.log(isValid)
            if(isValid){
                this.$emit("submitAnnotation", processedAnnotationResults)
            }else{
                console.error(this.annotationResults)
            }
        }
    },
}
</script>

<style scoped>
.translate-conversation {
    overflow: auto;
    width: 50%;
    height: 900px;
}
h2 {
    margin: 30px 0 15px 0;
    text-align: center;
}
h3 {
    margin: 30px 0 15px 0;
    text-align: center;
}
.conversation-operation {
    margin-top: 15px;
    margin-bottom: 10px;
}
.conversation-tag {
    display: inline-block;
    margin-left: 20px;
    width: 100px;
    height: 30px;
    line-height: 30px;
    text-align: center;
    border-width: 1px 1px 1px 1px;
    border-style: solid;
    border-radius: 6px;
    padding: 5px 5px 5px 5px;
    cursor: pointer;
}
.conversation-tag:hover {
    color: white;
    background-color: grey;
}
.conversation-note {
    width: 300px;
    height: 30px;
    font-size: 17px;
    margin-left: 20px;
    padding: 2px 2px 2px 2px;
}
.submit {
    width: 100px;
    height: 40px;
    margin-left: 20px;
    margin-top: 5px;
}
</style>
