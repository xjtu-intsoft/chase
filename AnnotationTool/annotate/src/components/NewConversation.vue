<template>
    <div class="scrollable-block new-conversation">
        <h2>Conversation: {{conversationId}}</h2>
        <h3>Minimum Turns: {{minimumTurn}}</h3>
        <NewConversationBlock v-for="c in annotationResults" v-bind:key="c.questionId" :databaseId="databaseId"
            :questionId="c.questionId" :initQuestion="c.question" :initSql="c.sql" :initLinking="c.linking" 
            :initTokenizedQuestion="c.tokenizedQuestion" :initContextualPhenomena="c.contextualPhenomena"
            :initTables="c.tables" :initColumns="c.columns" :initValues="c.values" :initQuestionStyle="c.questionStyle"
            :initPragmatics="c.pragmatics" :initIntent="c.intent" :questionIndex="c.questionIndex"
            :intentOptions="intentOptions" :pragmaticsOptions="pragmaticsOptions" :questionStyleOptions="questionStyleOptions"
            @updateAnnotation='updateAnnotation' @addTurn='addTurnById' @removeTurn='removeTurnById' />
        <div class="conversation-operation">
            <button type="button" class="submit" @click="submitAnnotation">Submit</button>
        </div>
    </div>
</template>

<script>
import Vue from 'vue'
import Utils from '../utils.js'
import NewConversationBlock from './NewConversationBlock.vue'

export default {
    name: 'NewConversation',
    components: {
        NewConversationBlock
    },
    props: {
        databaseId: String,
        conversationId: String,
        conversations: Array,
    },
    data() {
        return {
            questionStyleOptions: ["陈述句", "关键词", "疑问句"],
            pragmaticsOptions: ["主动", "被动", "同义表达", "名词->动词"],
            intentOptions: ['相同实体实例的属性', '相同实体类型的不同实例的属性', '不同实体类型的实例', '修改对结果的展示形式', '其他'],
            annotationResults: new Array,
            minimumTurn: this.sampleMinimumTurn(this.conversations)
        }
    },
    watch: {
        conversations: {
            handler: function(){
                console.log("Update Conversations")
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
                        questionStyle: turn.questionStyle,
                        pragmatics: turn.pragmatics,
                        intent: turn.intent,
                        data_description: new Array,
                    })
                }
                if(this.annotationResults.length === 0){
                    this.addTurn(0)
                }
                this.minimumTurn = this.sampleMinimumTurn(this.conversations)
                this.updateQuestionIndex()
            },
            deep: true
        }
    },
    methods: {
        sampleMinimumTurn: function(conversations){
            if(conversations.length > 0){
                return conversations.length
            }
            return 3 + Math.floor(Math.random() * Math.floor(3));
        },
        addTurn: function(index){
            var sampledQuestionStyle = Utils.sample(this.questionStyleOptions)
            var sampledPragmatics = Utils.sample(this.pragmaticsOptions)
            var sampledIntents = Utils.sample(this.intentOptions)
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
                // Random Assign
                questionStyle: sampledQuestionStyle,
                pragmatics: sampledPragmatics,
                intent: sampledIntents,
                data_description: new Array,
            }
            if(index == 0){
                // The first question
                template.intent = "其他"
                template.contextualPhenomena = ["Context Independent"]
            }
            this.annotationResults.splice(index, 0, Vue.util.extend({}, template))
            this.updateQuestionIndex()
        },
        removeTurn: function(index){
            this.annotationResults.splice(index, 1)
            this.updateQuestionIndex()
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
            this.updateQuestionIndex()
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
            this.updateQuestionIndex()
        },
        updateQuestionIndex: function(){
            for(var i = 0; i < this.annotationResults.length; i++){
                this.annotationResults[i].questionIndex = i + 1
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
                    target.data_description = new Array
                }else if(type === 'questionStyle'){
                    target.questionStyle = results
                }else if(type === 'pragmatics'){
                    target.pragmatics = results
                }else if(type === 'intent'){
                    target.intent = results
                }else if(type === 'data_description'){
                    target.data_description = results
                }
            }
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
                if(turn.tokenizedQuestion.length == 0 || turn.data_description.length == 0){
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
                    tokenizedQuestion: tokens,
                    questionStyle: turn.questionStyle,
                    pragmatics: turn.pragmatics,
                    intent: turn.intent
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
.new-conversation {
    overflow: auto;
    width: 50%;
    height: 940px;
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
