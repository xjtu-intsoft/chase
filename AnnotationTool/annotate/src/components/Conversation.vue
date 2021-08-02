<template>
    <div class="conversation">
        <h2>Conversation: {{conversationId}}</h2>
        <ConversationBlock v-for="q in questions" :key="q.questionId" :questionId="q.questionId" :question="q.question" :googleTranslation="q.googleTranslation" 
            :baiduTranslation="q.baiduTranslation" :sql="q.sql" :tables="q.tables" :columns="q.columns" :values="q.values" v-on:updateAnnotation='updateAnnotation'
            :initialLinkings="q.linking" :initialTokenizedChineseQuestion="q.tokenizedAnnotatedQuestion" :initialValue="q.annotatedQuestion"
            :initialPhenomena="q.contextualPhenomena" />
        <div class="conversation-operation">
            <span class="conversation-tag" @click="toggleProblematic" :style="problematicStyle">Problematic</span>
            <input class="conversation-note" type="text" name="note" :value="problematicNote" @change="updateNote" />
            <button type="button" v-if="allowSubmit" class="submit" @click="submitAnnotation">Submit</button>
        </div>
    </div>
</template>

<script>
import Vue from 'vue'
import ConversationBlock from './ConversationBlock.vue'

export default {
    name: 'Conversation',
    components: {
        ConversationBlock
    },
    props: {
        isAnnotated: Boolean,
        conversationId: String,
        questions: Array,
        allowSubmit: Boolean,
        problematic: Boolean,
        problematicNote: String,
    },
    data() {
        return {
            annotationResults: new Array
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
        questions: function(){
            console.log("Initialization", this.isAnnotated)
            this.annotationResults = {};
            for(let i = 0; i < this.questions.length; i++){
                let turn = this.questions[i];
                if(this.isAnnotated){
                    this.annotationResults[turn.questionId] = {
                        annotatedQuestion: turn.annotatedQuestion,
                        tokenizedAnnotatedQuestion: Vue.util.extend([], turn.tokenizedAnnotatedQuestion),
                        linking: Vue.util.extend([], turn.linking),
                        contextualPhenomena: Vue.util.extend([], turn.contextualPhenomena),
                    }
                }else{
                    this.annotationResults[turn.questionId] = {}
                }
            }
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
        updateAnnotation: function(type, questionId, results){
            if(type === 'question'){
                this.annotationResults[questionId]['annotatedQuestion'] = results.annotatedQuestion
                this.annotationResults[questionId]['tokenizedAnnotatedQuestion'] = results.tokenizedAnnotatedQuestion
            }else if(type === 'linking'){
                this.annotationResults[questionId]['linking'] = results
            }else if(type === 'phenomena'){
                this.annotationResults[questionId]['contextualPhenomena'] = results
            }
        },
        submitAnnotation: function(){
            // TODO: Check whether linking is valid
            console.log("Submit")
            var processedAnnotationResults = new Array
            var isValid = true
            for(let i = 0; i < this.questions.length; i++){
                let turn = this.questions[i];
                let tokens = new Array;
                if(!this.annotationResults[turn.questionId]['tokenizedAnnotatedQuestion']){
                    isValid = false
                    break
                }
                for(let j = 0; j < this.annotationResults[turn.questionId]['tokenizedAnnotatedQuestion'].length; j++){
                    tokens.push(this.annotationResults[turn.questionId]['tokenizedAnnotatedQuestion'][j])
                }
                let turnAnnotation = {
                    questionId: turn.questionId,
                    englishQuestion: turn.question,
                    googleTranslation: turn.googleTranslation,
                    baiduTranslation: turn.baiduTranslation,
                    sql: turn.sql,
                    annotatedQuestion: this.annotationResults[turn.questionId]['annotatedQuestion'],
                    linking: this.annotationResults[turn.questionId]['linking'],
                    contextualPhenomena: this.annotationResults[turn.questionId]['contextualPhenomena'],
                    tokenizedAnnotatedQuestion: tokens
                }
                // Check is valid
                if(turnAnnotation.annotatedQuestion.length == 0 || !turnAnnotation.contextualPhenomena || turnAnnotation.contextualPhenomena.length == 0){
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
    }
}
</script>

<style scoped>
.conversation {
    width: 45%;
}
h2 {
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
