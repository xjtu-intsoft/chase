<template>
    <div class="flex-container">
      <TranslateConversation :databaseId="databaseId" :conversationId="formattedConversationId" :conversations="conversations"
        @submitAnnotation="submitAnnotation" @deleteAnnotation="deleteAnnotation"
        :problematic="problematic" :problematicNote="problematicNote" @updateNote="updateNote" @toggleProblematic="toggleProblematic" />
      <Database :database="database" :databaseGraph="databaseGraph" :databaseId="databaseId" :showInputBlock="true" />
    </div>
</template>

<script>
import Database from '../Database.vue'
import TranslateConversation from './TranslateConversation.vue'
import Network from '../../network.js'
import { EventBus } from '../../event_bus.js'

export default {
    name: "TranslateAnnotation",
    components: {
        Database,
        TranslateConversation
    },
    data() {
        return {
            conversationId: null,
            databaseId: "",
            database: [],
            conversations: [],
            isSubmitting: false,
            databaseGraph: null,
            problematic: false,
            problematicNote: ""
        }
    },
    computed: {
        formattedConversationId: function(){
            if(this.conversationId != null){
                return this.conversationId
            }
            return ""
        }
    },
    mounted() {
        if(this.$route.name === 'translateAnnotation'){
            this.conversationId = this.$route.params.id
            this.getConversationById(this.conversationId)
        }
    },
    methods: {
        updateNote: function(note){
            this.problematicNote = note
        },
        toggleProblematic: function(){
            this.problematic = !this.problematic
        },
        getDatabase: function(databaseId){
            Network.getTranslatedDatabaseByid(databaseId).then(response => {
                var data = response.data
                var status = response.status
                if(status == 200){
                    this.database = data.entities
                    this.databaseGraph = data.graph
                }
            }).catch(error => {
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }
            })
        },
        getConversationById: function(conversationId){
            Network.getTranslateConversationbyId(conversationId).then(response => {
                var data = response.data
                var status = response.status
                console.log(response)
                if(status == 200 && data.databaseId != null){
                    this.databaseId = data.databaseId
                    this.conversationId = conversationId
                    this.conversations = data.conversations
                    this.problematic = data.problematic
                    this.problematicNote = data.problematicNote
                    this.getDatabase(this.databaseId)
                }
            }).catch(error => {
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }else if(error.response.status == 403){
                    this.$router.push({name: "list"})
                }
            })
        },
        deleteAnnotation: function(){
            if(this.isSubmitting){
                return
            }
            this.isSubmitting = true
            Network.deleteTranslateConversation(this.conversationId).then(response => {
                console.log(response)
                this.isSubmitting = false
                EventBus.$emit("updateMeta")
                this.$router.push({name: "translateConversations", params: {id: this.databaseId}})
            }).catch(error => {
                this.isSubmitting = false
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }
            })
        },
        submitAnnotation: function(annotationResults){
            if(this.isSubmitting){
                return
            }
            this.isSubmitting = true
            var result = {
                conversationId: this.conversationId,
                databaseId: this.databaseId,
                annotation: annotationResults,
                problematic: this.problematic,
                problematicNote: this.problematicNote
            }
            console.log(result)
            Network.submitTranslateConversation(result).then(response => {
                console.log(response)
                this.isSubmitting = false
                EventBus.$emit("updateMeta");
                this.$router.push({name: "translateConversations", params: {id: this.databaseId}})
            }).catch(error => {
                this.isSubmitting = false
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }else if(error.response.status == 403){
                    this.$router.push({name: "list"})
                }
            })
        },
    },
}
</script>


<style scoped>
.flex-container {
  display: flex;
  width: 98%;
  margin-left: auto;
  margin-right: auto;
}
</style>