<template>
    <div class="flex-container">
      <NewConversation :databaseId="databaseId" :conversationId="formattedConversationId" :conversations="conversations"
        @submitAnnotation="submitAnnotation" />
      <Database :database="database" :databaseGraph="databaseGraph" :databaseId="databaseId" :showInputBlock="true" />
    </div>
</template>

<script>
import Database from './Database.vue'
import NewConversation from './NewConversation.vue'
import Network from '../network.js'
import { EventBus } from '../event_bus.js'

export default {
    name: "NewAnnotation",
    components: {
        Database,
        NewConversation
    },
    data() {
        return {
            conversationId: null,
            databaseId: "",
            database: [],
            conversations: [],
            isSubmitting: false,
            databaseGraph: null
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
        if(this.$route.name === 'newConversation'){
            this.databaseId = this.$route.params.id
            this.getDatabase(this.databaseId)
            this.conversationId = null
            this.conversations = new Array
        }else if(this.$route.name === 'detailedConversation'){
            console.log("Revise")
            this.conversationId = this.$route.params.id
            this.getConversationById(this.conversationId)
        }
    },
    watch: {
        '$route.params.id': function() {
            if(this.$route.name === 'newConversation'){
                // change database
                this.databaseId = this.$route.params.id
                this.getDatabase(this.databaseId)
                this.conversationId = null
                this.conversations = new Array
            }else{
                // change conversation
                console.log("Revise")
                this.conversationId = this.$route.params.id
            }
        }
    },
    methods: {
        getDatabase: function(databaseId){
            Network.getDatabase(databaseId).then(response => {
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
            Network.getConversationbyId(conversationId).then(response => {
                var data = response.data
                var status = response.status
                console.log(response)
                if(status == 200 && data.databaseId != null){
                    this.databaseId = data.databaseId
                    this.conversationId = conversationId
                    this.conversations = data.conversations
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
        submitAnnotation: function(annotationResults){
            var result = {
                conversationId: this.conversationId,
                databaseId: this.databaseId,
                annotation: annotationResults
            }
            console.log(result)
            if(this.isSubmitting){
                return
            }
            this.isSubmitting = true
            Network.submitNewConversation(result).then(response => {
                console.log(response)
                this.isSubmitting = false
                EventBus.$emit("updateMeta");
                if(this.conversationId == null){
                    // Clear
                    this.conversations = new Array
                }else{
                    this.conversationId = null
                    this.conversations = new Array
                    this.$router.push({name: "newConversation", params: {id: this.databaseId}})
                }
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