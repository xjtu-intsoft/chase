<template>
    <div class="flex-container">
      <Conversation :questions="questions" :conversationId="conversationId" :allowSubmit="true" :isAnnotated="isAnnotated" :problematic="problematic" 
                    :problematicNote="problematicNote" @updateNote="updateNote" @toggleProblematic="toggleProblematic" @submitAnnotation="submitAnntation" />
      <Database :database="database" :databaseId="databaseId" :showInputBlock="false"/>
    </div>
</template>

<script>
import Network from '../network.js'
import Conversation from './Conversation.vue'
import Database from './Database.vue'
import { EventBus } from '../event_bus.js'

export default {
  name: 'Annotation',
  components: {
    Conversation,
    Database
  },
  data() {
    return {
      isSubmitting: false,
      objectId: null,
      conversationId: "0",
      problematic: false,
      problematicNote: "",
      split: "train",
      isAnnotated: false,
      questions: [
        {
            questionId: "0",
            question: "How many dorms have a TV Lounge?",
            googleTranslation: "有多少宿舍带有电视休息室？",
            baiduTranslation: "有几个宿舍有电视休息室？",
            sql: "SELECT count(*) FROM dorm WHERE tv != NULL",
            tables: [
                {
                    name: "dorm",
                    tableId: 0
                }
            ],
            columns: [
                {
                    columnId: 1,
                    name: "id",
                    tableName: "dorm",
                    type: "string"
                },
                {
                    columnId: 2,
                    name: "name",
                    tableName: "dorm",
                    type: "string"
                },
                {
                    columnId: 3,
                    name: "built",
                    tableName: "dorm",
                    type: "number"
                },
                {
                    columnId: 3,
                    name: "capacity",
                    tableName: "dorm",
                    type: "number"
                }
            ],
            values: [
                {
                    value: "TV Lounge",
                    column: "name",
                    columnId: 1
                }
            ]
        },
      ],
      databaseId: "Dorms",
      database: [
        {
            tableName: "Dorm",
            columns: [
                {
                    name: "id",
                    type: "string"
                },
                {
                    name: "name",
                    type: "string"
                },
                {
                    name: "built",
                    type: "number"
                },
                {
                    name: "capacity",
                    type: "number"
                },
                {
                    name: "id",
                    type: "string"
                },
                {
                    name: "name",
                    type: "string"
                },
                {
                    name: "built",
                    type: "number"
                },
                {
                    name: "capacity",
                    type: "number"
                },
            ]
        },
        {
            tableName: "Dorm",
            columns: [
                {
                    name: "id12322222",
                    type: "string"
                },
                {
                    name: "name",
                    type: "string"
                },
                {
                    name: "built",
                    type: "number"
                },
                {
                    name: "capacity",
                    type: "number"
                }
            ]
        },
        {
            tableName: "Dorm",
            columns: [
                {
                    name: "id",
                    type: "string"
                },
                {
                    name: "name",
                    type: "string"
                },
                {
                    name: "built",
                    type: "number"
                },
                {
                    name: "capacity",
                    type: "number"
                }
            ]
        },
      ]
    }
  },
  computed: {
      specifiedConversationoId() {
        return this.$route.params.id
      }
  },
  watch: {
    '$route.params.id': function() {
        this.getAnnotation(this.$route.params.id)
    }
  },
  mounted() {
    if(this.specifiedConversationoId){
        console.log("Get annotation with id", this.specifiedConversationoId)
        this.getAnnotation(this.specifiedConversationoId)
    }else{
        this.getAnnotation()
    }
  },
  methods: {
    updateNote: function(note){
        this.problematicNote = note
    },
    toggleProblematic: function(){
        this.problematic = !this.problematic
    },
    getAnnotation: function(cid){
      Network.getAnnotation(cid).then(response => {
          var data = response.data
          var status = response.status
          console.log(response)
          if(status == 200 && data.databaseId != null){
            this.split = data.split
            this.questions = data.interaction
            this.databaseId = data.databaseId
            this.conversationId = data.exampleId
            this.database = data.database
            this.isAnnotated = data.isAnnotated
            this.problematic = data.problematic
            this.problematicNote = data.problematicNote
            if(this.isAnnotated){
                this.objectId = data.objectId
            }else{
                this.objectId = null
            }
            if(this.$route.params.id != this.conversationId){
                this.$router.push({name: "detail", params: {id: this.conversationId}})
            }
            document.body.scrollTop=0
            document.documentElement.scrollTop=0
          }
      }).catch(error => {
          if(error.response.status == 401){
              this.$router.replace({name: "login"})
          }
      })
    },
    getAnnotationByDatabaseId: function(dbId){
      Network.getAnnotationByDatabaseId(dbId).then(response => {
          var data = response.data
          var status = response.status
          console.log(response)
          if(status == 200){
              if(data.databaseId == null){
                // No available unannotated conversations for this database
                // Get a random one
                Network.getAnnotation().then(response => {
                    var data = response.data
                    var status = response.status
                    console.log(response)
                    if(status == 200 && data.databaseId != null){
                        this.split = data.split
                        this.questions = data.interaction
                        this.databaseId = data.databaseId
                        this.conversationId = data.exampleId
                        this.database = data.database
                        this.isAnnotated = data.isAnnotated
                        this.problematic = data.problematic
                        this.problematicNote = data.problematicNote
                        if(this.isAnnotated){
                            this.objectId = data.objectId
                        }else{
                            this.objectId = null
                        }
                        if(this.$route.params.id != this.conversationId){
                            this.$router.push({name: "detail", params: {id: this.conversationId}})
                        }
                    }
                })
              }else{
                this.split = data.split
                this.questions = data.interaction
                this.databaseId = data.databaseId
                this.conversationId = data.exampleId
                this.database = data.database
                this.isAnnotated = data.isAnnotated
                this.problematic = data.problematic
                this.problematicNote = data.problematicNote
                if(this.isAnnotated){
                    this.objectId = data.objectId
                }else{
                    this.objectId = null
                }
                if(this.$route.params.id != this.conversationId){
                    this.$router.push({name: "detail", params: {id: this.conversationId}})
                }
              }
          }
      }).catch(error => {
          if(error.response.status == 401){
              this.$router.replace({name: "login"})
          }
      })
    },
    submitAnntation: function(annotationResults){
      if(this.isSubmitting){
        return
      }
      this.isSubmitting = true
      var result = {
        objectId: this.objectId, // indicate udpate Annotation or new Annotation
        conversationId: this.conversationId,
        split: this.split,
        databaseId: this.databaseId,
        annotation: annotationResults,
        problematic: this.problematic,
        problematicNote: this.problematicNote
      }
      console.log(result)
      Network.postAnnotation(result).then(response => {
        console.log(response)
        EventBus.$emit("updateMeta");
        if(this.objectId == null){
            // Get next First by database id
            this.getAnnotationByDatabaseId(this.databaseId)
            this.isSubmitting = false
        }else{
            this.isSubmitting = false
            // this.$router.push({name: "list"})
        }
      }).catch(error => {
          this.isSubmitting = false
          if(error.response.status == 401){
              this.$router.replace({name: "login"})
          }else if(error.response.status == 403){
              this.$router.push({name: "list"})
          }
      })
    }
  },
}
</script>

<style>
.flex-container {
  display: flex;
  width: 95%;
  margin-left: auto;
  margin-right: auto;
}
</style>
