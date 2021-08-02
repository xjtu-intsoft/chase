<template>
    <div class="meta-bar">
        <span>
            <strong @click="browseAnnotationList" class="browseable">Remaining SparC Conversations</strong>: {{ remainingConversation }};
        </span>
        <span>
            <strong @click="browseTranslateDatabaseList" class="browseable">Remaining SparC Databases</strong>: {{remainingDatabase}}
        </span>
        <span>
            <strong>Remaining SparC Conversations to Refine</strong>: {{remainingSparctoRefine}}
        </span>
        <span>
            <strong @click="browseConversationList" class="browseable">New Conversations</strong>: {{newConversation}}
        </span>
        <span>
            <strong @click="browseDuSQLDatabaseList" class="browseable">To DuSQL Databases</strong>
        </span>
    </div>
</template>>

<script>
export default {
    name: 'MetaBar',
    props: {
        remainingConversation: Number,
        remainingDatabase: Number,
        newConversation: Number,
        remainingSparctoRefine: Number
    },
    methods: {
        browseAnnotationList: function(){
            this.$router.push({name: "list"})
        },
        browseTranslateDatabaseList: function(){
            this.$router.push({name: "translateDatabases"})
        },
        browseConversationList: function(){
            this.$router.push({name: "newConversationList"})
        },
        browseDuSQLDatabaseList: function(){
            this.$router.push({name: "databases"})
        }
    },
}
</script>

<style scoped>
.meta-bar {
    margin: -20px 0 0 0;
}
span {
  margin: 0 40px 0 40px;
}
.browseable {
    cursor: pointer;
}
.browseable:hover {
    color: blue;
}
</style>
