<template>
    <div class="translate-conversation-list">
        <table class="conversation-details">
            <caption>{{ databaseId }} Conversations: {{total}}</caption>
            <thead>
                <tr>
                    <th>Conversation Id</th>
                    <th>Deleted</th>
                    <th>Created By</th>
                    <th>Revised By</th>
                    <th>Problematic Note</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(a, index) in conversations" v-bind:key="index" @click="clickConverstaion(a.conversationId)" :style="conversationStyles[index]">
                    <td>{{a.conversationId}}</td>
                    <td>{{a.isDeleted}}</td>
                    <td>{{a.createdBy}}</td>
                    <td>{{a.revisedBy}}</td>
                    <td>{{a.problematicNote}}</td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<script>
import Network from '../../network.js'

export default {
    name: 'TranslateConversationList',
    data() {
        return {
            total: 0,
            databaseId: 0,
            conversations: [
                {
                    conversationId: "ex_0",
                    problematic: false,
                    problematicNote: "",
                    isAnnotated: true,
                    createdBy: "",
                    revisedBy: "",
                    isDeleted: false
                }
            ]
        }
    },
    computed: {
        conversationStyles: function(){
            var styles = new Array
            for(var i = 0; i < this.conversations.length; i++){
                if(this.conversations[i].isDeleted){
                    styles.push({backgroundColor: "yellow"})
                }else if(this.conversations[i].problematic){
                    styles.push({backgroundColor: "red"})
                }else{
                    styles.push({})
                }
            }
            return styles
        }
    },
    mounted() {
        this.databaseId = this.$route.params.id
        this.getList(this.databaseId)
    },
    methods: {
        getList: function(databaseId){
            Network.getTranslateConversationList(databaseId).then(response => {
                var data = response.data
                var status = response.status
                console.log(response)
                if(status == 200){
                    this.conversations = data.conversations
                    this.total = data.total
                }
            }).catch(error => {
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }
            })
        },
        clickConverstaion: function(cid){
            this.$router.push({name: "translateAnnotation", params: {id: cid}})
        }
    },
}
</script>

<style scoped>
.translate-conversation-list {
    display: flex;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
}
.conversation-details {
    width: 600px;
    margin-top: 30px;
    margin-left: 40px;
    font-size: 17px;
    text-align: left;
    border-collapse: collapse;
}
table caption {
    text-align: left;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 20px;
}
table tr {
    height: 35px;
    cursor: pointer;
    border-top: 1px solid black;
}
table thead tr {
    height: 35px;
    cursor: pointer;
    border-bottom: 1px solid black;
}
tr:first-of-type {
    border-top: none;
}
tr:last-of-type {
    border-bottom: 1px solid black;
}
.conversation-details tbody tr:hover{
    color: white;
    background-color: grey;
}
.user-details {
    width: 300px;
    margin-top: 30px;
    margin-left: 50px;
    font-size: 17px;
    text-align: left;
    border-collapse: collapse;
}
</style>