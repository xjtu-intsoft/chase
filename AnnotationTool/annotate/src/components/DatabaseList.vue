<template>
    <div class="database-list">
        <table class="database-details">
            <caption>Valid Databases: {{databaseNum}}; Total Valid Conversations: {{conversationNum}}</caption>
            <thead>
                <tr>
                    <th>Database Id</th>
                    <th># Conversations</th>
                    <th>{{username}} # Conversations</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(d, index) in databases" v-bind:key="index" @click="clickDatabase(d.databaseId)">
                    <td>{{d.databaseId}}</td>
                    <td>{{d.conversationNum}}</td>
                    <td>{{d.userConversationNum}}</td>
                </tr>
            </tbody>
        </table>
        <table class="database-details">
            <p><a href="/api/dusql/">Download DuSQL data</a></p>
            <p><a href="/api/dusql_tables/">Download DuSQL Tables</a></p>
        </table>
    </div>
</template>

<script>
import Network from '../network.js'

export default {
    name: "DatabaseList",
    data() {
        return {
            databaseNum: 0,
            conversationNum: 0,
            username: "jiaqi",
            databases: [
                {
                    databaseId: "flight_1",
                    conversationNum: 0,
                    userConversationNum: 0
                }
            ]
        }
    },
    mounted() {
        this.getList()
    },
    methods: {
        getList: function(){
            Network.getDatabaseList().then(response => {
                var data = response.data
                this.username = data.username
                this.databases = data.results
                this.databaseNum = this.databases.length
                var num = 0
                for(var i = 0; i < this.databases.length; i++){
                    num += this.databases[i].conversationNum
                }
                this.conversationNum = num
            }).catch(error => {
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }
            })
        },
        clickDatabase: function(databaseId){
            this.$router.push({name: "newConversation", params: {id: databaseId}})
        }
    },
}
</script>

<style scoped>
.database-list {
    display: flex;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
}
.database-details {
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
.database-details tbody tr:hover{
    color: white;
    background-color: grey;
}
</style>