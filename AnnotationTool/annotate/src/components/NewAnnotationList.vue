<template>
    <div class="annotation-list">
        <table class="conversation-details">
            <caption>Annotated Conversations: {{total}}</caption>
            <thead>
                <tr>
                    <th>Conversation Id</th>
                    <th>Database Id</th>
                    <th>Created By</th>
                    <th>Revised By</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(a, index) in annotations" v-bind:key="index" @click="clickConverstaion(a.conversationId)">
                    <td>{{a.conversationId}}</td>
                    <td>{{a.databaseId}}</td>
                    <td>{{a.createdBy}}</td>
                    <td>{{a.revisedBy}}</td>
                </tr>
            </tbody>
        </table>
        <table class="user-details">
            <caption>User Statistics:</caption>
            <thead>
                <tr>
                    <th>User</th>
                    <th>Total</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(u, index) in users" v-bind:key="index">
                    <td>{{u.username}}</td>
                    <td>{{u.total}}</td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<script>
import Network from '../network.js'

export default {
    name: 'NewAnnotationList',
    beforeRouteLeave(to, from, next) {
        this.listScroll = document.documentElement.scrollTop || document.body.scrollTop
        console.log(this.listScroll)
        sessionStorage['listScroll'] = this.listScroll
        next()
    },
    data() {
        return {
            total: 0,
            annotations: [
                {
                    conversationId: "ex_0",
                    databaseId: "dorm",
                    createdBy: "",
                    revisedBy: ""
                }
            ],
            users: [
                {
                    username: "jiaqi",
                    total: 50,
                }
            ]
        }
    },
    mounted() {
        this.getList()
    },
    methods: {
        getList: function(){
            Network.getNewAnnotationList().then(response => {
                var data = response.data
                var status = response.status
                console.log(response)
                if(status == 200){
                    this.annotations = data.conversations
                    this.total = data.total
                    this.users = data.users
                }
                this.listScroll = sessionStorage['listScroll']
                this.$nextTick(() => {
                    document.body.scrollTop = this.listScroll
                    document.documentElement.scrollTop = this.listScroll
                })
            }).catch(error => {
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }
            })
        },
        clickConverstaion: function(cid){
            this.$router.push({name: "detailedConversation", params: {id: cid}})
        }
    },
}
</script>

<style scoped>
.annotation-list {
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