<template>
    <div class="database-list">
        <table class="database-details">
            <caption>Total Databases: {{total}}; Remaining Databases: {{untranslated}}. </caption>
            <thead>
                <tr>
                    <th>Database Id</th>
                    <th>Translator</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(d, index) in databases" v-bind:key="index" @click="clickDatabase(d.databaseId)">
                    <td>{{d.databaseId}}</td>
                    <td>{{d.translator}}</td>
                </tr>
            </tbody>
        </table>
        <table class="database-details">
            <p><a href="/api/sparc_refined/">Download SparC data</a></p>
            <p><a href="/api/sparc_tables/">Download SparC Tables</a></p>
            <p><a href="">Download SparC Databases</a></p>
        </table>
    </div>
</template>

<script>
import Network from '../../network.js'

export default {
    name: "TranslateDatabaseList",
    data() {
        return {
            total: 0,
            untranslated: 0,
            databases: [
                {
                    databaseId: "flight_1",
                    translator: "",
                }
            ]
        }
    },
    mounted() {
        this.getList()
    },
    methods: {
        getList: function(){
            Network.getTranslateDatabaseList().then(response => {
                var data = response.data
                this.databases = data
                this.total = this.databases.length
                var num = 0
                for(var i = 0; i < this.databases.length; i++){
                    if(this.databases[i].translator === ""){
                        num += 1
                    }
                }
                this.untranslated = num
            }).catch(error => {
                if(error.response.status == 401){
                    this.$router.replace({name: "login"})
                }
            })
        },
        clickDatabase: function(databaseId){
            this.$router.push({name: "translate", params: {id: databaseId}})
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