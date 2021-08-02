<template>
    <div class="table">
    <table>
        <tr>
            <td class="item-name">Table: </td>
            <td class="entity" @click="selectTable(tableName)"><strong> {{ tableName }} </strong></td>
        </tr>
        <tr>
            <td class="item-name">Columns: </td>
            <td class='column entity' v-for="(column, index) in columns" v-bind:key="index">
                <strong>{{ column.name }}</strong><br />{{ column.type }}
            <td/>
        </tr>
    </table>
    </div>
</template>


<script>
export default {
    name: 'Table',
    props: {
        tableName: String,
        columns: Array,
    },
    methods: {
        selectTable: function(tableName){
            this.$emit("selectTable", tableName)
        }
    },
}
</script>


<style scoped>
.item-name {
    text-align: right;
    /* font-weight: bold; */
    padding-right: 10px;
}
table {
    border-spacing: 10px 15px;
    margin: 0 0 0 20px;
}
tr {
    font-size: 17px;
    margin: 0 0 13px 0;
}
td.column {
    min-width: 50px;
    text-align: center;
    border-width: 1px 1px 1px 1px;
    border-style: solid;
    padding: 5px 5px 5px 5px;
}
.entity {
    cursor: pointer;
}
.entity:hover {
    color: blue;
}
</style>
