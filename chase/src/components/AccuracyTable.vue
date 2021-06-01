<template>
    <div class="accuracy-table">
        <el-table :data="tableData" style="width: 100%">
            <el-table-column type="expand" :default-sort = "{prop: 'rank', order: 'descending'}">
                <template slot-scope="props">
                    <el-form label-position="left" inline class="table-expand">
                        <el-form-item label="Model">
                            <span>{{ props.row.model }}</span>
                        </el-form-item>
                        <el-form-item label="Affliation">
                            <span>{{ props.row.affliation }}</span>
                        </el-form-item>
                        <el-form-item label="SQL Hardness (QM)">
                            <span class="score-span"><b>Easy</b>: {{ props.row.hardness.easy }} </span>
                            <span class="score-span"><b>Medium</b>: {{ props.row.hardness.medium }} </span>
                            <span class="score-span"><b>Hard</b>: {{ props.row.hardness.hard }} </span>
                            <span><b>Extra Hard</b>: {{ props.row.hardness.extra }} </span>
                        </el-form-item>
                        <el-form-item label="Contextual Dependency (QM)">
                            <span class="score-span"><b>Independent</b>: {{ props.row.context.independent }} </span>
                            <span class="score-span"><b>Dependent</b>: {{ props.row.context.dependent }} </span>
                            <span class="score-span"><b>Coreference</b>: {{ props.row.context.coreference }} </span>
                            <span class="score-span"><b>Ellipsis</b>: {{ props.row.context.ellipsis }} </span>
                        </el-form-item>
                        <el-form-item label="Dev Question Match">
                            <span>{{ props.row.devQM }}</span>
                        </el-form-item>
                        <el-form-item label="Dev Interaction Match">
                            <span>{{ props.row.devIM }}</span>
                        </el-form-item>
                        <el-form-item label="Dev SQL Hardness (QM)">
                            <span class="score-span"><b>Easy</b>: {{ props.row.devHardness.easy }} </span>
                            <span class="score-span"><b>Medium</b>: {{ props.row.devHardness.medium }} </span>
                            <span class="score-span"><b>Hard</b>: {{ props.row.devHardness.hard }} </span>
                            <span><b>Extra Hard</b>: {{ props.row.devHardness.extra }} </span>
                        </el-form-item>
                        <el-form-item label="Dev Contextual Dependency (QM)">
                            <span class="score-span"><b>Independent</b>: {{ props.row.devContext.independent }} </span>
                            <span class="score-span"><b>Dependent</b>: {{ props.row.devContext.dependent }} </span>
                            <span class="score-span"><b>Coreference</b>: {{ props.row.devContext.coreference }} </span>
                            <span><b>Ellipsis</b>: {{ props.row.devContext.ellipsis }} </span>
                        </el-form-item>
                        <el-form-item label="Paper">
                            <a :href=props.row.paper>{{ props.row.paper }}</a>
                        </el-form-item>
                        <el-form-item label="Code">
                            <a :href=props.row.code>{{ props.row.code }}</a>
                        </el-form-item>
                    </el-form>
                </template>
            </el-table-column>
            <el-table-column label="Rank" prop="rank" sortable width="80"></el-table-column>
            <el-table-column label="Model" prop="model"></el-table-column>
            <el-table-column label="Pre-train LM" prop="pretrainLM"></el-table-column>
            <el-table-column label="Ensemble" prop="ensemble" width="90"></el-table-column>
            <el-table-column label="Question Match" prop="QM" sortable></el-table-column>
            <el-table-column label="Interaction Match" prop="IM" sortable></el-table-column>
            <el-table-column label="Date" prop="date" sortable width="100"></el-table-column>
        </el-table>
    </div>
</template>

<script>
export default {
  name: 'AccuracyTable',
  props: ["tableData"]
}
</script>

<style>
.table-expand {
font-size: 0;
}
.table-expand label {
color: #99a9bf;
}
.table-expand .el-form-item {
margin-right: 0;
margin-bottom: 0;
width: 80%;
}
.score-span {
    margin-right: 10px;
}
</style>