<template>
  <div id="app">
    <Banner />
    <div id="content">
      <row container :gutter="40">
        <column :xs="12" :md="4" :lg="4">
          <Aside />
        </column>
        <column :xs="12" :md="8" :lg="8">
          <MainLeaderboard />
        </column>
      </row>
    </div>
  </div>
</template>

<script>
import Banner from './components/Banner.vue'
import Aside from './components/Aside.vue'
import MainLeaderboard from './components/MainLeaderboard.vue'

export default {
  name: 'App',
  components: {
    Banner,
    Aside,
    MainLeaderboard,
  }
}
</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  background-color: #EBEBEB;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  width: 100%;
}

#content {
  width: 90%;
  margin: 30px auto 30px auto;
}

.el-row {
  margin-bottom: 20px;
  &:last-child {
    margin-bottom: 0;
  }
}
.el-col {
  border-radius: 4px;
}
.bg-purple-dark {
  background: #99a9bf;
}
.bg-purple {
  background: #d3dce6;
}
.bg-purple-light {
  background: #e5e9f2;
}
.grid-content {
  border-radius: 4px;
  min-height: 36px;
}
.row-bg {
  padding: 10px 0;
  background-color: #f9fafc;
}
</style>
