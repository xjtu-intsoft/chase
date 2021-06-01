import Vue from 'vue'
import App from './App.vue'
import { Row, Column } from 'vue-grid-responsive'
import { Table, TableColumn, Form, FormItem, Button } from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

Vue.config.productionTip = false
Vue.component('row', Row)
Vue.component('column', Column)
Vue.use(Table)
Vue.use(TableColumn)
Vue.use(Form)
Vue.use(FormItem)
Vue.use(Button)

new Vue({
  render: h => h(App),
}).$mount('#app')
