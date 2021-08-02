<template>
  <div id="app">
    <MetaBar :remainingConversation="remainingConversation" :newConversation="newConversation" :remainingDatabase="remainingDatabase" 
             :remainingSparctoRefine="remainingSparctoRefine" />
    <router-view />
  </div>
</template>

<script>
import Network from './network.js'
import MetaBar from './components/Meta.vue'
import { EventBus } from './event_bus.js'

export default {
  name: 'App',
  components: {
    MetaBar
  },
  data() {
    return {
        remainingConversation: 0,
        remainingDatabase: 0,
        newConversation: 0,
        remainingSparctoRefine: 0,
    }
  },
  mounted() {
      this.getRemainingConversation()
      EventBus.$on("updateMeta", () => {
        this.getRemainingConversation()
      });
  },
  methods: {
      getRemainingConversation: function(){
          Network.getRemainingConversatioin().then(response => {
              var data = response.data
              var status = response.status
              console.log(response)
              if(status == 200){
                  this.remainingConversation = data.remainingConversation
                  this.newConversation = data.newConversation
                  this.remainingDatabase = data.remainingDatabase
                  this.remainingSparctoRefine = data.remainingSparctoRefine
              }
          })
      }
  },
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  margin-top: 20px;
}
.scrollable-block::-webkit-scrollbar {
  width: 12px;               /* width of the entire scrollbar */
}
.scrollable-block::-webkit-scrollbar-thumb {
  background-color: #EEEEEE;    /* color of the scroll thumb */
  border-radius: 5px;       /* roundness of the scroll thumb */
}
</style>
