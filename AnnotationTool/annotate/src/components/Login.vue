<template>
    <div class="login">
        <form>
            <div class="input-block">
                <label for="username">Username: </label>
                <input type="text" name="username" v-model="username" id="username">
            </div>
            <div class="input-block">
                <label for="password">Password: </label>
                <input type="password" name="password" v-model="password" id="password">
            </div>
            <div class="input-block error-message" v-if="showErrorMessage">
                Incorrect username or password.
            </div>
            <div class="input-block submit-block">
                <input type="button" name="submit" value="Login" id="loginInput" @click="submit">
            </div>
        </form>
    </div>
</template>

<script>
import Network from '../network.js'

export default {
    name: 'Login',
    data() {
        return {
            username: "",
            password: "",
            showErrorMessage: false
        }
    },
    methods: {
        submit: function(){
            var payload = {
                username: this.username,
                password: this.password
            }
            Network.login(payload).then(response => {
                var data = response.data
                var status = response.status
                if(status == 200 && data.status === "success"){
                    this.$router.push({name: "root"})
                }else{
                    this.showErrorMessage = true
                }
            })
        }
    },
}
</script>

<style scoped>
.login {
    width: 500px;
    margin: 50px auto 0 auto;
}
.input-block {
    width: 70%;
    margin: 0 auto 20px auto;
}
.input-block > label {
    display: block;
    margin-bottom: 5px;
    font-size: 20px;
}
.input-block input {
    width: 100%;
    height: 30px;
    padding: 5px 5px 5px 5px;
    font-size: 20px;
}
.error-message {
    color: red;
    margin-bottom: 10px;
}
.submit-block {
    width: 200px;
    margin: 30px auto 0 auto;
}
#loginInput {
    width: 200px;
    height: 40px;
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 18px;
    background-color: #3AA701;
    cursor: pointer;
}
</style>