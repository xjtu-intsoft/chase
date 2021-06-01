<template>
    <div id="banner">
        <h1><b>{{ title }}</b> {{ version }}</h1>
        <h2>{{ subTitle }}</h2>
        <img id="img-jiaoda" class="logo" alt="Xi'an Jiaotong Univeristy" src="../assets/jiaoda.png" />
        <img id="img-beihang" class="logo" alt="Xi'an Jiaotong Univeristy" src="../assets/beihang_logo.gif" />
        <img id="img-microsoft" class="logo" alt="Xi'an Jiaotong Univeristy" src="../assets/msra.png" />
    </div>
</template>

<script>
export default {
  name: 'Banner',
  data() {
      return {
          title: "Chase",
          version: "1.0",
          subTitle: "A Large-Scale and Pragmatic Chinese Dataset for Cross-Database Context-Dependent Text-to-SQL"
      }
  },
}
</script>

<style>
#banner {
    width: 100%;
    background-color: #05A1E8;
    color: #ffffff;
    font-family: 'Lato, Arial, sans-serif';
    padding-top: 30px;
    padding-bottom: 40px;
    letter-spacing: 0.005em;
    line-height: 1.5;
}
#banner > h1 {
    width: 80%;
    font-size: 6em;
    font-weight: 90;
    text-align: center;
    margin: 0 auto 0 auto;
    padding: 0px;
    font-variant-caps: small-caps;
}

#banner > h2 {
    width: 80%;
    font-size: 2em;
    font-weight: 100;
    text-align: center;
    margin: 0 auto 0 auto;
    padding: 0px;
}

.logo {
    float: left;
    top: 5px;
    position: absolute;
}

#img-jiaoda {
    left: 10px;
    top: 30px;
    width: 150px;
}

#img-beihang {
    left: 15px;
    top: 170px;
    height: 60px;
}

#img-microsoft {
    float: left;
    top: 100px;
    left: 15px;
    height: 40px;
}
</style>