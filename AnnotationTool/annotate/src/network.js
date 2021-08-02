import axios from "axios";

const baseurl = "/"

const getRemainingConversatioin = () => {
  const url = baseurl + "api/meta";
  return axios.get(url)
}

const getAnnotationList = () => {
    const url = baseurl + "api/annotation_list"
    return axios.get(url)
}

const getAnnotation = (cid) => {
    const url = baseurl + "api/annotate"
    if (cid) {
        return axios.get(url + "/" + cid)
    }
    return axios.get(url)
}

const getAnnotationByDatabaseId = (dbId) => {
    const url = baseurl + "api/annotate/db/" + dbId
    return axios.get(url)
}

const postAnnotation = (payload) => {
    const url = baseurl + "api/annotate"
    return axios.post(url, payload)
}

const login = (payload) => {
    const url = baseurl + "api/login"
    return axios.post(url, payload)
}

const getDatabase = (dbId) => {
    const url = baseurl + "api/db/" + dbId
    return axios.get(url)
}

const getDatabaseList = () => {
    const url = baseurl + "api/db_list/"
    return axios.get(url)
}

const parseSql = (payload) => {
    const url = baseurl + "api/sql/"
    return axios.post(url, payload)
}

const submitNewConversation = (payload) => {
    const url = baseurl + "api/conversation/"
    return axios.post(url, payload)
}

const getNewAnnotationList = () => {
    const url = baseurl + "api/new_annotation_list"
    return axios.get(url)
}

const getConversationbyId = (cid) => {
    const url = baseurl + "api/conversation/" + cid
    return axios.get(url)
}

// Translate Databases
const getTranslateDatabaseList = () => {
    const url = baseurl + "api/translate/database/"
    return axios.get(url)
}

const getTranslateDatabaseById = (dbId) => {
    const url = baseurl + "api/translate/database/" + dbId
    return axios.get(url)
}

const submitTranslation = (payload) => {
    const url = baseurl + "api/translate/database/"
    return axios.post(url, payload)
}

// Translate SparC conversations
const getTranslateConversationList = (dbId) => {
    const url = baseurl + "api/translate/database/" + dbId + "/conversations/"
    return axios.get(url)
}

const getTranslatedDatabaseByid = (dbId) => {
    const url = baseurl + "api/translated/database/" + dbId
    return axios.get(url)
}

const getTranslateConversationbyId = (cid) => {
    const url = baseurl + "api/translate/conversation/" + cid
    return axios.get(url)
}

const deleteTranslateConversation = (cid) => {
    const url = baseurl + "api/translate/conversation/" + cid
    return axios.delete(url)
}

const submitTranslateConversation = (payload) => {
    const url = baseurl + "api/translate/conversation/"
    return axios.post(url, payload)
}

export default {
  getRemainingConversatioin,
  getAnnotationList,
  getAnnotation,
  getAnnotationByDatabaseId,
  postAnnotation,
  login,
  getDatabase,
  getDatabaseList,
  parseSql,
  submitNewConversation,
  getNewAnnotationList,
  getConversationbyId,
  getTranslateDatabaseList,
  getTranslateDatabaseById,
  submitTranslation,
  getTranslateConversationList,
  getTranslatedDatabaseByid,
  getTranslateConversationbyId,
  deleteTranslateConversation,
  submitTranslateConversation
}