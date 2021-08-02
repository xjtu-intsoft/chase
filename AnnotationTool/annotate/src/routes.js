import Login from './components/Login.vue'
import Annotation from './components/Annotation.vue'
import AnnotationList from './components/AnnotationList.vue'
import DatabaseList from './components/DatabaseList.vue'
import NewAnnotation from './components/NewAnnotation.vue'
import NewAnnotationList from './components/NewAnnotationList.vue'
import TranslateDatabaseList from './components/translation/TranslateDatabaseList.vue'
import TranslateDatabase from './components/translation/TranslateDatabase.vue'
import TranslateConversationList from './components/translation/TranslateConversationList.vue'
import TranslateAnnotation from './components/translation/TranslateAnnotation.vue'

const routes = [
  { path: "/login", component: Login, name: "login" },
  { path: "/translate/conversation/:id", component: TranslateAnnotation, name: "translateAnnotation"}, 
  { path: "/translate/database/:id/conversations/", component: TranslateConversationList, name: "translateConversations" }, 
  { path: "/translate/database/:id", component: TranslateDatabase, name: "translate" }, 
  { path: "/translate/database", component: TranslateDatabaseList, name: "translateDatabases" },
  { path: "/database", component: DatabaseList, name: "databases" },
  { path: "/database/:id", component: NewAnnotation, name: "newConversation" },
  { path: "/conversation/:id", component: NewAnnotation, name: "detailedConversation" },
  { path: "/conversation", component: NewAnnotationList, name: "newConversationList" },
  { path: "/list", component: AnnotationList, name: "list" },
  { path: "/:id(ex_\\d+)", component: Annotation, name: "detail" },
  { path: "/", component: Annotation, name: "root" },
];

export default routes