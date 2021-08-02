const copy = (obj) => {
    return JSON.parse(JSON.stringify(obj));
}

const newEmptyArray = (length) => {
    var array = new Array
    for(var i = 0; i < length; i++){
        array.push(new Array)
    }
    return array
}

const sample = (array) => {
    return array[Math.floor(Math.random() * array.length)]
}

const searchEdges = (group1, group2, edges) => {
    var candidateEdges = new Array
    for(let i = 0; i < group1.nodes.length; i++){
        for(let j = 0; j < group2.nodes.length; j++){
            for(let z = 0; z < edges.length; z++){
                let currEdge = edges[z]
                if((currEdge.node1 === group1.nodes[i].tableName && currEdge.node2 === group2.nodes[j].tableName) || (currEdge.node2 === group1.nodes[i].tableName && currEdge.node1 === group2.nodes[j].tableName)){
                    candidateEdges.push(edges[z])
                }
            }
        }
    }
    return candidateEdges
}

const searchJoinPath = (nodes, edges, selectedTables) => {
    var selectedNodes = new Array
    for(let i = 0; i < selectedTables.length; i++){
        for(let j = 0; j < nodes.length; j++){
            if(selectedTables[i] === nodes[j].tableName){
                selectedNodes.push(nodes[j])
                break
            }
        }
    }

    // Initially, each node is a group
    var groups = new Array
    for(let i = 0; i < selectedNodes.length; i++){
        groups.push({
            nodes: [selectedNodes[i]],
            edges: []
        })
    }
    var oldGroupLength = groups.length
    var newGroups = new Array
    // Merge Groups
    while(newGroups.length != 1 && newGroups.length != oldGroupLength){
        oldGroupLength = newGroups.length
        newGroups = new Array
        while(groups.length > 0){
            // Pop one from the left most position
            let currGroup = groups.splice(0, 1)[0]
            // Merge with other groups
            let groupsToRemove = new Array
            for(let i = 0; i < groups.length; i++){
                let candidateEdges = searchEdges(currGroup, groups[i], edges)
                if(candidateEdges.length > 0){
                    // Merge
                    groupsToRemove.push(groups[i])
                    currGroup.nodes = currGroup.nodes.concat(groups[i].nodes)
                    currGroup.edges.push(candidateEdges[0]) // Only get the first edge, Prone to introduce bugs
                    currGroup.edges = currGroup.edges.concat(groups[i].edges)
                }
            }
            newGroups.push(currGroup)
            //: Remove
            for(let i = 0; i < groupsToRemove.length; i++){
                let index = groups.indexOf(groupsToRemove[i])
                groups.splice(index, 1)
            }
        }
        groups = newGroups
    }

    console.log(newGroups)
    // To From Clause
    if(newGroups.length != 1){
        return ""
    }
    var fromClause = ""
    var finalGroup = newGroups[0]
    if(finalGroup.edges.length === 0){
        fromClause = `from ${finalGroup.nodes[0].tableName}`
    }else{
        var tableAliasMap = {}
        var currTableAliasIndex = 0
        for(let i = 0; i < finalGroup.edges.length; i++){
            let e = finalGroup.edges[i]
            let node1 = e['node1']
            let node2 = e['node2']
            let col1 = e['col1']
            let col2 = e['col2']
            if(i == 0){
                currTableAliasIndex += 1
                tableAliasMap[node1] = currTableAliasIndex
                currTableAliasIndex += 1
                tableAliasMap[node2] = currTableAliasIndex
                fromClause = `from ${node1} AS T${tableAliasMap[node1]} JOIN ${node2} AS T${tableAliasMap[node2]} ON T${tableAliasMap[node1]}.${col1} = T${tableAliasMap[node2]}.${col2}`
            }else{
                if(node1 in tableAliasMap && !(node2 in tableAliasMap)){
                    currTableAliasIndex += 1
                    tableAliasMap[node2] = currTableAliasIndex
                    fromClause += ` JOIN ${node2} AS T${tableAliasMap[node2]} ON T${tableAliasMap[node1]}.${col1} = T${tableAliasMap[node2]}.${col2}`
                }else if(node2 in tableAliasMap && !(node1 in tableAliasMap)){
                    currTableAliasIndex += 1
                    tableAliasMap[node1] = currTableAliasIndex
                    fromClause += ` JOIN ${node1} AS T${tableAliasMap[node1]} ON T${tableAliasMap[node2]}.${col2} = T${tableAliasMap[node1]}.${col1}`
                }
            }
        }
    }
    return fromClause
}

export default {
    copy,
    newEmptyArray,
    searchJoinPath,
    sample
}