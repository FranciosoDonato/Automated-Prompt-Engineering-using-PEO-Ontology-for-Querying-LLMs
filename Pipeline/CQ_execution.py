from rdflib import Graph
def query_sparql():
    g = Graph()

    g.parse("PEO_extension.rdf", format="xml")

    query = """ 
    PREFIX peo: <https://w3id.org/peo#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?task ?taskClass
    WHERE {
    ?characteristic rdf:type peo:Language.
    ?class rdfs:subClassOf* peo:View.
    ?view rdf:type ?class.
    ?view peo:hasCharacteristic ?characteristic.
    ?task peo:hasView ?view.
    ?task rdf:type ?taskClass.
    }  
    """
    return g.query(query)

result = query_sparql()
for row in result:
    if "NamedIndividual" not in str(row.taskClass):
        print("-" * 40)
        print(f"Task: {row.task}")
        print(f"Task Class: {row.taskClass}") 
print("-" * 40)