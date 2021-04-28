# Categotical Analysis API

## Aim: 
The API has been designed to process semantic analysis using deep learning algorithm for classification.
The API uses the code from the class ClassifierSemantic.py to process transformations and image generation
along with the semantic graph analysis.

### Tools used: 
1. Flask
2. pipenv 
3. Postman

### Inputs:
under the folder "Postman Collection" you may find a postman collection to test 
the flask end node with some preset parameters.

PS: Input file is not a random csv tabular data
#### input type:

The input type is based on the BExIS management system. The tabular data uploaded in
the CRC AquaDiva data portal is processed into relational tables.

Within the portal, we provide an internal API to do the process of recreating an input 
data for our system.
the endpoint is not public and users must have access to view the data : 
https://aquadiva-dev1.inf-bb.uni-jena.de/asm/DataSetSummary/classificationAsync?ds={{dataset_id}}&flag=&operation={{operation_desired}},
where : 
{{dataset_id}} is the reltive id of the dataset desired to prepare the file for the pipeline.
{{operation_desired}} is the operation asked to e performed from the module, in our case is equal to "prepare_only"

https://aquadiva-dev1.inf-bb.uni-jena.de/asm/DataSetSummary/classificationAsync?ds=&flag=&operation=prepare_only

|datasetID|Datasetversion_id|variable_id|unit           |type   |entity_id|entity                                                                                             |charachteristic_id|charachteristic                                                                                 |standard_id|standard                                                      |dataset_title                                                              |owner |project|variable_id_from_table|variable_value                                                                                                                                                                                                                                                                                                                                          |FIELD17|
|---------|-----------------|-----------|---------------|-------|---------|---------------------------------------------------------------------------------------------------|------------------|------------------------------------------------------------------------------------------------|-----------|--------------------------------------------------------------|---------------------------------------------------------------------------|------------------------|-------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
|***      |***              |2829       |none           |date   |367      |http://www.aquadiva.uni-jena.de/ad-ontology/ad-ontology.0.0/ad-ontology-entities.owl#TemporalEntity|9                 |http://www.aquadiva.uni-jena.de/ad-ontology/ad-ontology.0.0/ad-ontology-characteristics.owl#Date|8666       |http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Standard|Water and nitrogen fluxes in stemflow, biweekly, B02 forest plot, year 2016|***   | none  | DateTime             | 12/21/2016 10:00:00 AM-12/7/2016 10:00:00 AM-11/23/2016 10:00:00 AM-11/9/2016 10:00:00 AM-10/26/2016 10:00:00 AM-4/26/2016 9:59:59 AM-4/12/2016 9:59:59 AM-3/29/2016 9:59:59 AM-3/15/2016 9:59:59 AM-3/1/2016 9:59:59 AM-2/16/2016 9:59:59 AM-1/19/2016 9:59:59 AM-1/5/2016 9:59:59 AM-12/21/2015 9:59:59 AM-12/7/2015 9:59:59 AM-2/2/2016 9:59:59 AM- |       |
|***      |***              |2833       |Meter          |Decimal|909      |http://purl.obolibrary.org/obo/OBI_0000968                                                         |1102              |http://purl.obolibrary.org/obo/GAZ_00000448                                                     |8666       |http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Standard|Water and nitrogen fluxes in stemflow, biweekly, B02 forest plot, year 2016|***   | none  | X                    | 598783.6-598902.255-598829.515-598787.707-                                                                                                                                                                                                                                                                                                             |       |
|***      |***              |2834       |Meter          |Decimal|909      |http://purl.obolibrary.org/obo/OBI_0000968                                                         |1102              |http://purl.obolibrary.org/obo/GAZ_00000448                                                     |8666       |http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Standard|Water and nitrogen fluxes in stemflow, biweekly, B02 forest plot, year 2016|***   | Y                    | 5662783.461-5662781.793-5662819.595-5662813.565-                                                                                                                                                                                                                                                                                                       |       |
|***      |***              |2839       |milligram/Liter|Decimal|5899     |http://purl.obolibrary.org/obo/CHEBI_16301                                                         |206               |http://purl.obolibrary.org/obo/PATO_0000033                                                     |8666       |http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Standard|Water and nitrogen fluxes in stemflow, biweekly, B02 forest plot, year 2016|***   | Conc_SF_W_NO2-N      |                                                                                                                                                                                                                                                                                                                                                        |       |

knowing that the columns=['owner', 'datasetID','Datasetversion_id','standard_id','variable_id','entity_id','charachteristic_id','standard'] will be dropped
Under the postman collection folder, you may find also a dummy data to test the pipeline.

### Output:
a json object under  the structure :
```json
{
    "PN-Datum": { 
        "input": {
            "unit": "none",
            "entity": "http://purl.obolibrary.org/obo/OBI_0000747",
            "charachteristic": "http://www.aquadiva.uni-jena.de/ad-ontology/ad-ontology.0.0/ad-ontology-characteristics.owl#Date",
            "type": "Date",
            "dataset_title": "D03 - INFRA3 Hainich CZE_groundwater hydrochemistry",
            "variable_id_from_table": " PN-Datum ",
            "variable_value": " 2/8/2011 12:00:00 AM-3/22/2011 12:00:00 AM-11/23/2010 12:00:00 AM-2/22/2011 12:00:00 AM-11/24/2010 12:00:00 AM-1/12/2011 12:00:00 AM-3/23/2011 12:00:00 AM-4/20/2011 12:00:00 AM-10/27/2010 12:00:00 AM-5/4/2011 12:00:00 AM-3/14/2013 12:00:00 AM-10/26/2010 12:00:00 AM-12/7/2010 12:00:00 AM-5/31/2011 12:00:00 AM-5/19/2011 12:00:00 AM-6/22/2011 12:00:00 AM-3/6/2012 12:00:00 AM-3/7/2012 12:00:00 AM-7/18/2011 12:00:00 AM-2/7/2011 12:00:00 AM-6/8/2011 12:00:00 AM-2/15/2011 12:00:00 AM-4/5/2011 12:00:00 AM-3/8/2011 12:00:00 AM-3/10/2011 12:00:00 AM-4/6/2011 12:00:00 AM-7/20/2011 12:00:00 AM-10/19/2011 12:00:00 AM-7/19/2011 12:00:00 AM-3/1/2013 12:00:00 AM- "
        },
        "class_score": {
            "0": -18.294200897216797,
            "1": -50.246768951416016,
            "2": -25.4317684173584,
            "3": -42.723690032958984,
            "4": -31.546470642089844,
            "5": -39.1209602355957,
            "6": -69.26103973388672,
            "7": -78.90359497070312,
            "8": -85.01768493652344,
            "9": -36.921241760253906,
            "10": -89.94821166992188,
            "11": -66.41468048095703
        },
        "predicted_class": " ; 0 ; 2 ; 4 ; 9",
        "onto_match": [ 
            "['http://purl.obolibrary.org/obo/OBI_0000747', 'http://purl.obolibrary.org/obo/OBI_0100051', 'http://purl.obolibrary.org/obo/BFO_0000040', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Entity', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#MeasurementType', '_:b31', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#measuresCharacteristic']",
            "['http://purl.obolibrary.org/obo/OBI_0000747', 'http://purl.obolibrary.org/obo/OBI_0100051', 'http://purl.obolibrary.org/obo/BFO_0000040', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Entity', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#MeasurementType', '_:b33', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#measuresUsingProtocol']"
        ],
        "onto_no_path": [ 
            "http://purl.obolibrary.org/obo/ENVO_01000391",
            "http://purl.obolibrary.org/obo/ENVO_00002238"
        ],
        "onto_no_node": [],
        "db_match": [ 
            "['http://purl.obolibrary.org/obo/OBI_0000747', 'http://purl.obolibrary.org/obo/OBI_0100051', 'http://purl.obolibrary.org/obo/BFO_0000040', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Entity', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Measurement', '_:b5', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#MeasuredValue']",
            "['http://purl.obolibrary.org/obo/OBI_0000747', 'http://purl.obolibrary.org/obo/OBI_0100051', 'http://purl.obolibrary.org/obo/BFO_0000040', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Entity', 'http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#Standard']"
        ],
        "db_no_path": [ 
            "http://purl.obolibrary.org/obo/PATO_0000146",
            "http://purl.obolibrary.org/obo/ENVO_09100001"
        ],
        "db_no_node": [],
        "onto_target_file": [
            "prediction\\entities_domain\\A04.csv",
            "prediction\\entities_domain\\D03.csv",
            "prediction\\entities_domain\\A02.csv",
            "prediction\\entities_domain\\C03.csv"
        ]
    },
    "data attribute name - 2": {
        
    }
}
```

### How it works: 
The API consists of 3 main files: API.py, data_preparation.py and ClassifierSemantic.py.
1. API.py: 
   Main API file, contains the API endpoint.
2. ClassifierSemantic.py: 
   contains the methods seperated step by step to classify and/or run semantic linking.
3. data_preparation.py:
   contains the methods to generate images per tuples in the data frame.

### How to use: 
1. install packages using pipenv. NOTE: This step is important as of writing as the latest version of numpy does not work. 
2. set flask_app to /API/API.py (used default: http://127.0.0.1:5000/)
3. send requests using postman to check enpoints 
   (used default: http://127.0.0.1:5000/) (Postman collection provided)

