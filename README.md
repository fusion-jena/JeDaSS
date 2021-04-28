# JeDaSS
This repository is used to summarize and report the methodology and results of "Towards Scientific Data Synthesis Using DeepLearning and Semantic Web". This study has been  submitted and accepted at 18th International Conference on Extended Semantic Web Conference (ESWC2021) (https://2021.eswc-conferences.org/): Poster and Demo track.

Abstract. One  of  the  added  values  of  long  running  and  large  scalecollaborative projects is the ability to answer complex research questionsbased on the extensive use of collected data. In practice, however, findingand identifying related data in the central repository of these projectsoften proves to be a demanding task. In this paper, we aim to release datafrom silos, thereby enabling cross-cutting analyses that were earlier out ofreach. To achieve that we introduce a new data summarization//alsayed:(analysis)//and profiling approach exploiting the semantics of annotateddatasets  using  a  domain  specific  ontology.  In  particular,  the  proposedapproach makes use of the capability of machine learning to categorizea give dataset into a domain topic and to extract hidden links across itsdata  attributes  and  other  data  attributes  from  different  datasets.  Theproposed approach has been developed and has been applied to datasetscollected in the CRC AquaDiva.

# Paper availability
The accepted version can be accesed vis the [publisher web site](https://link.springer.com/chapter/10.1007/978-3-030-62327-2_1)

# Repository folders
This repository includes almost the material related to the development of Jena Dataset Summarization 
and Synthesis (JeDaSS) tool. It includes the following:

* folder API :
    - API.py class to run the server endpoint (flask):
        - set FLASK_APP=API\API.py
        - flask run
    - ClassifierSemantic:
        - main class containing the hypothesis and the workflow.
    - config.yaml :
        - configuration files for relative paths
    - data_preparation.py : 
        - processing of image generation.
    - model.py:
        - resnet module loader.
    - helper_lib.py:
        helper scripts to read the config file and some other methods.
    - Postman collection folder : 
        - Postman API test for the endpoints provided by the flask server and the AquaDiva dataset Extractor.
    
* folder Prediction : 
    - contains the tuples and keywords used to explore the AquaDIva ontology as a graph connecting the classes with a set of relationship : entity - relation - entity.

* folder uploads:
    - contains the requested datasets for the API where they are saved.
    
# Usage
view [API readme](API/README.md) file to have much more information on the usuage.

Acknowledgments: This work has been mostly funded by the Deutsche Forschungsgemeinschaft (DFG) as part of the CRC 1076 AquaDiva [http://www.aquadiva.uni-jena.de/]
