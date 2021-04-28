import os
import pandas as pd
import copy
import csv
import json
import helper_lib as helper
import sys
import torch
from model import model_res
from torch.utils.data import DataLoader, TensorDataset
import glob
import warnings
from data_preparation import data_to_image
from helper_lib import NpEncoder
import networkx as nx
sys.setrecursionlimit(1000000)
#sys.path.append('/usr/lib/python3/dist-packages')

class preprocessor_andAnalyser_lib:

    def __init__(self):
        self.home = os.path.dirname(__file__)
        self.config = helper.read_yaml(self.home+"/config.yaml")
        
        self.all_data = pd.read_csv(self.home+self.config['API']['PREDICTION_SUBJECT_RELATION_OBJECT'], index_col=False, engine='python', encoding='utf-8')
        self.lab = pd.read_csv(self.home+self.config['API']['PREDICTION_DS_PROJECT'], index_col=False, engine='python', encoding='utf-8')
        self.all_data_Annot = pd.read_csv(self.home+self.config['API']['PREDICTION_ANNOTATION_KNOWLEDGE_GRAPH'], index_col=False, engine='python', encoding='utf-8', delimiter=";")
        self.all_Obs = pd.read_csv(self.home+self.config['API']['PREDICTION_OBSERVATION_KNOWLEDGE'], index_col=False, engine='python', encoding='utf-8')

        self.ARIAL_FONT_PATH = self.home+self.config['API']['ARIAL_FONT_PATH']

        self.all_Obs.drop(columns=['version_id','contextualized_entity_id','contextualizing_entity_id','id'], inplace=True)
        self.all_Obs.columns=["label","entity1","entity2"]

        print ("initialise predict files ")
        self.all_Obs.append(self.all_data_Annot, ignore_index=True)
        print (self.all_Obs)

        self.all_data.append(self.all_Obs)
        print (self.all_data_Annot)

        print ("generating the graph...")
        self.g = nx.DiGraph()
        for i in range (0, len(self.all_data)):
            tup = ( self.all_data['entity1'][i] , self.all_data['entity2'][i] )
            self.g.add_edges_from( [tup] )
        print ("edges in graph : " , self.g.number_of_edges() , " with nodes : " , len(self.g))

        self.root_folder_for_uploads = self.home+self.config['API']['ROOT_FOLDER_FOR_UPLOADS']
        self.ser_admin = self.config['API']['USER_ADMIN']
        self.group_admin = self.config['API']['GROUP_ADMIN']
        self.keywordmap = glob.glob(self.home+self.config['API']['KEYWORD_MAP_GLOB'])

        torch.backends.cudnn.enabled = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        warnings.filterwarnings("ignore")

        self.model_res = torch.load(self.home+self.config['API']['TORCH_CLASSIFICATION_MODEL'], map_location=self.device)
        self.model = self.model_res.to(self.device)

    def prepare_file_to_frame(self,file_path):


    	
        test_ = pd.read_csv(file_path, delimiter = ';', index_col=False, engine='python', encoding='utf-8')
        test_.fillna(0, inplace=True)
        test__ = test_

        ## preprocessing the data frame's types 
        test_['datasetID']=test_['datasetID'].astype(str)
        print ("project dashboard loaded")
        print (test_)
        original_labels = test_['project']
        test_['project'] = pd.factorize(test_['project'])[0]
        encoded_labels = test_['project']

        test = test_.copy(deep=True)
        test = test_.drop(columns=['owner', 'datasetID','Datasetversion_id','standard_id','variable_id','entity_id','charachteristic_id','standard'])
        print ("**********************")
        print (test_.columns)
        print ("**********************")
        print (test.columns)
        print ("**********************")

        test.drop_duplicates(inplace=True)
        target_data_ = test['project']
        print (target_data_)
        test.drop(columns=['project'], inplace=True)
        print (test['variable_id_from_table'].unique())
        return test, target_data_

    def convert_images(self,test):
        test_images = data_to_image(test.to_numpy(),self.ARIAL_FONT_PATH)
        return test_images

    def classify(self,test,test_images,target_data_):
        ## loading a torch tensor from the test data and target data and processing the model for classification
        X_test = torch.from_numpy(test_images).float()
        y_test = torch.from_numpy(target_data_.values).long()
        test_dataset = TensorDataset(X_test, y_test)

        batch_size=1
        dataloaders = {'test': DataLoader(test_dataset,batch_size, shuffle=False)}
        dataset_sizes = {'test': len(test_dataset)}
        i = 10
        out_data = {}

        print ("processing classification .... please wait ")
        with torch.no_grad():
            test_index_input = -1 
            for inputs, labels in dataloaders['test']:
                if torch.cuda.is_available():
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                # obtain the outputs from the model
                outputs = self.model.forward(inputs)
                
                label_images_batch = labels.data.cpu().numpy()
                input_images_batch = inputs.data.cpu().numpy()
                rows = int(dataset_sizes['test']/batch_size)
                
                '''  uncomment to view images
                for j in range(rows):
                    num = 0
                    fig = plt.figure(figsize=(100,30))
                    for k in input_images_batch:
                        num = num +1 
                        ax1 = fig.add_subplot(rows,batch_size,num)
                        ax1.imshow(k[0, :, :])
                    plt.show()
                '''     
                
                #print ("    **************** " )
                test_index_input  = test_index_input +1
                data = {}
                head = {}

                for j in range (test.iloc[test_index_input].count()):
                    #print (list(test_.columns)[j] , test_.iloc[test_index_input][j])
                    head[list(test.columns)[j]] = test.iloc[test_index_input][j]
                    head['entity'] =str(test.iloc[test_index_input]['entity']).strip() # str(list(test_[test_['entity']==test.iloc[test_index_input]['entity']]['entity'])[0]).strip()
                    head['charachteristic'] =str(test.iloc[test_index_input]['charachteristic']).strip() # str(list(test_[test_['charachteristic']==test.iloc[test_index_input]['charachteristic']]['charachteristic'])[0]).strip()
                    #head['charachteristic'] = str(list(test_[test_['charachteristic_id']==test.iloc[test_index_input]['charachteristic_id']]['charachteristic'])[0]).strip()

                #key = str(head['variable_id_from_table'])
                key = str(test.iloc[test_index_input]['variable_id_from_table'])
                #print (key)
                data['input'] = head
                #print ("Annotated output  : " , target_data_.iloc[test_index_input])
                #print("original text : ", original_labels[list(encoded_labels).index(target_data_.iloc[test_index_input])])
                data_ = {}
                index = -1
                for k in outputs[0]:
                    index = index + 1 
                    #print("Prediction for the class : " ,index , " = " ,k.item() ) #the model output
                    data_[index] = float(k.item())
                data['class_score'] = data_
                _, predicted = outputs.max(dim=1)

                if key in out_data:
                    if (str(out_data[key]['predicted_class']).find(str(predicted.item()).strip()) < 0):
                        out_data[key]['predicted_class']= str(out_data[key]['predicted_class']) + " ; " +str(predicted.item()).strip()
                        out_data[key]['input']['variable_value']= str(out_data[key]['input']['variable_value']) + " ; " + str(test.iloc[test_index_input]['variable_value']).strip()
                if key not in out_data:
                    #print (key , " classified ... to " , str(predicted.item()).strip())
                    data['predicted_class']= str(predicted.item()).strip()
                    out_data [key] = data
                #data['predicted_class'] = str(predicted.item())
                #print("probability prediction : ",torch.exp(_).item()) # the predicted probability
                equals = predicted == labels.data
                #print("Comapred the correct classes from the Missclassified classes : \n" ,equals.item())
                #print("Mean of equal classes : ", equals.float().mean().item())
                #print ("    ****************")
        
        ## end of classification
        return out_data

        #print (json_dictionary)
        #return json.dumps(json_dictionary, cls=NpEncoder)

    def semantic_linking(self,out_data):
        ## processing the semantic extraction
        print ("****** mapping to ontology *************")
        
        json_dictionary = json.loads(json.dumps(out_data, cls=NpEncoder))
        for key in json_dictionary:
            #json_dictionary[key]['target'] = set([])
            json_dictionary[key]['onto_match'] = set([])
            json_dictionary[key]['onto_no_path'] = set([])
            json_dictionary[key]['onto_no_node'] = set([])
            json_dictionary[key]['db_match'] = set([])
            json_dictionary[key]['db_no_path'] = set([])
            json_dictionary[key]['db_no_node'] = set([])
            json_dictionary[key]['onto_target_file'] = set([]) ;

            classification_result = sorted(json_dictionary[key]['class_score'].items(), key=lambda x: x[1] , reverse=True)
            print (classification_result)
            predicts = str(json_dictionary[key]['predicted_class']).split(";")
            top = 0
            predictions = "" ; 
            #for predict_ in predicts:
            for predict_ in classification_result:
                top = top +1;
                if (top == 5) :
                    json_dictionary[key]['predicted_class'] = predictions
                    break;
                #predict = int(predict_)
                predict = int(predict_[0])
                predictions = predictions +  " ; " + str(predict) ;
                class_to= "all"
                if  ( (predict ==5 )):
                    class_to = "A01"
                elif ( (predict ==4 )):
                    class_to = "A02" 
                elif  ( (predict ==6 )):
                    class_to = "A03" 
                elif ( (predict ==8 )):
                    class_to = "B01" 
                elif  ( (predict ==3 )):
                    class_to = "B02" 
                elif  ( (predict ==7 )):
                    class_to = "B03" 
                elif ( (predict ==9 )):
                    class_to = "C03"
                elif ( (predict == 0)):
                    class_to = "D03"
                elif ( (predict == 1 )):
                    class_to = "B04" 
                elif ( (predict == 2 )):
                    class_to = "A04" 
                elif ( (predict == 10 )):
                    class_to = "A06" 
                elif ( (predict == 11 )):
                    class_to = "C05" 

                #print(key, " : predict : ",  str(class_to) , " : " , str(predict) , " : ",json_dictionary[key]['input']['variable_id_from_table']," -> ",json_dictionary[key]['input']['entity'] )
            
                ## class_to represents the Classification result/Group that are referenced to a file containing the relationship tuple extracted from the ontology using their keywords and research questions
                keyword_file_list = []
                for keyword in self.keywordmap:
                    if (keyword.find(class_to)> -1 ):
                        keyword_file_list.append(keyword)
                        json_dictionary[key]['onto_target_file'].add(keyword) 

                for filename in keyword_file_list:
                    #print ("file to search : " , filename)
                    try:
                        sub_data = pd.read_csv(filename, index_col=False, engine='python', encoding='utf-8' , sep=" , ")
                        sub_data.drop(columns='"label"', inplace=True)
                    except:
                        continue

                    for colname in sub_data.columns:
                        targets= sub_data[colname].astype(str).tolist()
                        for x in targets:
                            #print("target  -> ", x.replace('"','').strip() )
                            #json_dictionary[key]['target'].add("onto : " + str(x))
                            try:
                                n=nx.shortest_path(self.g, source= str(json_dictionary[key]['input']['entity']).strip(), target=str(x).replace('"','').strip())
                                n_=nx.shortest_path(self.g, source= str(json_dictionary[key]['input']['charachteristic']).strip(), target=str(x).replace('"','').strip())
                                #print ("match")
                                json_dictionary[key]['onto_match'].add(str(n)) 
                                json_dictionary[key]['onto_match'].add(str(n_)) 
                                #print (n)
                            except nx.NetworkXNoPath:
                                json_dictionary[key]['onto_no_path'].add(str(x))
                                continue;
                                #print ('No path')
                            except nx.NodeNotFound:
                                #json_dictionary[key]['onto_no_node'].add(str(x))
                                continue;
                
                print (" going to database matches " )
                
                try:
                    sub_data = self.all_Obs ## this is the sematic tuples that are extracted from the AquaDiva annotations and observations
                    sub_data.drop(columns="label", inplace=True)
                except:
                    continue
                for colname in sub_data.columns:
                    targets= sub_data[colname].astype(str).tolist()
                    for x in targets:
                        #print("target  -> ", x.strip() )
                        #json_dictionary[key]['target'].add("db : " + str(x))
                        try:
                            n=nx.shortest_path(self.g, source= str(json_dictionary[key]['input']['entity']).strip(), target=str(x).strip())
                            n_=nx.shortest_path(self.g, source= str(json_dictionary[key]['input']['charachteristic']).strip(), target=str(x).strip())
                            json_dictionary[key]['db_match'].add(str(n) ) 
                            json_dictionary[key]['db_match'].add(str(n_) )  
                            #print ("match")
                            #print (n)
                        except nx.NetworkXNoPath:
                            json_dictionary[key]['db_no_path'].add(str(x))
                            continue;
                            #print ('No path')
                        except nx.NodeNotFound:
                            #json_dictionary[key]['db_no_node'].append(str(x))
                            continue;
                        
        print ("finished")
        #return json.dumps(out_data, cls=NpEncoder)
        #print (json_dictionary)
        return json.dumps(json_dictionary,cls=NpEncoder)
        
