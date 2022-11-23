import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")



from PIL import Image
import base64
import pandas as pd
import streamlit as st
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

#For KERAS
import random
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
import time

import numpy
from sklearn.model_selection import GridSearchCV

import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#from keras.layers import Dense
#from keras.layers import Dropout
# Function to create model, required for KerasClassifier


def create_model(optimizer='RMSprop', learn_rate=0.1, momentum=0.4, activation='sigmoid', dropout_rate=0.0):
    
    keras_model = Sequential()
    keras_model.add(Dense(128, input_dim=train_encoded.shape[1], activation=activation))
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(32, activation=activation)) 
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(8,activation=activation)) 
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(1,activation='linear'))
    keras_model.summary()
    # Compile model
    keras_model.compile(loss='mean_squared_error', optimizer=optimizer)

    return keras_model


######################
# Custom function
######################
## Calculate molecular descriptors

def get_ecfc(smiles_list, radius=2, nBits=2048, useCounts=True):
    """
    Calculates the ECFP fingerprint for given SMILES list
    
    :param smiles_list: List of SMILES
    :type smiles_list: list
    :param radius: The ECPF fingerprints radius.
    :type radius: int
    :param nBits: The number of bits of the fingerprint vector.
    :type nBits: int
    :param useCounts: Use count vector or bit vector.
    :type useCounts: bool
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """     
    
    ecfp_fingerprints=[]
    erroneous_smiles=[]
    for smiles in smiles_list:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_smiles.append(smiles)
        else:
            mol=Chem.AddHs(mol)
            if useCounts:
                ecfp_fingerprints.append(list(AllChem.GetHashedMorganFingerprint(mol, radius, nBits)))  
            else:    
                ecfp_fingerprints.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits).ToBitString()))  
    
    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = smiles_list)
    # Remove erroneous data
    if len(erroneous_smiles)>0:
        print("The following erroneous SMILES have been found in the data:\n{}.\nThe erroneous SMILES will be removed from the data.".format('\n'.join(map(str, erroneous_smiles))))           
        df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')    
    
    return df_ecfp_fingerprints




## generate dataset it is diffrent from origin one  
import deepchem as dc
from deepchem.models import GraphConvModel

def generate(SMILES, verbose=False):

    featurizer = dc.feat.ConvMolFeaturizer()
    gcn = featurizer.featurize(SMILES)
    properties = [random.randint(-1,1)/100  for i in range(0,len(SMILES))]
    dataset = dc.data.NumpyDataset(X=gcn, y=np.array(properties))
    
    return dataset


######################
# Page Title
######################



st.write("""# Accelerated reaction energy prediction for redox batteries  🧪 """)
st.write('By: [Alishba Imran](https://www.linkedin.com/in/alishba-imran-/)')




#%%
# About PART

about_part = st.expander("Learn More About Project", expanded=False)
with about_part:
    st.write('''
	     #### About
	     Redox flow batteries (RFB) are widely being explored as a class of electrochemical energy storage devices for large-scale energy storage applications. Redox flow batteries convert electrical energy to chemical energy via electrochemical reactions (through reversible oxidation and reduction) of compounds. 
	     
	     To develop next-gen redox flow batteries with high cycle life and energy density, we need to speed up the discovery of electroactive materials with desired properties. This process can currently be very slow and expensive given how large and diverse the chemical space of the candidate compounds is.
	     
	      Using an attention-based graph convolutional neural network technique, I've developed a model that can take in reactants as SMILEs and predict the reaction energy in the redox reaction. 
	     	     
	      A lot of this work was inspired and built on top of the paper [here](https://chemrxiv.org/engage/chemrxiv/article-details/60c7575f469df44a40f45465). Feel free to give it a try and reach out for any feedback. Email: alishbai734@gmail.com.
	     
	    
	''')




st.write('**Insert your SMILES**')

st.write('Type any SMILES used as a reactant in the redox reaction. This model will output the reaction energy.')

## Read SMILES input
SMILES_input = "Oc1cccc(c12)c(O)c(nn2)O\nc1cccc(c12)cc(nn2)O\nOc1c(O)ccc(c12)cc(nn2)O"

SMILES = st.text_area('press ctrl+enter to run model!', SMILES_input, height=20)
SMILES = SMILES.split('\n')
SMILES = list(filter(None, SMILES))



# st.header('Input SMILES')
# SMILES[1:] # Skips the dummy first item

# Use only top 1000
if len(SMILES)>1000:
    SMILES=SMILES[0:1000]
	
## Calculate molecular descriptors
ecfc_encoder = get_ecfc(SMILES)

#Import pretrained models

#---------------------------------------------------------------------------------
### generate dataset from SMILES and function generate
generated_dataset = generate(SMILES)

### transformer for gcn 
filename = 'final_models/transformers.pkl'
infile = open(filename,'rb')
transformers = pickle.load(infile)
infile.close()


## model for gcn 
model_dir = 'final_models/tf_chp_initial'
gcne_model = dc.models.GraphConvModel(n_tasks=1, batch_size=100, mode='regression', dropout=0.25,model_dir= model_dir,random_seed=0)
gcne_model.restore('final_models/tf_chp_initial/ckpt-94/ckpt-197')
#print(gcne_model)


## predict energy from gcn model 
pred_gcne = gcne_model.predict(generated_dataset, transformers)


#---------------------------------------------------------------------------------
##keras model load
from keras.models import model_from_json

keras_final_model = model_from_json(open('./final_models/keras_final_model_architecture.json').read())
keras_final_model.load_weights('./final_models/keras_final_model_weights.h5')

#keras_final_model = pickle.load(open(r'./final_models/keras_final_model.txt', "rb"))
rf_final_model = pickle.load(open(r'./final_models/rf_final_model.txt', "rb"))
#xgbm_final_model = pickle.load(open(r'.\final_models\xgbm_final_model.txt', "rb"))



#predict test data (Keras,RF, GCN)
pred_keras = keras_final_model.predict(ecfc_encoder)   
pred_rf  = rf_final_model.predict(ecfc_encoder)

##reshape (n,)    ----> (n,1)

pred_rf_r = pred_rf.reshape((len(pred_rf),1))
#pred_xgb = xgbm_final_model.predict(ecfc_encoder)   


#calculate consensus
pred_consensus = (pred_keras + pred_gcne + pred_rf)/3
# predefined_models.get_errors(test_logS_list,pred_enseble)

#%% Weighted 

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------






from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

## Test 1 Experiments

test1_mae = []

test1_mae.append(0.00705) # 0 - GCN
test1_mae.append(0.00416) # 1 - Keras
test1_mae.append(0.0035) # 3 - RF



## Test 2 Experiments

test2_mae = []

test2_mae.append(0.00589) # 0 - GCN
test2_mae.append(0.00483) # 1 - Keras
test2_mae.append(0.00799) # 3 - RF



weighted_pred_0_1_3=( np.power(2/(test1_mae[0]+test2_mae[0]),3) * pred_gcne + 
            np.power(2/(test1_mae[1]+test2_mae[1]),3) * pred_keras + 
            np.power(2/(test1_mae[2]+test2_mae[2]),3) * pred_rf_r ) / (
            np.power(2/(test1_mae[0]+test2_mae[0]),3) + np.power(2/(test1_mae[1]+test2_mae[1]),3) + np.power(2/(test1_mae[2]+test2_mae[2]),3)) 



#--------

#### ????  array shape not correct and no difference with pred_consensus

pred_weighted = (pred_gcne + pred_keras + pred_rf_r)/3







#%%
# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])

df_results = pd.DataFrame(SMILES, columns=['SMILES Reactant'])
df_results["Predicted Reaction Energy"]= weighted_pred_0_1_3
#df_results["reaction_energy"]= pred_weighted
df_results=df_results.round(6)

# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)


# Results DF

st.header('Prediction of Reaction Energy for RFB')
df_results # Skips the dummy first item






# download=st.button('Download Results File')
# if download:
csv = df_results.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings
 


