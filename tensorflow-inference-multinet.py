
# Import some things
import os,sys,cv2


import numpy as np
from matplotlib import pyplot as plt


# Bring in Xilinx ML Suite Compiler, Quantizer, PyXDNN
from xfdnn.tools.compile.bin.xfdnn_compiler_tensorflow import TFFrontend as xfdnnCompiler
from xfdnn.tools.quantize.quantize_tf import tf_Quantizer as xfdnnQuantizer
import xdnn ,xdnn_io
import time
import ipywidgets

import warnings
import time
import gc
import pandas as pd
warnings.simplefilter("ignore", UserWarning)

print("Current working directory: %s" % os.getcwd())
print("Running on host: %s" % os.uname()[1])
print("Running w/ LD_LIBRARY_PATH: %s" %  os.environ["LD_LIBRARY_PATH"])
print("Running w/ XILINX_OPENCL: %s" %  os.environ["XILINX_OPENCL"])
print("Running w/ XCLBIN_PATH: %s" %  os.environ["XCLBIN_PATH"])
print("Running w/ PYTHONPATH: %s" %  os.environ["PYTHONPATH"])
print("Running w/ SDACCEL_INI_PATH: %s" %  os.environ["SDACCEL_INI_PATH"])


def initializeFpgaModel(sProtoBufPath, qMode):
    config = {} # Config dict
    config["platform"] = 'aws'
    
    sInputNode,sOutputNode = getModelInputOutputNode(sProtoBufPath)
    # Compiler Arguments
    config["name"] = "googlenet"
    config["net"] = "googlenet_v1"
    config["datadir"]= "/home/centos/ml-suite/examples/classification/data/googlenet_v1_data" #already comiled
    config["quantizecfg"] = "/home/centos/ml-suite/examples/classification/data/googlenet_v1_8b.json" # already quantized
    config["in_shape"] = (3,224,224)
    config["memory"] = 5 # Available on-chip SRAM
    config["dsp"] = 28 # Width of Systolic Array
    config["netcfg"] = '/home/centos/ml-suite/examples/classification/data/{}_{}.json'.format(config['net'],config['dsp'])
    # Quantizing
    config["img_mean"] = [104.007, 116.669, 122.679] # Mean of the training set
    config["bitwidths"] = [qMode,qMode,qMode] # Supported quantization precision
    config["img_raw_scale"] = 255.0 # Raw scale of input pixels, i.e. 0 <-> 255
    config["img_input_scale"] = 1.0 # Input multiplier, Images are scaled by this factor after mean subtraction

    # Create a handle with which to communicate to the FPGA
    # The actual handle is managed by xdnn
    config["xclbin"] = "../overlaybins/" + config["platform"] + "/overlay_1.xclbin" # Chosen Hardware Overlay
    return config

def getModelInputOutputNode(sProtobufPath):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        with gfile.FastGFile(sProtobufPath,'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]

        return graph_nodes[0].name, graph_nodes[len(graph_nodes)-1].name

# Quantize, and transfer the weights to FPGA DDR
def TransferWeightsFPGA(iBatchSize,config,handles,iPe):
    # config["datadir"] = "work/" + config["caffemodel"].split("/")[-1]+"_data" # From Compiler
    config["scaleA"] = 10000 # Global scaler for weights (Must be defined)
    config["scaleB"] = 30 # Global scaler for bias (Must be defined)
    config["PE"] = iPe # Run on Processing Element 0 - Different xclbins have a different number of Elements
    config["batch_sz"] = iBatchSize # We will load 1 image at a time from disk
     # We will resize images to 224x224
    return config

# Allocate space in host memory for inputs, Load images from disk
def AllocateMemoryToHost(config):

    # Allocate space in host memory for outputs
    if config["name"] == "googlenet":
        config["fpgaoutsz"] = 1024 # Number of elements in the activation of the last layer ran on the FPGA
    
    elif config["name"] == "ResNet50":
        config["fpgaoutsz"] = 2048 # Number of elements in the activation of the last layer ran on the FPGA

    config["outsz"] = 1000 # Number of elements output by FC layers (1000 used for imagenet)
    config["fpgaoutsz"] = 1024
    fpgaOutput = np.empty ((config["batch_sz"], config['fpgaoutsz'],), dtype=np.float32, order='C') # Space for fpga output
    fcOutput = np.empty((config["batch_sz"], config['outsz'],), dtype=np.float32, order='C') # Space for output of inner product
   
    return fpgaOutput, fcOutput,config

def generateRandomBatch(iBatchSize,config):
    return np.random.rand(iBatchSize,3,224,224).astype(np.float32)


def executeOnFPGA(sProtoBufPath,Qmode,Inference_Data,handle,name,num_models):
    TOTAL_IMAGES = 128;
    
    # Create handle for FPGA
    ret, handle=xdnn.createHandle("../overlaybins/" + "aws" + "/overlay_1.xclbin","kernelSxdnn_0")
    
    #Initialize objects to store results
    fpgaRT          = {}
    fpgaOutput     = {}
    fcWeight       = {}
    fcBias         = {}
    netFiles        = {}
    confNames       = []
    
    #Generate batch
    batch_array=generateRandomBatch(TOTAL_IMAGES,None) 
    
    #Get Image batch to start inference
    
    for i in range(0,num_models):    
        confNames += [str(i)]
        #Generate batch 10 * batchsize
        config=initializeFpgaModel(sProtoBufPath,Qmode)
        config["PE"]=i
        config["name"]=config["name"]+"_"+str(i);
        # Load weights to FPGA
        config=TransferWeightsFPGA(len(batch_array),config,handle,i)
        fpgaRT[str(i)] = xdnn.XDNNFPGAOp(handle,config)
        (fcWeight[str(i)],fcBias[str(i)])=xdnn_io.loadFCWeightsBias(config)    
        fpgaOutput[str(i)], fcOutput,config=AllocateMemoryToHost(config)
        
                       
    start0 = time.time()
    # Schedule FPGA execution asynchronously
    for i in range(0,num_models):
        fpgaRT[str(i)].exec_async(batch_array,fpgaOutput[str(i)],i)
    
    start1 = time.time()
    
    #Fetch results of all parallel executions
    for i in range(0,num_models):
        #Get FPGA output
        ret = fpgaRT[str(i)].get_result(i)
        #Compute Inner product - fully connected layer
        xdnn.computeFC(fcWeight[str(i)],fcBias[str(i)],fpgaOutput[str(i)],
        config['batch_sz'],config['outsz'],config['fpgaoutsz'],fcOutput)
        #Compute output softmax
        softmaxOut = xdnn.computeSoftmax(fcOutput)

    #xdnn_io.printClassification(softmaxOut, config['images'], labels);
    end = time.time()
    print("throughput",(num_models*len(batch_array)/(end-start0)),"duration",end-start0)
    Inference_result=[]
    #Append results
    Inference_Data.append({"experiment":str(Qmode)+"_bit_mode","duration_overall":end-start0,"imgsPerSecAll":num_models*len(batch_array)/(end-start0),"num_models_parallel":num_models})
    xdnn.closeHandle() 
    
    Inference_Data = pd.DataFrame(Inference_Data)
#    Inference_Data.to_csv('multinet_results.csv')
    result = pd.read_csv('multinet_results.csv');
    result=result.append(Inference_Data);
    result.to_csv('multinet_results.csv')
    #Inference_Data.to_csv('inference_'++str(q_Mode)+'.csv')


#Provide the Model checkpoint path

sProtoBufPath="/home/centos/models/tensorflow/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_test.pb"
Inference_Data =[]
q_Mode = 16  
num_models=1
executeOnFPGA(sProtoBufPath,q_Mode,Inference_Data,None,"G1",num_models)