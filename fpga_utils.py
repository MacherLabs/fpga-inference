import os,sys,cv2
import numpy as np
from matplotlib import pyplot as plt
from xfdnn.tools.compile.bin.xfdnn_compiler_caffe import CaffeFrontend as xfdnnCompilerCaffe
from xfdnn.tools.quantize.quantize import CaffeFrontend as xfdnnQuantizerCaffe

from xfdnn.tools.compile.bin.xfdnn_compiler_tensorflow import TFFrontend as xfdnnCompilerTensorflow
from xfdnn.tools.quantize.quantize_tf import tf_Quantizer as xfdnnQuantizerTensorflow

import xfdnn.rt.xdnn as xdnn
import xfdnn.rt.xdnn_io as xdnn_io
import time
import ipywidgets
import warnings
import time
import gc
import pandas as pd
warnings.simplefilter("ignore", UserWarning)

# @output : FPFA Confif, FPGA handle
def initializeFpgaModelTensorflow(sProtoBufPath, qMode, doCompile):
    config = {} # Config dict
    config["platform"] = 'aws'
    
    sInputNode,sOutputNode = getModelInputOutputNode(sProtoBufPath)
    # Compiler Arguments
    config["model"] = "GoogLeNet"
    config["protobuf"] = sProtoBufPath
    config["netcfg"] = "work/fpga_googleNet_"+str(qMode)+".cmds" # Compiler will generate FPGA instructions
    config["memory"] = 5 # Available on-chip SRAM
    config["dsp"] = 56 # Width of Systolic Array
    config["finalnode"] = sOutputNode # Terminal node in your tensorflow graph
   
    compiler = xfdnnCompilerTensorflow(
        networkfile=config["protobuf"],      # Protobuf filename: input file
        #anew=config["outmodel"],            # String for intermediate protobuf NOT YET SUPPORTED
        generatefile=config["netcfg"],       # Script filename: output file
        memory=config["memory"],             # Available on chip SRAM within xclbin
        dsp=config["dsp"],                   # Rows in DSP systolic array within xclbin # keep defaults 
        finalnode=config["finalnode"],       # Terminal node in your tensorflow graph
        weights=True                         # Instruct Compiler to generate a weights directory for runtime
    )

    # Invoke compiler
    try:
        if doCompile==True:
            compiler.compile()

        # The compiler extracts the floating point weights from the .caffemodel. 
        # This weights dir will be stored in the work dir with the appendex '_data'. 
        # The compiler will name it after the caffemodel, and append _data
        config["datadir"] = "work/" + os.path.basename(config["protobuf"])+"_data"

        if os.path.exists(config["datadir"]) and os.path.exists(config["netcfg"]+".json"):
            print("Compiler successfully generated JSON and the data directory: {:s}".format(config["datadir"]))
        else:
            print("Compiler failed to generate the JSON or data directory: {:s}".format(config["datadir"]))
            raise

        print("**********\nCompilation Successful!\n")

        import json
        data = json.loads(open(config["netcfg"]+".json").read())
        print("Network Operations Count: {:d}".format(data['ops']))
        print("DDR Transfers (bytes): {:d}".format(data['moveops']))

    except Exception as e:
        print("Failed to complete compilation:",e)

    # Quantizing
    config["img_mean"] = [104.007, 116.669, 122.679] # Mean of the training set
    config["quantizecfg"] = "work/quantization_params_googleNet_"+str(qMode)+".json" # Quantizer will generate quantization params
    config["calibration_directory"] = "../xfdnn/tools/quantize/calibration_directory" # Directory of images for quantizer
    config["calibration_size"] = 15 # Number of calibration images quantizer will use
    config["bitwidths"] = [qMode,qMode,qMode] # Supported quantization precision
    config["img_raw_scale"] = 255.0 # Raw scale of input pixels, i.e. 0 <-> 255
    config["img_input_scale"] = 1.0 # Input multiplier, Images are scaled by this factor after mean subtraction

    # Invoke quantizer
    quantizer = xfdnnQuantizerTensorflow(
        model_file=config["protobuf"],          # Prototxt filename: input file
        quantize_config=config["quantizecfg"],  # Quant filename: output file
        bitwidths=config["bitwidths"],          # Fixed Point precision: 8b or 16b
        cal_size=config["calibration_size"],    # Number of calibration images to use
        img_mean=config["img_mean"],            # Image mean per channel to caffe transformer
        cal_dir=config["calibration_directory"] # Directory containing calbration images
    )

    # Invoke quantizer
    try:
        if doCompile==True:
            quantizer.quantize(inputName = sInputNode, outputName = sOutputNode)

        import json
        data = json.loads(open(config["quantizecfg"]).read())
        print("**********\nSuccessfully produced quantization JSON file for %d layers.\n"%len(data['network']))
    except Exception as e:
        print("Failed to quantize:",e)

    ## NOTE: If you change the xclbin, we likely need to change some arguments provided to the compiler
    if qMode == 16:
        config["xclbin"] = "../overlaybins/" + config["platform"] + "/overlay_3.xclbin" # Chosen Hardware Overlay
    if qMode == 8:
        config["xclbin"] = "../overlaybins/" + config["platform"] + "/overlay_0.xclbin" # Chosen Hardware Overlay
   
    # Create a handle with which to communicate to the FPGA
    ret, handle = xdnn.createHandle(config['xclbin'],"kernelSxdnn_0")
    if ret:                                                             
        print("ERROR: Unable to create handle to FPGA")
    else:
        print("INFO: Successfully created handle to FPGA")
    

    # If this step fails, most likely the FPGA is locked by another user, or there is some setup problem with the hardware
    return config, handle



def initializeFpgaModelCaffe(sProtoBufPath,qMode,doCompile):
    config = {} # Config dict
    config["platform"] = 'aws'   
    config["model"] = "inception_v3"
    config["prototxt"]  =sProtoBufPath+"_deploy.prototxt"
    config["caffemodel"]=sProtoBufPath+".caffemodel"
    config["outmodel"]    = "work/opt_inception_model"
    
    config["netcfg"] = "work/fpga_inception_v3.cmds" # Compiler will generate FPGA instructions
    config["memory"] = 5 # Available on-chip SRAM
    config["dsp"] = 56 # Width of Systolic Array
    
    
    compiler = xfdnnCompilerCaffe(
        networkfile=config["prototxt"],      # Protobuf filename: input file
        anew=config["outmodel"],            # String for intermediate protobuf NOT YET SUPPORTED
        generatefile=config["netcfg"],       # Script filename: output file
        memory=config["memory"],             # Available on chip SRAM within xclbin
        dsp=config["dsp"],                   # Rows in DSP systolic array within xclbin # keep defaults 
        weights=config["caffemodel"]                         # Instruct Compiler to generate a weights directory for runtime
    )

# Invoke compiler
    try:
        if doCompile == True:
            compiler.compile()

        # The compiler extracts the floating point weights from the .caffemodel.
        # As it makes optimizations it will augment the weights, and generate a weights dir
        # This weights dir will be stored in the work dir with the appendex '_data'. 
        # In the future, the compiler will generate a more efficient format such as hdf5
        config["datadir"] = "work/" + os.path.basename(config["caffemodel"]) + "_data"    
        if os.path.exists(config["datadir"]) and os.path.exists(config["netcfg"]+".json"):
            print("Compiler successfully generated JSON and the data directory: %s" % config["datadir"])
        else:
            print("Compiler failed to generate the JSON or data directory: %s" % config["datadir"])
            raise

        print("**********\nCompilation Successful!\n")

        import json
        data = json.loads(open(config["netcfg"]+".json").read())
        print("Network Operations Count: %d"%data['ops'])
        print("DDR Transfers (bytes): %d"%data['moveops']) 

    except Exception as e:
        print("Failed to complete compilation:",e)

    # Quantizing
    config["img_mean"] = [104.007, 116.669, 122.679] # Mean of the training set
    config["output_json"] = "work/quantization_params_caffe.json"
    config["quantizecfg"] =  config["output_json"] # Quantizer will generate quantization params
    config["calibration_directory"] = "../xfdnn/tools/quantize/calibration_directory" # Directory of images for quantizer
    config["calibration_size"] = 15 # Number of calibration images quantizer will use
    config["bitwidths"] = [qMode,qMode,qMode] # Supported quantization precision
    config["img_raw_scale"] = 255.0 # Raw scale of input pixels, i.e. 0 <-> 255
    config["img_input_scale"] = 1.0 # Input multiplier, Images are scaled by this factor after mean subtraction
    config["transpose"] = [2,0,1] # (H,W,C)->(C,H,W) transpose argument to quantizer
    config["channel_swap"] = [2,1,0] # (R,G,B)->(B,G,R) Channel Swap argument to quantizer


# Compiler instance
    quantizer = xfdnnQuantizerCaffe(
        deploy_model=config["outmodel"]+".prototxt",          # Model filename: input file
        weights=config["outmodel"]+".caffemodel",             # Floating Point weights
        output_json=config["output_json"],                    # Quantization JSON output filename
        bitwidths=config["bitwidths"],                        # Fixed Point precision: 8,8,8 or 16,16,16
        transpose=config["transpose"],                        # Transpose argument to caffe transformer
        channel_swap=config["channel_swap"],                  # Channel swap argument to caffe transfomer
        raw_scale=config["img_raw_scale"],                    # Raw scale argument to caffe transformer
        mean_value=config["img_mean"],                        # Image mean per channel to caffe transformer
        input_scale=config["img_input_scale"],                # Input scale argument to caffe transformer
        calibration_size=config["calibration_size"],          # Number of calibration images to use
        calibration_directory=config["calibration_directory"] # Directory containing calbration images
    )

    # Invoke quantizer
    try:
        if doCompile == True:
            quantizer.quantize()

        import json
        data = json.loads(open(config["quantizecfg"]).read())
        print("**********\nSuccessfully produced quantization JSON file for %d layers.\n"%len(data['network']))
    except Exception as e:
        print("Failed to quantize:",e)

    # Create a handle with which to communicate to the FPGA
    # The actual handle is managed by xdnn
    if qMode == 16:
        config["xclbin"] = "../overlaybins/" + config["platform"] + "/overlay_3.xclbin" # Chosen Hardware Overlay
    if qMode == 8:
        config["xclbin"] = "../overlaybins/" + config["platform"] + "/overlay_0.xclbin" # Chosen Hardware Overlay

    ret, handles = xdnn.createHandle(config['xclbin'],"kernelSxdnn_0")

    if ret:                                                             
        print("ERROR: Unable to create handle to FPGA")
    else:
        print("INFO: Successfully created handle to FPGA")

    # If this step fails, most likely the FPGA is locked by another user, or there is some setup problem with the hardware
    return config,handles


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

# Allocate space in host memory for inputs, Load images from disk
def AllocateMemoryToHost(config,iBatchSize):

    # Allocate space in host memory for outputs
    if config["model"] == "GoogLeNet":
        config["fpgaoutsz"] = 1024 # Number of elements in the activation of the last layer ran on the FPGA
    elif config["model"] == "ResNet50":
        config["fpgaoutsz"] = 2048 # Number of elements in the activation of the last layer ran on the FPGA
    elif config["model"] == "inception_v3":
        config["fpgaoutsz"] = 8192 # Number of elements in the activation of the last layer ran on the FPGA

    config["outsz"] = 1000 # Number of elements output by FC layers (1000 used for imagenet)
    
    fpgaOutput = np.empty ((iBatchSize, config['fpgaoutsz'],), dtype=np.float32, order='C') # Space for fpga output
    fcOutput = np.empty((iBatchSize, config['outsz'],), dtype=np.float32, order='C') # Space for output of inner product
   
    return fpgaOutput, fcOutput,config

# Quantize, and transfer the weights to FPGA DDR
def TransferWeightsFPGA(iBatchSize,config,handles):
    # config["datadir"] = "work/" + config["caffemodel"].split("/")[-1]+"_data" # From Compiler
    config["scaleA"] = 10000 # Global scaler for weights (Must be defined)
    config["scaleB"] = 30 # Global scaler for bias (Must be defined)
    config["PE"] = 0 # Run on Processing Element 0 - Different xclbins have a different number of Elements
    config["batch_sz"] = iBatchSize # We will load 1 image at a time from disk
    config["in_shape"] = (3,224,224) # We will resize images to 224x224

    #(weightsBlob, fcWeight, fcBias ) = pyxfdnn_io.loadWeights(config)
    fpgaRT = xdnn.XDNNFPGAOp(handles,config)
    (fcWeight, fcBias) = xdnn_io.loadFCWeightsBias(config)
    
    return fpgaRT,fcWeight,fcBias,config

# Create Random batch array of given size
def generateRandomBatch(iBatchSize):
    return np.random.rand(iBatchSize,3,224,224).astype(np.float32)


def runOnFPGA( fpgaRT,fcOutput,fpgaOutput,iBatchSize,config,handle,batchArray,fcWeight, fcBias):    
    # Write FPGA Instructions to FPGA and Execute the network!
    start = time.time()
    #print("executing..",len(batchArray)," images")
    fpgaRT.execute(batchArray, fpgaOutput)
    
    #Compute inner Product - on CPU
    xdnn.computeFC(fcWeight, fcBias, fpgaOutput, len(batchArray), config['outsz'], config['fpgaoutsz'], fcOutput)
    
    # Compute the softmax to convert the output to a vector of probabilities -on CPU
    softmaxOut = xdnn.computeSoftmax(fcOutput)
    end = time.time()
    
    #Return the output and execution time
    return softmaxOut, end-start  

def executeOnFPGA(sProtoBufPath,Qmode,Inference_Data,num_images,doCompile,modelType):
    # Initialize the model, get the model specific config and FPGA handle.
    if modelType=='Tensorflow':
        config,handle=initializeFpgaModelTensorflow(sProtoBufPath,Qmode,doCompile)
    elif modelType =='Caffe':
        config,handle=initializeFpgaModelCaffe(sProtoBufPath,Qmode,doCompile)
    
    #Get Image batch to start inference
    batch_array=generateRandomBatch(num_images) 
    for i in range(0,7):    

        iBatchSize = 2**i

        # Load weights to FPGA
        fpgaRT,fcWeight,fcBias,config=TransferWeightsFPGA(iBatchSize,config,handle)

        #Allocate Memory to host
        fpgaOutput, fcOutput,config=AllocateMemoryToHost(config,iBatchSize)
        
        #Do some dry runs
        print('dry run..10 rounds')
        for k in range(0,10):
            _,_=runOnFPGA(fpgaRT,fcOutput,fpgaOutput,iBatchSize,config,handle,batch_array[0:iBatchSize],fcWeight, fcBias)
            
        print("starting prediction for batchsize : {} over {} images with 100 iterations".format(iBatchSize,len(batch_array)))

        # Do 100 runs for all the batch sizes
        for iRuns in range(0,50):
        #Run prdeiction on FPGA
            startIndex = 0
            endIndex = startIndex + iBatchSize;
            
            iTotalActualTime = 0
            startTime = time.time()
            while (endIndex<len(batch_array)): 
                #Run the model on FPGA and last layer of softmax on CPU
                out,actualTime = runOnFPGA(fpgaRT,fcOutput,fpgaOutput,iBatchSize,config,handle,batch_array[startIndex:endIndex],fcWeight, fcBias)
                iTotalActualTime = iTotalActualTime+actualTime
                startIndex = endIndex
                endIndex = endIndex+ iBatchSize

            endTime = time.time()
            # Calculate the duration
            duration = endTime-startTime
            
            #Append the result to the inference table
            Inference_Data.append({"experiment":str(Qmode)+"_bit_mode","duration_overall":duration, "duration_run_on_fpga":iTotalActualTime
                                                 ,"imgsPerSecAll": (len(batch_array)-iBatchSize)/duration,"batchSize":iBatchSize,
                                                  "imgsPerSecFPGA": (len(batch_array)-iBatchSize)/iTotalActualTime})

        del out,fpgaRT,fcWeight,fcBias,fcOutput,fpgaOutput
        gc.collect()
       
    del config
    return Inference_Data

def doAggregations(Inference_Data):
    # Calculate mean and standard deviation
    result=Inference_Data.groupby(['batchSize'],as_index=False).agg({'imgsPerSecAll':['mean','std'],'experiment':'max'})
    result.columns = [col[0]+"_"+col[1] for col in result.columns]
    #Max deviation
    result['imgsPerSecAll_max']=result['imgsPerSecAll_mean']+result['imgsPerSecAll_std']
    #Min Deviation
    result['imgsPerSecAll_min']=result['imgsPerSecAll_mean']-result['imgsPerSecAll_std']
    return result

#Method for plotting results
def plotSingleModelInference(model):
    # Load 8 bit results
    result= pd.read_csv(model+'_8bit.csv')
    # Load 16 bit results
    result16= pd.read_csv(model+'_16bit.csv')    
    # Append results
    result=result.append(result16)
    
    import matplotlib
    import seaborn as sns
    
    f, ax = plt.subplots(figsize=(20, 20))
    colors = ["#ff0000","#02fcef"]
    palette = sns.color_palette(colors)
    ax.set_xscale('log')
    ax.set_xticks([2**i for i in range(7)])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    #Plot all results
    sns.lineplot(x="batchSize_", y="imgsPerSecAll_mean", hue="experiment_max", palette=palette, ax=ax, markers=True, style=True, data=result)
    plt.legend(prop={'size': 15})
    sns.lineplot(x="batchSize_", y="imgsPerSecAll_max", hue="experiment_max", palette=palette, ax=ax, markers=True, style=True, legend=False, data=result, size=1)
    sns.lineplot(x="batchSize_", y="imgsPerSecAll_min", hue="experiment_max", palette=palette, ax=ax, markers=True, style=True, legend=False, data=result, size=1)
    
def plotMultiModelInference():
    import pandas as pd
    from matplotlib import pyplot as plt
    result= pd.read_csv('multinet_results.csv')
    import matplotlib
    import seaborn as sns
    f, ax = plt.subplots(figsize=(20, 20))
    colors = ["#ff0000","#02fcef"]
    palette = sns.color_palette(colors)
    #ax.set_xscale('log')
    ax.set_xticks([i+1 for i in range(4)])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    sns.lineplot(x="num_models_parallel", y="imgsPerSecAll", hue="experiment", palette=palette, ax=ax, markers=True, style=True, data=result)
    plt.legend(prop={'size': 15})
    
