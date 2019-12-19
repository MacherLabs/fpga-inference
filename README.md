# FPGA-inference
<b>This experiment was done to check performance of Large machine learning tensorflow/caffe models over FPGA using the Xilinx stack on AWS</b><br>
<b>Prerequisites</b><br>
Installed libraries -Xilinx ml-suite, Tensorflow,caffe,jupyter notebook<br>
Open connections to  AWS ml-suite AMI and run the following commands in the terminal : <br>
sudo su; source ~centos/.bashrc; source activate ml-suite; source ml-suite/overlaybins/setup.sh aws<br>
Now you can execute the jupyter notebooks<br>
<b>inference_results</b> folder has the data of the experiment results for inception v1 and v3 models with 8 bit and 16 bit quantizations<Br>
<b>inceptionv1-tensorflow-final-inference notebook</b>- allows to run 8-bit/16 bit quantized inception v1 model. It can be also used to run any supported tensorflow model on xilinx<Br>
<b>inceptionv3-caffe-final-inference notebook</b> - allows to run 8-bit/16 bit quantized inception v3 model. It can be also used to run any supported Caffe model on xilinx<Br>
<b>tensorflow-inference-multinet.py</b> - Allows you to run upto 4 Inception v1 models in parallel. It can also allow running 4 different models in parallel say v1 and  v3 together.<br>
<b>fpga_utils.py</b> - Contains helper wrapper functions using Xilinx API to communicate with FPGA<br>
<b>Limitations_xilinx.txt</b> - Limitations of the Xilinx stack.
