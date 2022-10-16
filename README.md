# RebarStateModel
Tiny-YOLO algrithm trained on images of rebar for utilization on low performance equipment or CPU.

Prerequisite: 

Microsoft Visual Studio 2015 or 2017 | Python 3.6 | Tensorflow 1.09 GPU | CUDNN 9.1 | Open CV 

Hardware: Web Cam, any CPU, GPU (Any NVidia including mobile, and mx ranges)

Livetrack.py loads the algorithm from cfg folder, and weights from ckpt folder, captures frame from web cam via Open CV.
![image](https://user-images.githubusercontent.com/30218570/156555893-d7597153-bdf3-49bc-953d-812aec7a3387.png)
![image](https://user-images.githubusercontent.com/30218570/156555927-136d74f8-6a39-4097-a0b5-0cbb48948e28.png)

### Screenshot from the WebCam
![image](https://user-images.githubusercontent.com/30218570/156556117-6600c643-7959-49b9-9a95-a070e1ed56a1.png)

### Performance Analysis (nVidia M850 GPU)
![image](https://user-images.githubusercontent.com/30218570/156557018-42f4ecfc-3964-42ba-824e-822b9672d9ec.png)
