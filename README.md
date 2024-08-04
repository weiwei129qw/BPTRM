# BPTRM
<div align="center">
    <p>
    <h1>
    VulDecgre - Implementation
    </h1>
    <img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version">
    <img src="https://img.shields.io/badge/Python-3.7+-orange" alt="version">
    <img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit">
</div>

## 📥 Setup
#### 1、Clone this repo

- `$ git clone https://github.com/SoftEngineerTeam/VulDecgre.git`

#### 2、Install Prerequisites

- `$ pip install -r requirements.txt`

#### 3、Run the testcase

- `$ cd VulDegre/cli`
- `$ python train.py GGNN GraphBinaryClassification ../data/data/testcase`

#### 4、 Load trained model and predict

- `$ cd VulDegre/cli`
- `$ python test.py GGNN GraphBinaryClassification ../data/data/predict --storedModel_path "./trained_model/GGNN_GraphBinaryClassification_best.pkl"`

## 🚨 Guide

#### 1、Preprocessing

- (1) **Slicing data**:
  - `cd VulDecgre/Edge_processing/slicec_8edges_funcblock/src/main/java/slice`
  - Run `ClassifyFileOfProject.java` to extract C files.
  - Run `Main.java` to slice code functions.

- (2) **Extracting eight types of relationship edges**:
  - `cd VulDecgre/Edge_processing/slicec_8edges_funcblock/src/main/java/eightEdges`
  - We use Joern to generate the code structure graph and we provide a compiled version of Joern.
  - Run `Main.java` to extract edges.
  - Run `concateJoern.java` to concatenate all edges to the graph.

## 🤯 Dataset

To investigate the effectiveness of VulDecgre in vulnerability detection, we adopt three open-source vulnerability datasets and a self-collected dataset from these papers:

- Fan et al. [1]:
  - [Dataset Link](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing)

- Reveal [2]:
  - [Dataset Link](https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOyF)

- FFMPeg+Qemu [3]:
  - [Dataset Link](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF)

- Our self-collected dataset:
  - [Dataset Link](https://drive.google.com/file/d/1P0NsDzpL75g5-EKJ59qCYK384UE2ohZ5/view?usp=drive_link)

**References**:

[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.
