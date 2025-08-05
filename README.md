# embedding-to-VectorDatabase
High-performance knowledge base data import

## baai_m3_server

### 生产启动docker服务命令

~~~
docker run -itd --net=host --restart always --shm-size=50g --name=Lantai_embedd2vector --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/devmm_svm --device=/dev/davinci{0,1,2,3,4,5,6,7} -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /etc/hccn.conf:/etc/hccn.conf -v /var/log/npu/:/usr/slog -v /etc/localtime:/etc/localtime -v /data2/embedding-to-VectorDatabase:/workspace -w /workspace -v /data2/embed:/data swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.T3.1-800I-A2-py311-openeuler24.03-lts
~~~

进入启动服务

~~~
docker exec -it embedding bash

cd baai_m3_simple_server

python3 m3_server.py
~~~

## 脚本启动服务

~~~
docker run -itd --net=host --restart always --shm-size=50g --name=Lantai_embedding2vector -v /data/Downloads/niuzm/embedding-to-VectorDatabase:/workspace -w /workspace embedvector:v1
conda activate embedding

# 在镜像中/ect/hosts
加入hadoop节点信息映射
~~~


## TEI bge-m3 server

~~~
docker run -u root -itd -e ENABLE_BOOST=True -e POOLING=splade -e CLEAN_NPU_CACHE=True -e ASCEND_VISIBLE_DEVICES=2 --name=LanTai_spase_tei --net=host -v /data/Downloads/niuzm/embedding_model/:/home/HwHiAiUser/model swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:6.0.0-800I-A2-aarch64  bge-m3 0.0.0.0 8181
~~~
