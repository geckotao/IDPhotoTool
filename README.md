# IDPhotoTool   证件照生成工具

让AI参考HivisionIDPhotos写的GUI程序，我只是移除部份功能并调试

Python 3.11.9版本

创建虚拟环境

python -m venv id_photo_env

激活虚拟环境（Windows）

id_photo_env\Scripts\activate

安装必要库

pip install -r requirements.txt

打包成EXE

PyInstaller -F -w   --icon=IDPhotoTool.ico  --add-data "IDPhotoTool.ico:."  --name="IDPhotoTool"   app.py

制作此软件原因与目的：帮一个负责学籍管理的老师专门制作，用来批量处理生成符合学籍系统规格的相片（HivisionIDPhotos无批量功能），而且由于工作电脑配置较差，所以去除了rmbg-1.4.onnx、birefnet-v1-lite.onnx等模型。
