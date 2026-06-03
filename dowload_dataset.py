from roboflow import Roboflow

rf = Roboflow(api_key="2HWZKz4mE7I0TJUj4Aea")

project = rf.workspace("madhumitha-jc-hvsdd").project("candlestick-pattern")

dataset = project.version(1).download("yolov8")