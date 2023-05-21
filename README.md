<div align='center'>
    <h1 align='center'>Traffic Rules Violation Detection</h1>
    <hr style='width:100%'>
</div>

<div>
    <h2>Research Team: </h2>
    <p>Abdullah Al Nahian Kanon &nbsp(<a href='https://github.com/Nahian98'>Nahian98</a>) 🔥🔥🔥</p>
    <p>Md. Saidul Islam Akib &nbsp(<a href='https://github.com/Akib558'>Akib558</a>) 🔥🔥🔥</p>

</div>

<div>
    <h2>Resources:</h2>
    <ul>
    <li><p>Model Used for trainning is YOLOv5 => &nbsp(<a href='https://github.com/ultralytics/yolov5'>YOLOv5</a>)</p></li>
    <li><p>For uniquely giving id to each vehicle this repository is used => &nbsp(<a href='https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet'>StrongSort_OSNet</a>)<p></li>
    <li><p>Dataset used for training purpose => &nbsp(<a href='https://data.mendeley.com/datasets/pwyyg8zmk5/2'>Poribohon-BD</a>)</p></li>
    <li><p>Our Updated dataset => &nbsp(<a href=''>On development</a>)</p></li>
    <li><p>Slides and Reports => &nbsp(<a href=''>On development</a>)</p></li>
    </ul>
</div>



<h2>Running the program :</h2>
<p>1. Clone the repository: </p>

```console
    gti clone https://github.com/Akib558/traffic_rules_violation_detection.git   
```

<p>2. Move to the clone folder and open a terminal from that folder </p>
<p>3. Install all the requirements : </p>

```console
    pip install -r requirements.txt  
```

<p>4. After all the requirement is downloaded, then to run the program following format need to follow :</p>

```console
    ## format
    python track.py --source {video_name that need to be detected} --yolo-weights {yolov5 weights : best.pt or best_1.pt or best_2.pt} --save-vid
    ## demo
    python track.py --source vid_1.mkv --yolo-weights best.pt --save-vid 
```
