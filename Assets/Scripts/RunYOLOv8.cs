using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis.Layers;

public class RunYOLOv8 : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Model model;
    private IWorker worker;
    //For using tensor operators:
    Ops ops;
    public CameraUpdate cam;
    public Image boxPrefbe;
    public int maxOutputBoxesPerClass = 20;
    public float iouThreshold = 0.5f;
    public float scoreThreshold = 0.7f;
    private List<Image> boxes = new List<Image>();
    string[] _labels = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                        "scissors", "teddy bear", "hair drier", "toothbrush"};
    
    List<(int, int, int)> _labelColors = new List<(int, int, int)>{(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                                                                    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                                                                    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
                                                                    (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
                                                                    (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
                                                                    (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
                                                                    (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
                                                                    (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
                                                                    (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
                                                                    (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
                                                                    (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
                                                                    (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
                                                                    (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
                                                                    (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
                                                                    (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
                                                                    (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
                                                                    (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
                                                                    (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
                                                                    (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
                                                                    (246, 0, 122), (191, 162, 208)};
    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        ops = WorkerFactory.CreateOps(BackendType.GPUCompute, null);
        
        //Set input
        model.AddInput("maxOutputBoxesPerClass",DataType.Int, new SymbolicTensorShape(1));
        model.AddInput("iouThreshold",DataType.Float, new SymbolicTensorShape(1));
        model.AddInput("scoreThreshold",DataType.Float, new SymbolicTensorShape(1));

        //Set constants   
        model.AddConstant(new Constant("0", new int[] { 0 }));
        model.AddConstant(new Constant("1", new int[] { 1 }));
        model.AddConstant(new Constant("4", new int[] { 4 }));
        model.AddConstant(new Constant("84", new int[] { 84 }));

        //Add layers    
        model.AddLayer(new Slice("boxCoords0", "output0", "0", "4", "1")); //1x84x8400 -> 1x4x8400
        model.AddLayer(new Transpose("boxCoords", "boxCoords0", new int[] { 0, 2, 1 })); // 1x4x8400 -> 1x8400x4

        model.AddLayer(new Slice("scores0", "output0", "4", "84", "1"));  //1x84x8400 -> 1x80x8400
        model.AddLayer(new ReduceMax("scores", new[] { "scores0", "1" }));  //1x80x8400 -> 1x1x8400   最有可能类别的置信度
        model.AddLayer(new ArgMax("classIDs", "scores0", 1)); //1x80x8400 -> 1x1x8400 最有可能类别的index

        model.AddLayer(new NonMaxSuppression("nmsOutput", "boxCoords", "scores",
            "maxOutputBoxesPerClass", "iouThreshold", "scoreThreshold",
            centerPointBox: CenterPointBox.Center
        ));

        model.AddOutput("boxCoords");
        model.AddOutput("classIDs");
        model.AddOutput("nmsOutput");   
    }

    public void Predict(WebCamTexture camImage)
    {
        using Tensor inputImage = TextureConverter.ToTensor(camImage,width:640,height:640, channels: 3);
        var m_Inputs = new Dictionary<string, Tensor>
        {
            {"images", inputImage },
            {"maxOutputBoxesPerClass", new TensorInt(new TensorShape(1), new int[] { maxOutputBoxesPerClass })},
            {"iouThreshold", new TensorFloat(new TensorShape(1), new float[] { iouThreshold })},
            {"scoreThreshold",new TensorFloat(new TensorShape(1), new float[] { scoreThreshold })}
        };
        worker.Execute(m_Inputs);
        var boxCoords = worker.PeekOutput("boxCoords") as TensorFloat;  //1x8400x4 所有的预测框的xywh
        var nmsOutput = worker.PeekOutput("nmsOutput") as TensorInt;  //Nx3, N指的是最终保留了几个框. 返回结果为[boxID,clsID,boxCoordID]   
        var classIDs = worker.PeekOutput("classIDs") as TensorInt; //1x1x8400  每个预测框的类别id

        ClearBoundingBoxes();
        if(nmsOutput.shape[0] == 0)//如果一个框也未剩下 则终止
        {
            return;
        }

        var boxCoordIDs = ops.Slice(nmsOutput, new int[] { 2 }, new int[] { 3 }, new int[] { 1 }, new int[] { 1 }); //Nx3 -> Nx1  取出boxCoordID
        var boxCoordIDsFlat = boxCoordIDs.ShallowReshape(new TensorShape(boxCoordIDs.shape.length)) as TensorInt;  //Nx1 -> N  展平
        var output = ops.Gather(boxCoords, boxCoordIDsFlat, 1) as TensorFloat; //1x8400x4 -> 1xNx4
        var labelIDs = ops.Gather(classIDs, boxCoordIDsFlat, 2) as TensorInt; //1x1x8400 -> N

        output.MakeReadable();
        labelIDs.MakeReadable();

        for(int i = 0; i<output.shape[1]; i++)
        {
            float x=output[0,i,0];
            float y=output[0,i,1];
            float w=output[0,i,2];
            float h=output[0,i,3];

            DrawBoundingBoxes(x,y,w,h,labelIDs[i]);//绘制一个Bounding Box
        }
    }

    public void ClearBoundingBoxes()
    {
        foreach (var obj in boxes)
        {
            Destroy(obj.gameObject);
        }
        boxes.Clear(); // 清空列表
    }

    public void DrawBoundingBoxes(float x, float y, float w, float h,int clsID)
    {
        Transform rawImage = cam.rawImage.transform;
        RectTransform camImageRect = rawImage.GetComponent<RectTransform>();
        Image box = Instantiate(boxPrefbe.gameObject,rawImage).GetComponent<Image>();

        float wRatio = camImageRect.sizeDelta.x / 640.0f;
        float hRatio = camImageRect.sizeDelta.y / 640.0f;
        box.rectTransform.localPosition = new Vector2((x - 640.0f/2.0f)*wRatio, -(y - 640.0f/2.0f)*hRatio);
        box.GetComponent<RectTransform>().sizeDelta = new Vector2(w * wRatio,h * hRatio);   
        
        //给不同标签的BoundingBoxes设置不同的颜色
        box.color = new Color( _labelColors[clsID].Item1 / 255.0f, 
                               _labelColors[clsID].Item2 / 255.0f, 
                               _labelColors[clsID].Item3 / 255.0f, 
                               0.25f);;

        Text textPrefbe = box.GetComponent<Box>().textPrefbe;
        textPrefbe.text = _labels[clsID];
        textPrefbe.color = new Color( _labelColors[clsID].Item1 / 255.0f, 
                               _labelColors[clsID].Item2 / 255.0f, 
                               _labelColors[clsID].Item3 / 255.0f, 
                               1.0f);
        boxes.Add(box);
    }

    // Update is called once per frame
    void Update()
    {
        if(cam.webCamTexture != null && cam.webCamTexture)
        {
            Predict(cam.webCamTexture);
        }

        
        maxOutputBoxesPerClass = (int)GameObject.Find("MaxOutputBoxesPerClass").GetComponent<Slider>().value;
        iouThreshold = GameObject.Find("IouThreshold").GetComponent<Slider>().value;
        scoreThreshold = GameObject.Find("ScoreThreshold").GetComponent<Slider>().value;
        GameObject.Find("MaxOutputBoxesPerClassValue").GetComponent<Text>().text = maxOutputBoxesPerClass.ToString();
        GameObject.Find("IouThresholdValue").GetComponent<Text>().text = iouThreshold.ToString();
        GameObject.Find("ScoreThresholdValue").GetComponent<Text>().text = scoreThreshold.ToString();
        
    }
}
