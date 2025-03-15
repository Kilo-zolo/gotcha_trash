using Unity.Barracuda;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Collections;
using System.Collections.Generic;

public class YoloDetector : MonoBehaviour {
    public NNModel yoloModel;  // Assign YOLOv8-n .onnx asset in Inspector
    public TextAsset labelsFile; // Assign label_map.txt in Inspector
    public ARCameraManager arCameraManager; // Assign AR Camera's ARCameraManager in Inspector

    private IWorker worker;
    private string[] labels;
    private Texture2D cameraTexture;
    private List<Detection> detections = new List<Detection>();  

    private int modelInputSize = 640;  // YOLOv8 input resolution 
    private float confidenceThreshold = 0.5f;

    void Start() {
        labels = labelsFile.text.Split('\n');
        var model = ModelLoader.Load(yoloModel);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
    }

    void OnEnable() {
        if (arCameraManager != null)
            arCameraManager.frameReceived += OnCameraFrameReceived;
    }

    void OnDisable() {
        if (arCameraManager != null)
            arCameraManager.frameReceived -= OnCameraFrameReceived;
    }

    private void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs) {
        if (arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage)) {
            if (cameraTexture == null || 
                cameraTexture.width != cpuImage.width || cameraTexture.height != cpuImage.height) {
                cameraTexture = new Texture2D(cpuImage.width, cpuImage.height, TextureFormat.RGBA32, false);
            }

            XRCpuImage.ConversionParams convParams = new XRCpuImage.ConversionParams {
                inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
                outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
                outputFormat = TextureFormat.RGBA32,
                transformation = XRCpuImage.Transformation.None
            };

            NativeArray<byte> rawTextureData = cameraTexture.GetRawTextureData<byte>();
            cpuImage.Convert(convParams, rawTextureData);
            cpuImage.Dispose();
            cameraTexture.Apply();

            RunYoloDetection(cameraTexture);
        }
    }

    private void RunYoloDetection(Texture2D camTex) {
        Color32[] pixels = ScaleImageToModel(camTex, modelInputSize);

        using (Tensor inputTensor = CreateInputTensor(pixels, modelInputSize, modelInputSize)) {
            worker.Execute(inputTensor);
        }
        
        Tensor outputTensor = worker.PeekOutput();
        detections = ProcessYoloOutput(outputTensor, confidenceThreshold);
        outputTensor.Dispose();
    }

    public struct Detection {
        public string label;
        public float confidence;
        public Rect bbox;  // Bounding box in normalized coordinates (0-1)
    }

    private List<Detection> ProcessYoloOutput(Tensor output, float confThreshold) {
        var detections = new List<Detection>();
        float[] data = output.ToReadOnlyArray();
    
        int numBoxes = 8400;  
        int valuesPerBox = 12;  
    
        for (int i = 0; i < numBoxes; i++) {
            int offset = i * valuesPerBox;
            
            float cx = data[offset + 0];  
            float cy = data[offset + 1];  
            float w = data[offset + 2];   
            float h = data[offset + 3];   
            float objConf = data[offset + 4];  
    
            if (objConf < confThreshold) continue;  

            int bestClass = -1;
            float bestClassConf = 0f;
            for (int c = 5; c < valuesPerBox; c++) {  
                float classConf = data[offset + c];
                if (classConf > bestClassConf) {
                    bestClassConf = classConf;
                    bestClass = c - 5; 
                }
            }
    
            float finalConf = objConf * bestClassConf;
            if (finalConf < confThreshold) continue;  

            float x = cx - (w / 2f);
            float y = cy - (h / 2f);
            float xMax = cx + (w / 2f);
            float yMax = cy + (h / 2f);
    
            x = Mathf.Clamp01(x);
            y = Mathf.Clamp01(y);
            xMax = Mathf.Clamp01(xMax);
            yMax = Mathf.Clamp01(yMax);
    
            Detection det = new Detection {
                label = labels[bestClass],
                confidence = finalConf,
                bbox = new Rect(x, y, xMax - x, yMax - y)
            };
            detections.Add(det);
        }
    
        return ApplyNMS(detections, 0.45f);
    }

    private Color32[] ScaleImageToModel(Texture2D camTex, int modelSize) {
        // Create a temporary RenderTexture
        RenderTexture rt = new RenderTexture(modelSize, modelSize, 24);
        RenderTexture.active = rt;
    
        // Blit (copy) camTex into rt
        Graphics.Blit(camTex, rt);
    
        // Create a new Texture2D and read from RenderTexture
        Texture2D scaledTex = new Texture2D(modelSize, modelSize, TextureFormat.RGBA32, false);
        scaledTex.ReadPixels(new Rect(0, 0, modelSize, modelSize), 0, 0);
        scaledTex.Apply();
    
        // Clean up
        RenderTexture.active = null;
        rt.Release();
        Destroy(rt);
    
        return scaledTex.GetPixels32();
    }


    private Tensor CreateInputTensor(Color32[] pixels, int width, int height) {
        float[] inputData = new float[width * height * 3];
        for (int i = 0; i < pixels.Length; i++) {
            Color32 c = pixels[i];
            inputData[i * 3 + 0] = c.r / 255f;
            inputData[i * 3 + 1] = c.g / 255f;
            inputData[i * 3 + 2] = c.b / 255f;
        }
        return new Tensor(1, height, width, 3, inputData);
    }

    private List<Detection> ApplyNMS(List<Detection> detections, float iouThreshold) {
        detections.Sort((a, b) => b.confidence.CompareTo(a.confidence));
        List<Detection> finalDetections = new List<Detection>();

        for (int i = 0; i < detections.Count; i++) {
            Detection a = detections[i];
            bool shouldKeep = true;

            for (int j = 0; j < finalDetections.Count; j++) {
                Detection b = finalDetections[j];

                float iou = ComputeIoU(a.bbox, b.bbox);
                if (iou > iouThreshold) {
                    shouldKeep = false;
                    break;
                }
            }
            if (shouldKeep) finalDetections.Add(a);
        }

        return finalDetections;
    }

    private float ComputeIoU(Rect a, Rect b) {
        float x1 = Mathf.Max(a.xMin, b.xMin);
        float y1 = Mathf.Max(a.yMin, b.yMin);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);
        float interArea = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        float unionArea = (a.width * a.height) + (b.width * b.height) - interArea;
        return interArea / unionArea;
    }

    void OnGUI() {
        foreach (var det in detections) {
            Rect screenRect = new Rect(
                det.bbox.x * Screen.width,
                det.bbox.y * Screen.height,
                det.bbox.width * Screen.width,
                det.bbox.height * Screen.height
            );

            GUI.Box(screenRect, $"{det.label} ({det.confidence:P1})");
        }
    }
}
