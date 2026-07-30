#if !UNITY_WSA_10_0

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using OpenCVRange = OpenCVForUnity.CoreModule.Range;
using OpenCVRect = OpenCVForUnity.CoreModule.Rect;

namespace GotchaTrash
{
    [RequireComponent(typeof(MultiSource2MatHelper))]
    public class TrashCameraView : MonoBehaviour
    {
        [Header("Model (StreamingAssets-relative)")]
        public string modelPath = "OpenCVForUnity/dnn/best.onnx";
        public string classesPath = "OpenCVForUnity/dnn/trash.names";
        public int inputWidth = 640;
        public int inputHeight = 640;
        public float confThreshold = 0.25f;
        public float nmsThreshold = 0.45f;
        public int topK = 300;

        [Header("Output")]
        public RawImage resultPreview;
        public TMP_Text scoreText;

        [Header("Scoring")]
        [Tooltip("Minimum IoU for a new detection to match an existing track.")]
        public float trackMatchIouThreshold = 0.3f;
        [Tooltip("Fraction of a trash box that must be inside a Bin box to count as 'binned'.")]
        public float binContainmentThreshold = 0.7f;
        [Tooltip("Frames a track can be unseen before it dies (scores if wasBinned).")]
        public int trackDeathFrames = 15;

        public int Score { get; private set; }

        // Glass=0, Metal=1, Paper=2, Plastic=3, Waste=4, Bin=5, Trash=6, Person=7
        private static readonly HashSet<int> TrashClassIds = new HashSet<int> { 0, 1, 2, 3, 4, 6 };
        private const int BinClassId = 5;

        private MultiSource2MatHelper _source;
        private Net _net;
        private List<string> _classNames = new List<string>();
        private int _numClasses = 8;

        private Mat _bgrMat;
        private Mat _paddedImg;
        private Texture2D _texture;
        private List<Scalar> _palette;

        private readonly List<TrashTrack> _activeTracks = new List<TrashTrack>();
        private int _nextTrackId = 1;

        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private bool _initialized;

        private async void Start()
        {
            _source = GetComponent<MultiSource2MatHelper>();
            _source.OutputColorFormat = Source2MatHelperColorFormat.RGBA;

            string resolvedModel = await OpenCVEnv.GetFilePathTaskAsync(modelPath, cancellationToken: _cts.Token);
            if (string.IsNullOrEmpty(resolvedModel))
            {
                Debug.LogError($"[TrashCameraView] model not found under StreamingAssets: {modelPath}");
                return;
            }

            string resolvedClasses = await OpenCVEnv.GetFilePathTaskAsync(classesPath, cancellationToken: _cts.Token);
            if (!string.IsNullOrEmpty(resolvedClasses))
            {
                _classNames = File.ReadAllLines(resolvedClasses)
                    .Where(l => !string.IsNullOrWhiteSpace(l))
                    .ToList();
                _numClasses = _classNames.Count;
            }
            else
            {
                Debug.LogWarning($"[TrashCameraView] classes file not found: {classesPath}. Labels will be numeric.");
            }

            _net = Dnn.readNet(resolvedModel);
            _net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
            _net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

            _palette = BuildPalette();
            _initialized = true;

            _source.Initialize();
        }

        // Wire these three to the MultiSource2MatHelper's UnityEvents in the Inspector.
        public void OnSourceToMatHelperInitialized()
        {
            Mat rgba = _source.GetMat();
            _texture = new Texture2D(rgba.cols(), rgba.rows(), TextureFormat.RGBA32, false);
            OpenCVMatUtils.MatToTexture2D(rgba, _texture);

            if (resultPreview != null)
            {
                resultPreview.texture = _texture;
                var arf = resultPreview.GetComponent<AspectRatioFitter>();
                if (arf != null) arf.aspectRatio = (float)_texture.width / _texture.height;
            }

            _bgrMat = new Mat(rgba.rows(), rgba.cols(), CvType.CV_8UC3);
        }

        public void OnSourceToMatHelperDisposed()
        {
            _bgrMat?.Dispose(); _bgrMat = null;
            if (_texture != null) { Destroy(_texture); _texture = null; }
        }

        public void OnSourceToMatHelperErrorOccurred(Source2MatHelperErrorCode code, string message)
        {
            Debug.LogError($"[TrashCameraView] source error {code}: {message}");
        }

        private void Update()
        {
            if (!_initialized || _net == null || _bgrMat == null) return;
            if (!_source.IsPlaying() || !_source.DidUpdateThisFrame()) return;

            Mat rgba = _source.GetMat();
            Imgproc.cvtColor(rgba, _bgrMat, Imgproc.COLOR_RGBA2BGR);

            List<Det> detections = Infer(_bgrMat);
            UpdateTracksAndScore(detections);
            DrawDetections(rgba, detections);

            OpenCVMatUtils.MatToTexture2D(rgba, _texture);

            if (scoreText != null) scoreText.text = Score.ToString();
        }

        private void OnDestroy()
        {
            _cts.Cancel();
            _source?.Dispose();
            _net?.Dispose();
            _bgrMat?.Dispose();
            _paddedImg?.Dispose();
            if (_texture != null) Destroy(_texture);
            _cts.Dispose();
        }

        private struct Det
        {
            public float x1, y1, x2, y2;
            public float conf;
            public int cls;
        }

        private List<Det> Infer(Mat image)
        {
            // Corner-aligned letterbox: pad up to a square that is `ratio`x input_size,
            // then blobFromImage resizes to input_size. Boxes come back in image-space
            // when multiplied by `ratio`.
            float ratio = Mathf.Max((float)image.cols() / inputWidth, (float)image.rows() / inputHeight);
            int padW = Mathf.CeilToInt(inputWidth * ratio);
            int padH = Mathf.CeilToInt(inputHeight * ratio);

            if (_paddedImg == null || _paddedImg.cols() != padW || _paddedImg.rows() != padH)
            {
                _paddedImg?.Dispose();
                _paddedImg = new Mat(padH, padW, image.type(), Scalar.all(114));
            }
            else
            {
                Imgproc.rectangle(_paddedImg, new OpenCVRect(0, 0, padW, padH), Scalar.all(114), -1);
            }

            using (Mat roi = new Mat(_paddedImg, new OpenCVRect(0, 0, image.cols(), image.rows())))
            {
                image.copyTo(roi);
            }

            Mat blob = Dnn.blobFromImage(_paddedImg, 1.0 / 255.0, new Size(inputWidth, inputHeight),
                                         Scalar.all(0), true, false, CvType.CV_32F);
            _net.setInput(blob);

            List<Mat> outputs = new List<Mat>();
            _net.forward(outputs, _net.getUnconnectedOutLayersNames());
            blob.Dispose();

            List<Det> results = DecodeYolov8(outputs[0]);
            foreach (Mat o in outputs) o.Dispose();

            for (int i = 0; i < results.Count; i++)
            {
                Det d = results[i];
                d.x1 = Mathf.Round(d.x1 * ratio);
                d.y1 = Mathf.Round(d.y1 * ratio);
                d.x2 = Mathf.Round(d.x2 * ratio);
                d.y2 = Mathf.Round(d.y2 * ratio);
                results[i] = d;
            }

            return results;
        }

        private List<Det> DecodeYolov8(Mat outputBlob)
        {
            // Ultralytics YOLOv8 ONNX output: [1, 4+C, N]
            int channels = outputBlob.size(1);
            int numAnchors = outputBlob.size(2);
            int detected = channels - 4;
            if (detected != _numClasses)
            {
                Debug.LogWarning($"[TrashCameraView] class-count mismatch: model={detected}, file={_numClasses}");
                _numClasses = detected;
            }

            // [1, C, N] -> [C, N] -> transpose -> [N, C]. Each row is one anchor: [cx,cy,w,h, cls...].
            Mat reshaped = outputBlob.reshape(1, channels);
            Mat transposed = new Mat();
            Core.transpose(reshaped, transposed);

            Mat boxCols = transposed.colRange(new OpenCVRange(0, 4));
            Mat classCols = transposed.colRange(new OpenCVRange(4, 4 + _numClasses));

            List<Rect2d> boxes = new List<Rect2d>();
            List<float> confs = new List<float>();
            List<int> classes = new List<int>();

            for (int i = 0; i < numAnchors; i++)
            {
                using (Mat row = classCols.row(i))
                {
                    Core.MinMaxLocResult mm = Core.minMaxLoc(row);
                    float best = (float)mm.maxVal;
                    if (best < confThreshold) continue;

                    float[] b = new float[4];
                    boxCols.get(i, 0, b);
                    double x = b[0] - b[2] * 0.5;
                    double y = b[1] - b[3] * 0.5;
                    boxes.Add(new Rect2d(x, y, b[2], b[3]));
                    confs.Add(best);
                    classes.Add((int)mm.maxLoc.x);
                }
            }

            transposed.Dispose();

            List<Det> results = new List<Det>();
            if (boxes.Count == 0) return results;

            using (MatOfRect2d boxMat = new MatOfRect2d(boxes.ToArray()))
            using (MatOfFloat confMat = new MatOfFloat(confs.ToArray()))
            using (MatOfInt classMat = new MatOfInt(classes.ToArray()))
            using (MatOfInt indices = new MatOfInt())
            {
                Dnn.NMSBoxesBatched(boxMat, confMat, classMat, confThreshold, nmsThreshold,
                                    indices, 1f, topK);
                int[] idx = indices.toArray();
                foreach (int i in idx)
                {
                    Rect2d r = boxes[i];
                    results.Add(new Det
                    {
                        x1 = (float)r.x,
                        y1 = (float)r.y,
                        x2 = (float)(r.x + r.width),
                        y2 = (float)(r.y + r.height),
                        conf = confs[i],
                        cls = classes[i]
                    });
                }
            }

            return results;
        }

        private void UpdateTracksAndScore(List<Det> data)
        {
            List<Det> trashDets = new List<Det>();
            List<Det> binDets = new List<Det>();
            foreach (Det d in data)
            {
                if (TrashClassIds.Contains(d.cls)) trashDets.Add(d);
                else if (d.cls == BinClassId) binDets.Add(d);
            }

            foreach (TrashTrack t in _activeTracks) t.updatedThisFrame = false;

            foreach (Det d in trashDets)
            {
                TrashTrack best = null;
                float bestIou = trackMatchIouThreshold;
                foreach (TrashTrack t in _activeTracks)
                {
                    if (t.classId != d.cls || t.updatedThisFrame) continue;
                    float iou = Iou(d.x1, d.y1, d.x2, d.y2, t.x1, t.y1, t.x2, t.y2);
                    if (iou > bestIou) { bestIou = iou; best = t; }
                }

                if (best != null)
                {
                    best.x1 = d.x1; best.y1 = d.y1; best.x2 = d.x2; best.y2 = d.y2;
                    best.framesSinceSeen = 0;
                    best.updatedThisFrame = true;
                }
                else
                {
                    _activeTracks.Add(new TrashTrack
                    {
                        id = _nextTrackId++,
                        classId = d.cls,
                        x1 = d.x1, y1 = d.y1, x2 = d.x2, y2 = d.y2,
                        framesSinceSeen = 0,
                        wasBinned = false,
                        updatedThisFrame = true
                    });
                }
            }

            // Sticky wasBinned — a few occluded frames shouldn't undo it.
            foreach (TrashTrack t in _activeTracks)
            {
                if (!t.updatedThisFrame || t.wasBinned) continue;
                foreach (Det b in binDets)
                {
                    if (Containment(t.x1, t.y1, t.x2, t.y2, b.x1, b.y1, b.x2, b.y2) > binContainmentThreshold)
                    {
                        t.wasBinned = true;
                        break;
                    }
                }
            }

            for (int i = _activeTracks.Count - 1; i >= 0; i--)
            {
                TrashTrack t = _activeTracks[i];
                if (!t.updatedThisFrame) t.framesSinceSeen++;
                if (t.framesSinceSeen > trackDeathFrames)
                {
                    if (t.wasBinned)
                    {
                        Score++;
                        Debug.Log($"[TrashCameraView] Track #{t.id} ({LabelFor(t.classId)}) binned. Score: {Score}");
                    }
                    _activeTracks.RemoveAt(i);
                }
            }
        }

        private void DrawDetections(Mat image, List<Det> data)
        {
            foreach (Det d in data)
            {
                Scalar color = _palette[d.cls % _palette.Count];
                Imgproc.rectangle(image, new Point(d.x1, d.y1), new Point(d.x2, d.y2), color, 2);

                string label = $"{LabelFor(d.cls)}, {d.conf:F2}";
                int[] baseLine = new int[1];
                Size ls = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
                double top = System.Math.Max(d.y1, ls.height);
                Imgproc.rectangle(image, new Point(d.x1, top - ls.height),
                    new Point(d.x1 + ls.width, top + baseLine[0]), color, Core.FILLED);
                Imgproc.putText(image, label, new Point(d.x1, top),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(255), 1, Imgproc.LINE_AA);
            }
        }

        private string LabelFor(int cls)
        {
            if (_classNames != null && cls >= 0 && cls < _classNames.Count) return _classNames[cls];
            return cls.ToString();
        }

        private static float Iou(float ax1, float ay1, float ax2, float ay2,
                                 float bx1, float by1, float bx2, float by2)
        {
            float ix1 = Mathf.Max(ax1, bx1);
            float iy1 = Mathf.Max(ay1, by1);
            float ix2 = Mathf.Min(ax2, bx2);
            float iy2 = Mathf.Min(ay2, by2);
            if (ix2 <= ix1 || iy2 <= iy1) return 0f;
            float inter = (ix2 - ix1) * (iy2 - iy1);
            float uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
            return uni > 0f ? inter / uni : 0f;
        }

        // area(A ∩ B) / area(A). "What fraction of A is inside B?"
        private static float Containment(float ax1, float ay1, float ax2, float ay2,
                                         float bx1, float by1, float bx2, float by2)
        {
            float ix1 = Mathf.Max(ax1, bx1);
            float iy1 = Mathf.Max(ay1, by1);
            float ix2 = Mathf.Min(ax2, bx2);
            float iy2 = Mathf.Min(ay2, by2);
            if (ix2 <= ix1 || iy2 <= iy1) return 0f;
            float inter = (ix2 - ix1) * (iy2 - iy1);
            float areaA = (ax2 - ax1) * (ay2 - ay1);
            return areaA > 0f ? inter / areaA : 0f;
        }

        private static List<Scalar> BuildPalette()
        {
            return new List<Scalar>
            {
                new Scalar(255,  56,  56, 255), // Glass
                new Scalar(255, 157, 151, 255), // Metal
                new Scalar(255, 112,  31, 255), // Paper
                new Scalar(255, 178,  29, 255), // Plastic
                new Scalar(207, 210,  49, 255), // Waste
                new Scalar( 72, 249,  10, 255), // Bin  (bright green so it's obvious)
                new Scalar(146, 204,  23, 255), // Trash
                new Scalar( 61, 219, 134, 255), // Person
            };
        }

        private class TrashTrack
        {
            public int id;
            public int classId;
            public float x1, y1, x2, y2;
            public int framesSinceSeen;
            public bool wasBinned;
            public bool updatedThisFrame;
        }
    }
}

#endif
