using OpenCvSharp;
using OpenCvSharp.Face;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Yüz tanıma başlatılıyor...");

        string? cascadePath = FindFile("haarcascade_frontalface_default.xml");

        if (cascadePath is null)
        {
            Console.WriteLine("HATA: Haar Cascade dosyası bulunamadı!");
            Console.WriteLine("haarcascade_frontalface_default.xml");
            return;
        }

        using var faceDetector = new CascadeClassifier(cascadePath);

        string[] trainingFiles =
        {
            "zeynep1.jpg",
            "zeynep2.jpg",
            "zeynep3.jpg",
            "zeynep4.jpg"
        };

        List<Mat> trainingFaces = new List<Mat>();
        List<int> labels = new List<int>();

        const int ZEYNEP_LABEL = 1;

        foreach (string file in trainingFiles)
        {
            string? filePath = FindFile(file, Path.ChangeExtension(file, ".png"));
            if (filePath is null)
            {
                Console.WriteLine($"HATA: {file} bulunamadı!");
                return;
            }

            using Mat image = Cv2.ImRead(filePath);

            if (image.Empty())
            {
                Console.WriteLine($"HATA: {file} okunamadı!");
                return;
            }

            using Mat gray = new Mat();
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

            Rect[] faces = DetectFaces(faceDetector, gray);
            if (faces.Length == 0)
            {
                Console.WriteLine(
                    $"{file} içerisinde yüz bulunamadı!"
                );

                continue;
            }

            Rect face = faces
                .OrderByDescending(f => f.Width * f.Height)
                .First();

            trainingFaces.Add(PrepareFace(gray, face));
            labels.Add(ZEYNEP_LABEL);

            Console.WriteLine(
                $"{file} -> Yüz bulundu."
            );
        }

        if (trainingFaces.Count == 0)
        {
            Console.WriteLine(
                "Hiçbir eğitim yüzü bulunamadı!"
            );

            return;
        }

        Console.WriteLine();
        Console.WriteLine(
            $"{trainingFaces.Count} adet Zeynep yüzü ile eğitim yapılıyor..."
        );

        using var recognizer =
            LBPHFaceRecognizer.Create(
                radius: 1,
                neighbors: 8,
                gridX: 8,
                gridY: 8,
                threshold: double.MaxValue
            );

        recognizer.Train(
            trainingFaces,
            labels
        );

        Console.WriteLine("Eğitim tamamlandı.");

        string? inputFile = FindFile("farklı.png");

        if (inputFile is null)
        {
            Console.WriteLine(
                "HATA: farklı.png bulunamadı!"
            );

            return;
        }

        using Mat resultImage =
            Cv2.ImRead(inputFile);

        if (resultImage.Empty())
        {
            Console.WriteLine(
                $"{inputFile} okunamadı!"
            );

            return;
        }

        using Mat resultGray = new Mat();
        Cv2.CvtColor(
            resultImage,
            resultGray,
            ColorConversionCodes.BGR2GRAY
        );

        Rect[] detectedFaces = DetectFaces(faceDetector, resultGray);

        Console.WriteLine();
        Console.WriteLine(
            $"farklı.png içerisinde {detectedFaces.Length} yüz bulundu."
        );

        const double maxZeynepDistance = 70;
        const double minGapToOthers = 10;

        var scoredFaces = new List<(Rect Face, double Confidence)>();

        foreach (Rect face in detectedFaces)
        {
            using Mat prepared = PrepareFace(resultGray, face);

            recognizer.Predict(
                prepared,
                out int predictedLabel,
                out double confidence
            );

            Console.WriteLine(
                $"Yüz -> Label: {predictedLabel}, " +
                $"Confidence: {confidence:F2}"
            );

            scoredFaces.Add((face, confidence));
        }

        var ranked = scoredFaces
            .OrderBy(item => item.Confidence)
            .ToList();

        Rect? zeynepFace = null;

        if (ranked.Count > 0)
        {
            double best = ranked[0].Confidence;
            double second = ranked.Count > 1
                ? ranked[1].Confidence
                : double.MaxValue;

            if (best < maxZeynepDistance && (second - best) >= minGapToOthers)
                zeynepFace = ranked[0].Face;
        }

        int zeynepCount = 0;

        if (zeynepFace.HasValue)
        {
            var match = ranked[0];
            zeynepCount = 1;
            DrawLabel(resultImage, match.Face, "Zeynep", Scalar.Lime);
            Console.WriteLine(
                $"Zeynep bulundu! Confidence: {match.Confidence:F2}"
            );
        }

        string outputFile = "sonuc.png";

        Cv2.ImWrite(
            outputFile,
            resultImage
        );

        Console.WriteLine();
        Console.WriteLine("==============================");

        if (zeynepCount > 0)
        {
            Console.WriteLine(
                $"Zeynep bulundu! " +
                $"{zeynepCount} yüz işaretlendi."
            );
        }
        else
        {
            Console.WriteLine(
                "Zeynep bulunamadı."
            );
        }

        Console.WriteLine(
            $"Sonuç: {outputFile}"
        );

        Console.WriteLine("==============================");
    }

    static Rect[] DetectFaces(CascadeClassifier detector, Mat gray)
    {
        using Mat equalized = new Mat();
        Cv2.EqualizeHist(gray, equalized);

        return detector.DetectMultiScale(
            equalized,
            scaleFactor: 1.1,
            minNeighbors: 5,
            flags: HaarDetectionTypes.ScaleImage,
            minSize: new Size(80, 80)
        );
    }

    static Mat PrepareFace(Mat gray, Rect face)
    {
        Rect padded = ExpandRect(face, gray.Size(), 0.12);
        using Mat crop = new Mat(gray, padded);
        using Mat equalized = new Mat();
        Cv2.EqualizeHist(crop, equalized);

        Mat resized = new Mat();
        Cv2.Resize(equalized, resized, new Size(200, 200));
        return resized;
    }

    static Rect ExpandRect(Rect rect, Size imageSize, double pad)
    {
        int dx = (int)(rect.Width * pad);
        int dy = (int)(rect.Height * pad);
        int x = Math.Max(0, rect.X - dx);
        int y = Math.Max(0, rect.Y - dy);
        int width = Math.Min(imageSize.Width - x, rect.Width + (2 * dx));
        int height = Math.Min(imageSize.Height - y, rect.Height + (2 * dy));
        return new Rect(x, y, width, height);
    }

    static void DrawLabel(Mat image, Rect face, string text, Scalar color)
    {
        Cv2.Rectangle(image, face, color, 3);

        Size textSize = Cv2.GetTextSize(
            text,
            HersheyFonts.HersheySimplex,
            0.8,
            2,
            out _
        );

        int textX = face.X;
        int textY = face.Y - 10;
        if (textY - textSize.Height < 0)
            textY = face.Y + textSize.Height + 10;

        Cv2.Rectangle(
            image,
            new Point(textX, textY - textSize.Height - 5),
            new Point(textX + textSize.Width + 5, textY + 5),
            color,
            -1
        );

        Cv2.PutText(
            image,
            text,
            new Point(textX + 2, textY),
            HersheyFonts.HersheySimplex,
            0.8,
            Scalar.White,
            2
        );
    }

    static string? FindFile(params string[] names)
    {
        string[] dirs =
        {
            Directory.GetCurrentDirectory(),
            Path.Combine(Directory.GetCurrentDirectory(), "data"),
            AppContext.BaseDirectory,
            Path.Combine(AppContext.BaseDirectory, "data"),
        };

        foreach (string name in names)
        {
            if (string.IsNullOrWhiteSpace(name))
                continue;

            if (Path.IsPathRooted(name) && File.Exists(name))
                return name;

            foreach (string dir in dirs)
            {
                string path = Path.Combine(dir, name);
                if (File.Exists(path))
                    return path;
            }
        }

        return null;
    }
}
