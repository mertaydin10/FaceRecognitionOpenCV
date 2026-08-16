using OpenCvSharp;
using OpenCvSharp.Face;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Yüz tanıma başlatılıyor...");

        // ============================================================
        // 1. Yüz tespit modeli
        // ============================================================

        string cascadePath = "haarcascade_frontalface_default.xml";

        if (!File.Exists(cascadePath))
        {
            Console.WriteLine("HATA: Haar Cascade dosyası bulunamadı!");
            Console.WriteLine(cascadePath);
            return;
        }

        using var faceDetector = new CascadeClassifier(cascadePath);

        // ============================================================
        // 2. Zeynep'in eğitim fotoğrafları
        // ============================================================

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
            if (!File.Exists(file))
            {
                Console.WriteLine($"HATA: {file} bulunamadı!");
                return;
            }

            using Mat image = Cv2.ImRead(file);

            if (image.Empty())
            {
                Console.WriteLine($"HATA: {file} okunamadı!");
                return;
            }

            // Gri tona çevir
            using Mat gray = new Mat();

            Cv2.CvtColor(
                image,
                gray,
                ColorConversionCodes.BGR2GRAY
            );

            // Kontrastı biraz iyileştir
            Cv2.EqualizeHist(gray, gray);

            // Yüzleri bul
            Rect[] faces = faceDetector.DetectMultiScale(
                gray,
                scaleFactor: 1.1,
                minNeighbors: 5,
                flags: HaarDetectionTypes.ScaleImage,
                minSize: new Size(80, 80)
            );

            if (faces.Length == 0)
            {
                Console.WriteLine(
                    $"{file} içerisinde yüz bulunamadı!"
                );

                continue;
            }

            // İlk yüzü eğitim için kullan
            Rect face = faces[0];

            using Mat faceImage = new Mat(
                gray,
                face
            );

            // LBPH için yüzleri aynı boyuta getiriyoruz
            Mat resizedFace = new Mat();

            Cv2.Resize(
                faceImage,
                resizedFace,
                new Size(200, 200)
            );

            trainingFaces.Add(resizedFace);
            labels.Add(ZEYNEP_LABEL);

            Console.WriteLine(
                $"{file} -> Yüz bulundu."
            );
        }

        // ============================================================
        // 3. Eğitim verisi kontrolü
        // ============================================================

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

        // ============================================================
        // 4. LBPH yüz tanıyıcı oluştur
        // ============================================================

        using var recognizer =
            LBPHFaceRecognizer.Create(
                radius: 1,
                neighbors: 8,
                gridX: 8,
                gridY: 8,
                threshold: 80
            );

        recognizer.Train(
            trainingFaces,
            labels
        );

        Console.WriteLine("Eğitim tamamlandı.");

        // ============================================================
        // 5. Farklı.png dosyasını aç
        // ============================================================

        string inputFile = "farklı.png";

        if (!File.Exists(inputFile))
        {
            Console.WriteLine(
                $"HATA: {inputFile} bulunamadı!"
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

        // ============================================================
        // 6. Farklı.png -> Gri görüntü
        // ============================================================

        using Mat resultGray = new Mat();

        Cv2.CvtColor(
            resultImage,
            resultGray,
            ColorConversionCodes.BGR2GRAY
        );

        Cv2.EqualizeHist(
            resultGray,
            resultGray
        );

        // ============================================================
        // 7. Farklı.png içerisindeki yüzleri bul
        // ============================================================

        Rect[] detectedFaces =
            faceDetector.DetectMultiScale(
                resultGray,
                scaleFactor: 1.1,
                minNeighbors: 5,
                flags: HaarDetectionTypes.ScaleImage,
                minSize: new Size(80, 80)
            );

        Console.WriteLine();
        Console.WriteLine(
            $"farklı.png içerisinde {detectedFaces.Length} yüz bulundu."
        );

        // ============================================================
        // 8. Bulunan her yüzü Zeynep ile karşılaştır
        // ============================================================

        int zeynepCount = 0;

        foreach (Rect face in detectedFaces)
        {
            using Mat faceImage =
                new Mat(resultGray, face);

            using Mat resizedFace =
                new Mat();

            Cv2.Resize(
                faceImage,
                resizedFace,
                new Size(200, 200)
            );

            // LBPH tahmini
            recognizer.Predict(
                resizedFace,
                out int predictedLabel,
                out double confidence
            );

            Console.WriteLine(
                $"Yüz -> Label: {predictedLabel}, " +
                $"Confidence: {confidence:F2}"
            );

            // ========================================================
            // LBPH'de confidence DÜŞÜK oldukça eşleşme daha iyidir.
            //
            // threshold = 80 olarak belirledik.
            // ========================================================

            if (
                predictedLabel == ZEYNEP_LABEL &&
                confidence < 80
            )
            {
                zeynepCount++;

                // ====================================================
                // 9. Yüzün etrafına kare çiz
                // ====================================================

                Cv2.Rectangle(
                    resultImage,
                    face,
                    Scalar.Red,
                    3
                );

                // ====================================================
                // 10. "Zeynep" yazısını kare üzerine koy
                // ====================================================

                string text = "Zeynep";

                int baseline;

                Size textSize =
                    Cv2.GetTextSize(
                        text,
                        HersheyFonts.HersheySimplex,
                        0.8,
                        2,
                        out baseline
                    );

                // Yazının karenin üstünde kalması için
                // uygun konum hesaplanıyor.

                int textX = face.X;

                int textY =
                    face.Y - 10;

                // Eğer yüz görüntünün üst kısmındaysa
                // yazıyı karenin içine koy.
                if (textY - textSize.Height < 0)
                {
                    textY =
                        face.Y + textSize.Height + 10;
                }

                // Yazının arka planı
                Cv2.Rectangle(
                    resultImage,
                    new Point(
                        textX,
                        textY - textSize.Height - 5
                    ),
                    new Point(
                        textX + textSize.Width + 5,
                        textY + 5
                    ),
                    Scalar.Red,
                    -1
                );

                // Yazı
                Cv2.PutText(
                    resultImage,
                    text,
                    new Point(
                        textX + 2,
                        textY
                    ),
                    HersheyFonts.HersheySimplex,
                    0.8,
                    Scalar.White,
                    2
                );

                Console.WriteLine(
                    $"Zeynep bulundu! " +
                    $"Confidence: {confidence:F2}"
                );
            }
            else
            {
                Console.WriteLine(
                    $"Bu yüz Zeynep olarak tanınmadı. " +
                    $"Confidence: {confidence:F2}"
                );
            }
        }

        // ============================================================
        // 11. Sonucu kaydet
        // ============================================================

        string outputFile = "sonuc.png";

        Cv2.ImWrite(
            outputFile,
            resultImage
        );

        // ============================================================
        // 12. Sonuç
        // ============================================================

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
}
