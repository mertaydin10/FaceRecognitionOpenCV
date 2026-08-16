# Face Recognition OpenCV

OpenCvSharp ile Haar cascade ve LBPH kullanarak eğitim fotoğraflarındaki kişiyi (Zeynep) grup görselinde bulan bir .NET konsol uygulaması.

Program `zeynep1`–`zeynep4` görsellerinden yüzü öğrenir, `farklı.png` içindeki yüzlerle karşılaştırır ve yalnızca Zeynep olarak kabul ettiği yüze kutu çizerek `sonuc.png` üretir.

## Gereksinimler

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- Windows (OpenCvSharp native runtime: `OpenCvSharp4.runtime.win`)

## Proje yapısı

```text
reco/
├── recog.sln
├── recog/
│   ├── Program.cs
│   ├── recog.csproj
│   ├── data/
│   │   └── haarcascade_frontalface_default.xml
│   ├── zeynep1.png … zeynep4.png   # eğitim
│   └── farklı.png                  # test (grup fotoğrafı)
```

## Çalıştırma

Proje klasöründen:

```bash
cd recog
dotnet run
```

Çıktı `recog/sonuc.png` olarak kaydedilir. Konsolda her yüzün LBPH mesafesi (confidence) yazılır; düşük değer daha iyi eşleşme demektir.

## Girdi dosyaları

| Dosya | Rol |
| --- | --- |
| `zeynep1.jpg` / `.png` … `zeynep4.jpg` / `.png` | Zeynep’in eğitim fotoğrafları |
| `farklı.png` | İçinde Zeynep’in de olduğu grup görseli |
| `data/haarcascade_frontalface_default.xml` | Yüz tespiti (Haar cascade) |

Dosyalar proje klasöründe veya `bin` çıktı klasöründe olabilir. Eğitim görselleri `.jpg` yoksa otomatik olarak `.png` denenir.

Kendi fotoğraflarınla denemek için aynı isimlerle dosyaları değiştirmen yeterli.

## Nasıl çalışır?

1. Haar cascade her görselde yüzleri bulur.
2. Yüzler 200×200 griye çevrilir ve LBPH modeli yalnızca Zeynep etiketiyle eğitilir.
3. Grup fotoğrafındaki her yüz için mesafe hesaplanır.
4. En düşük mesafeli yüz, eşik ve diğer yüzlere göre fark yeterliyse Zeynep işaretlenir; diğer yüzler çizilmez.

Tek kişiyle eğitimde LBPH her yüze aynı etiketi verebileceği için sabit bir “herkes Zeynep” eşiği kullanılmaz; en iyi aday diğerlerinden ayrışmalıdır.

## NuGet paketleri

- `OpenCvSharp4` 4.10.0.20240615
- `OpenCvSharp4.Extensions` 4.10.0.20240615
- `OpenCvSharp4.runtime.win` 4.10.0.20240615
