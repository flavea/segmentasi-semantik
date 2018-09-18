# segmentasi semantik

Untuk melakukan training, ikuti langkah-langkah berikut:

- hapus file test.txt yang ada di folder-folder
- Download train data dari https://www.cs.toronto.edu/~vmnih/data/
- Simpan citra asli di folder building_input_files atau road_input_files tergantung tipenya dan simpan labelnya di building_label_files atau road_label_files tergantung tipenya
- Kemudian jalankan file train.py seperti berikut:

```python train.py \ --model="ResNet" \ --batch_size=1 \ --type="building" \ --training_step=5000 \ --saved=100```

- **model:** ResNet atau SSAI
- **batch_size:** Batch Size
- **type:** building atau road
- **training_step:** jumlah tahap training
- **saved:** model disimpan setiap berapa tahap


# Hasil Segmentasi SatNet
Dibuat berdasarkan model ResNet dengan Fully Convolutional Network
![alt text](https://image.ibb.co/mmHEsK/res.png "SatNet")

# Hasil Segmentasi SSAI
Dibuat berdasarkan paper “Multiple Object Extraction from Aerial Imagery with Convolutional Neural Networks” yang ditulis oleh Shunta Saito, Takayoshi Yamashita, Yoshimitsu Aoki pada tahun 2015
![alt text](https://image.ibb.co/hdWURe/ssai.png "SSAI")
