# segmentasi semantik

Untuk melakukan training, ikuti langkah-langkah berikut:

- hapus file test.txt yang ada di folder-folder
- Download train data dari https://www.cs.toronto.edu/~vmnih/data/
- Simpan citra asli di folder building_input_files atau road_input_files tergantung tipenya dan simpan labelnya di building_label_files atau road_label_files tergantung tipenya
- Kemudian jalankan file train.py seperti berikut:

```python train.py \ --model="ResNet" \ --batch_size=1 \ type="building" \ training_step=5000```

- **model:** ResNet atau SSAI
- **batch_size:** Batch Size
- **type:** building atau road
- **training_step:** jumlah tahap training
