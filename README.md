# Prediction Temperature Time Series Forcasting (Machine learning)

Ứng dụng dự đoán nhiệt độ 3 ngày tiếp theo dựa vào 5 ngày trước đó sử dụng các model machine learning

---

## 1.chức năng
- dự đoán nhiệt độ
- hỗ trợ nhiều model:
    - RandomForestRegressor
    - XGBoost
    - LinearRegression
- Hiển thị kết quả dự đoán dưới biểu đồ

---

## 2.cấu trúc thư mục 
```text
Temperature Forecasting/
├── README.md
├── data
│   ├── processed
│   └── raw
│       ├── DailyDelhiClimateTest.csv
│       └── DailyDelhiClimateTrain.csv
├── models/
├── reports/
├── requirements.txt
├── src/
```

---


## 3 Dataset

### 3.1 tải dữ liệu

- Kaggle :
    https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data

### 3.2 các cột dữ liệu bắt buộc

| Column |
|------|
| month |
| meantemp |
| meantemp_1 |
| meantemp_2 |
| meantemp_3 |
| meantemp_4 |
| humidity |
| wind_speed |
| meanpressure |

---

## 4. Cài đặt

### 4.1 Tạo môi trường ảo (khuyên dùng)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 4.2 Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 5. chỉnh cấu hình tham số mặc định
```text
config.py
```

---

## 6 Train model **(BẮT BUỘC)**

### 6.1 chạy các lệnh sau

```bash
# Linear Regression
python -m src.train --model_name LinearRegression

# Random Forest
python -m src.train --model_name RandomForestRegressor

# XGBoost
python -m src.train --model_name XGBoost
```

### 6.2 Model sau khi train sẽ nằm trong:
```text
models/*.pkl
```

---

### 7. chạy docker file
```bash
# build docker
docker build -t temperature .

#run docker containner
docker run -it  --rm -v ${PWD}/data/raw:/temperature/data/raw  -v ${PWD}/models:/temperature/models  temperature  bash
```
- sau khi đã vào trong containner hãy chạy các lệnh train model như phần 6.1 


---

## 8. Đánh giá mô hình

### metric đánh giá
- MAE
- MSE
- R2

---

## 9. Báo cáo

- EDA:
```
reports/edu/repot_temperature.html
```

- image_result:
```
reports/image_result/
```

- Kết quả huấn luyện:
```
reports/result/
```

## 👤 Tác giả

Phan Tùng  
GitHub: https://github.com/phantung205
