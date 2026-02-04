<div align="center">
    <h1>[DataFlow 20206 - HD4K - Learning Progress Prediction] <br> TRUSTEE: Tree-based Regression for Undergraduate Student Tracking and Educational Explainability</h1>
    
[![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![Kaggle](https://img.shields.io/badge/kaggle-20BEFF?logo=kaggle&logoColor=white)]()
[![Visual Studio](https://badgen.net/badge/icon/visualstudio?icon=visualstudio&label)](https://visualstudio.microsoft.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen)](https://lightgbm.readthedocs.io/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
</div>


## 📖 Giới thiệu

**TRUSTEE** là giải pháp dự báo kết quả học tập sinh viên do đội thi **HD4K** phát triển tham gia vòng loại cuộc thi **"DataFlow 2026: The Alchemy of Minds"**.

Dự án giải quyết bài toán hồi quy (Regression) trong lĩnh vực khai phá dữ liệu giáo dục. Mục tiêu cốt lõi là dự báo sớm số tín chỉ thực tế mà sinh viên sẽ hoàn thành trong kỳ học. Hệ thống không chỉ đưa ra dự đoán chính xác mà còn tập trung vào tính giải thích (Explainability), giúp nhà trường và sinh viên chủ động điều chỉnh lộ trình học tập để giảm thiểu rủi ro trượt môn hay chậm tiến độ.

Nếu bạn thấy dự án hữu ích, hãy cho nhóm một ngôi sao ⭐ trên GitHub nhé!

## 📂 Cấu trúc Dự án
```text
trustee/
├── data/
│   ├── vnhsge/                                   # Dữ liệu bổ sung: Phổ điểm thi THPTQG (2020-2024)
│   │   └── DATA_INFORMATION.md                   # Tài liệu chi tiết về dữ liệu điểm thi
│   ├── submissions/                              # Các file kết quả nộp bài (submission)
│   └── weights/                                  # Nơi lưu trữ trọng số (weights) của các mô hình
│
├── config/
│   ├── config_data.py                            # Thiết lập xử lý dữ liệu
│   └── config_model.py                           # Thiết lập cấu hình mô hình
│
├── preprocess/
│   ├── process_vnhsge.py                         # Script xử lý dữ liệu điểm thi THPTQG
│   └── process_data.py                           # Pipeline xử lý dữ liệu chính
│
├── model/
│   ├── hypertuning/                              # Tối ưu siêu tham số (Hyperparameter Tuning)
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── xgb.py
│   │   └── lgbm.py
│   │
│   ├── MODEL_HYPERPARAMETERS.md                  # Tài liệu ghi chép bộ tham số tốt nhất
│   │
│   ├── train/                                    # Script huấn luyện mô hình
│   │   └── train.py
│   │
│   └── test/                                     # Script kiểm thử và đánh giá
│       └── test.py
│
├── explainer/                                    # Module giải thích mô hình (xAI)
│   ├── shap_explainer.py                         # Phân tích toàn cục với SHAP
│   ├── lime_explainer.py                         # Phân tích cục bộ với LIME
│   └── dice_explainer.py                         # Phân tích phản chứng với DiCE
│
├── utils/
│   ├── set_up.py                                 # Thiết lập môi trường, đảm bảo tính tái lập
│   ├── evaluate.py                               # Các hàm tính toán metric đánh giá
│   └── save_submission.py                        # Xuất file kết quả chuẩn format cuộc thi
│
├── scripts/                                    
│   ├── dataflow2026_hd4k_process_vnhsge.ipynb    # Script xử lý dữ liệu điểm thi THPTQG
│   ├── dataflow2026_hd4k_run_model.ipynb         # Script huấn luyện và kiểm thử mô hình
│   ├── dataflow2026_hd4k_run_explainer.ipynb     # Script giải thích mô hình với xAI
│   │
│   ├── HOW_TO_RUN_COLAB.md                       # Hướng dẫn chạy trên Google Colab
│   └── HOW_TO_RUN_KAGGLE.md                      # Hướng dẫn chạy trên Kaggle
│
├── report/                                    
│   ├── img/                                      # Ảnh sử dụng trong report, README
│   ├── TRUSTEE_report.pdf                        # File báo cáo dự án
│   └── TRUSTEE_slide_pdf.pdf                     # Slide thuyết trình dự án (pdf)
│
├── .gitignore                       
├── .gitattributes
├── LICENSE                                       # Giấy phép MIT
├── requirements.txt                              # Danh sách thư viện cần thiết
└── README.md                                      
```

## 💻 Yêu cầu Hệ thống & Hướng dẫn Sử dụng
Có tổng tất cả 3 scripts, cụ thể:
- Script xử lý dữ liệu điểm thi THPTQG: dataflow2026_hd4k_process_vnhsge.ipynb
    - Chạy local ngay trên máy tính cá nhân
    - Đảm bảo đã cài đặt hai thư viện pandas và numpy
    - Đảm bảo dung lượng ổ cứng trống ít nhất 1GB

- Script huấn luyện và kiểm thử mô hình: dataflow2026_hd4k_run_model.ipynb
    - Chạy trên Google Colab hoặc Kaggle

- Script giải thích mô hình với xAI: dataflow2026_hd4k_run_explainer.ipynb
    - Chạy trên Google Colab hoặc Kaggle

Chi tiết thông tin, hướng dẫn và thời gian chạy từng script có thể đọc trong chính các file jupyter notebook.

## 📜 Giấy phép
Dự án được phân phối dưới giấy phép MIT. Xem file [LICENSE](LICENSE) để biết chi tiết.

## 📞 Liên hệ
Mọi thắc mắc hoặc góp ý, xin vui lòng liên hệ với chúng tôi qua GitHub Issues, LinkedIn hoặc Facebook:

[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/CryAndRRich/trustee)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/in/cryandrich/)
[![Facebook](https://img.shields.io/badge/Facebook-0866FF?style=flat&logo=facebook&logoColor=white)](https://www.facebook.com/namhai.tran.73550794)

Chúng tôi trân trọng mọi phản hồi và đóng góp của bạn để giúp dự án ngày càng hoàn thiện hơn!