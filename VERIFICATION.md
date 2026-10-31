# Kiểm chứng trước bàn giao

Ngày: 2026-10-31. Môi trường cục bộ: Windows, Python 3.11, thư viện ảnh theo `requirements.txt`.

Lệnh `python -m unittest discover -s tests -v`: **10 tests passed**.

Đã kiểm tra:

- Hash cell toán tử và DEFAULT_PARAMS khớp notebook Final trước đó.
- Determinism và quy tắc chọn 2–3 lỗi L2, hình dạng và dtype ảnh.
- Pipeline tự động đầy đủ trên fixture 40 ảnh, 80 câu hỏi của đủ train/val/test; 117 ảnh calibration và 28 ảnh ablation theo cấu hình test thu nhỏ.
- Export manifest, đường dẫn, kích thước ảnh, hash, test không bị bịa đáp án.
- Nhãn 70/30 mô phỏng được ghi provenance `human_review_performed=False`.
- Hai worker thực sự tạo lại PNG giống hệt lần tuần tự.
- Resume song song tái sử dụng output và giữ nguyên manifest.
- Phục hồi PNG thiếu sidecar sau gián đoạn.
- Dừng khi sidecar hoặc annotation bị thay đổi.
- Controller tạo đúng một script và một lời gọi `sbatch` (mock), lưu snapshot cấu hình; không fallback tính toán trên login node.

`python run_all.py --plan-only` đã chạy thành công với đường dẫn mặc định. Chưa submit job thật lên UIT, chưa xác minh quyền partition/account của trường, chưa tạo dữ liệu full trên server và chưa đo thời gian/RAM/dung lượng thực tế ở quy mô đó. GitHub Actions được cấu hình để chạy tests trên Ubuntu; kết quả CI phải xem riêng trên GitHub, không suy ra từ test Windows.
