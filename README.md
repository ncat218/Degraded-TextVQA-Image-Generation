# Tạo toàn bộ ảnh TextVQA suy thoái trên UIT HPC

Chạy **một lệnh `python run_all.py`** để tự tạo script `.sh`, submit **một job Slurm**, chuẩn bị môi trường Python và tạo ảnh suy thoái cho toàn bộ **train + validation + test**. Không chạy bốn notebook hay điền nhãn giữa các bước.

## 1. Dữ liệu và output đã cấu hình

Nguồn clean đã có đủ trên server:

```text
/datastore/dungmt/textvqa/data/benchmark_v1/raw/textvqa_clean/TextVQA/
├── train_images/                 # gồm ảnh train và validation
├── test_images/
├── TextVQA_0.5.1_train.json
├── TextVQA_0.5.1_val.json
└── TextVQA_0.5.1_test.json
```

| Split gốc | Ảnh chính | Câu hỏi giữ lại |
|---|---:|---:|
| train | 21.953 | 34.602 |
| val | 3.166 | 5.000 |
| test | 3.289 | 5.734 |
| **Tổng** | **28.408** | **45.336** |

Ảnh suy thoái và manifest được tạo trực tiếp trong:

```text
/datastore/dungmt/textvqa/data/benchmark_v1/raw/full_textvqa_degraded/
```

Code chỉ đọc dữ liệu nguồn, không chỉnh sửa clean, không sửa dataset suy thoái 1k/10k cũ. Không tải lại ảnh vì nguồn đã đầy đủ. Nếu thiếu JSON/ảnh hoặc số lượng không đúng, chương trình dừng và chỉ rõ nguyên nhân.

Đường dẫn nằm trong `configs/full_textvqa.json`; tài khoản khác có thể đổi ba trường `source_root`, `output_root`, `workspace_root`. Ba thư mục phải tách biệt, không lồng nhau. Output mới dùng thư mục trống hoặc chính output dở dang do repo này tạo.

## 2. Lấy code và chạy

Đăng nhập UIT HPC như thường lệ, rồi chạy lần đầu:

```bash
cd /datastore/dungmt
git clone https://github.com/ncat218/Degraded-TextVQA-Image-Generation.git
cd Degraded-TextVQA-Image-Generation
python run_all.py
```

Nếu server chỉ nhận lệnh `python3`, dùng `python3 run_all.py`. Có Python 3.11 là thuận tiện nhất; môi trường thư viện hỗ trợ Python 3.10–3.13.

Nếu đã clone trước đó và chưa sửa code:

```bash
cd /datastore/dungmt/Degraded-TextVQA-Image-Generation
git pull --ff-only
python run_all.py
```

Chương trình in **job ID**, nơi lưu log và output. Sau khi submit xong có thể đóng SSH; job do Slurm quản lý. Không chạy riêng setup, preflight, calibration, freeze hoặc final.

**Không cần API Kaggle, tài khoản Hugging Face, CUDA toolkit, PyTorch hay model checkpoint.** Lần chạy đầu cần Internet trong allocation để cài thư viện qua pip, hoặc có sẵn cache/mirror phù hợp. Môi trường nằm trong datastore, không trong Home.

Muốn xem cấu hình mà không submit hoặc đọc ảnh:

```bash
python run_all.py --plan-only
```

## 3. Slurm tự hoạt động thế nào?

- Trên login node: chỉ kiểm tra vị trí năm input (ba JSON, hai thư mục ảnh), lưu snapshot config, tạo `.sh`, gọi `sbatch`. Không hash/đọc hàng chục nghìn ảnh và không tạo dữ liệu ở login node.
- Trong Slurm allocation: chuẩn bị venv và thực hiện toàn bộ pipeline.
- Nếu chạy từ allocation đã có `SLURM_JOB_ID`: thực hiện ngay trong allocation đó, không submit thêm.
- Không có `sbatch` và cũng không có allocation: báo lỗi, không tự chuyển sang tính toán trên login node.

Cấu hình hiện dùng partition `normal` vì `small` đang không cấp được tài nguyên tại thời điểm chuyển cấu hình:

| Tham số | Mặc định |
|---|---|
| Account | `uit` |
| Partition | `normal` |
| GRES | `gpu:2g.35gb:1` |
| QOS | `mig2-35` |
| CPU | 12 |
| RAM hệ thống | 40 GB |
| Thời gian yêu cầu | 24 giờ |
| Số tiến trình tạo ảnh | 12 |

Cấu hình xin 12/16 CPU và 40/72 GB RAM trong giới hạn `normal` của tài liệu UIT. Giữ 12 worker để không tăng thêm nhu cầu CPU/RAM khi chỉ cần đổi hàng đợi. QOS `mig2-35` khớp job `normal` đã được tài khoản dungmt submit trên server. Preflight và các bước kiểm chứng chạy trong cùng job, không cần submit job thử riêng. 24 giờ là giới hạn được yêu cầu, không phải dự đoán thời gian hoàn tất; chuyển partition không bảo đảm được chạy ngay.

**Thuật toán gốc chạy CPU bằng OpenCV/NumPy; GPU MIG được yêu cầu theo mẫu allocation của trường nhưng không dùng cho biến đổi ảnh.** Việc chuyển sang `normal` nhằm đổi hàng đợi, không chuyển thuật toán sang GPU. Quyền chạy tác vụ này trên queue GPU vẫn phụ thuộc chính sách UIT; nếu quản trị viên cấp queue CPU riêng thì cần cấu hình theo queue được cấp. Không tự đoán rằng queue CPU tồn tại.

Cấu hình hiện kiểm tra ba profile được tài liệu nêu (`small`, `normal`, `large`), GRES tương ứng và giới hạn CPU/RAM. `gres` có thể để chuỗi rỗng nếu được phép dùng tác vụ CPU trong một trong các partition đó. Chưa kiểm chứng quyền submit/queue thực tế trên tài khoản trường. Nếu Slurm từ chối, gửi nguyên thông báo `sbatch`; chương trình sẽ không chuyển sang chạy trên login.

Giới hạn walltime không vượt 72 giờ. Không tự submit lại liên tục khi lỗi hoặc hết giờ; chạy lại cùng lệnh để resume, tuân thủ quy định giãn cách submit của UIT.

## 4. Theo dõi job

### Chuyển job 8021 đang pending ở small sang một job normal mới

Thực hiện trên server, không hủy các job khác đang dùng chung tài khoản:

```bash
scancel 8021
cd /datastore/dungmt/textvqa/Tuan/Degraded-TextVQA-Image-Generation
git pull --ff-only && python run_all.py
```

Lệnh cuối chỉ submit khi cập nhật Git thành công. Nó tạo script/config snapshot mới và in job ID mới; job 8021 cũ không tự đổi partition khi cập nhật repo. Không thay đường dẫn nguồn/output hoặc xóa dữ liệu. Nếu Git báo thay đổi cục bộ, dừng và kiểm tra; không reset/ghi đè cấu hình tùy ý.

Script `.sh`, snapshot cấu hình và log nằm tại:

```text
/datastore/dungmt/_workspace_full_textvqa_degraded/logs/
```

Sau khi submit, chương trình in lệnh `tail -f` có tên file cụ thể. Các lệnh thường dùng:

```bash
squeue -u "$USER"
ls -lt /datastore/dungmt/_workspace_full_textvqa_degraded/logs/
tail -f /datastore/dungmt/_workspace_full_textvqa_degraded/logs/<ten_file>.out
tail -n 80 /datastore/dungmt/_workspace_full_textvqa_degraded/logs/<ten_file>.err
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Thay `<ten_file>`/`<JOB_ID>` bằng giá trị thực tế. Log audit mỗi 1.000 ảnh và log tạo ảnh mỗi 100 output; lúc cài thư viện, pip cũng ghi log.

## 5. Pipeline tự chạy đến cuối

1. Đọc đủ JSON, kiểm tra ID, câu hỏi, đáp án train/val và tìm ảnh đúng split.
2. Kiểm tra header ảnh, kích thước và tính SHA-256 nguồn bằng các CPU được cấp. Giữ mọi image ID của nguồn; ảnh trùng byte nếu có được báo trong provenance, không lặng lẽ bỏ ảnh. Giữ nguyên ba split chính thức.
3. Tạo và khóa source image manifest, QA manifest và split policy.
4. Chọn cố định 300 ảnh train cho phần kiểm tra phụ, 120 ảnh trong đó cho các lỗi đơn; chọn riêng một tập 400 ảnh train cho ablation. Các ảnh này **vẫn có trong main**. Calibration ở đây không phải một holdout độc lập.
5. Preflight 25 ảnh và ước lượng dung lượng; kiểm tra cả giới hạn cấu hình lẫn dung lượng trống thực tế.
6. Tạo calibration L1/L2/L3: 300 × 3 mix + 120 × 7 × 3 lỗi đơn = **3.420 ảnh phụ**.
7. Xuất gallery, template review và tự gán nhãn 70/30. **Đây là nhãn mô phỏng ngẫu nhiên, không có human review**. CSV, provenance và frozen config đều ghi điều này. Gallery vẫn có thể mở xem sau.
8. Khóa config, nguồn, seed, code hashes và nhãn mô phỏng; freeze không có nghĩa là ảnh đã được con người duyệt về khả năng trả lời.
9. Frozen preflight, sau đó main **28.408 ảnh realistic_mix L2**.
10. Tạo ablation **400 × 7 = 2.800 ảnh lỗi đơn L2**. Recheck dung lượng sau khi đã biết kích thước main thực tế.
11. Kiểm tra lại hash mọi PNG, xuất manifest câu hỏi, metadata JSONL và báo cáo hoàn tất.

Mặc định 70% trên 615 dòng được làm tròn thành **431 EASY / 184 DIFFICULT_BUT_ANSWERABLE**. Nhãn không dùng để chọn/loại ảnh hoặc chứng minh ảnh answerable.

Preflight chỉ là ước lượng từ 25 mẫu. Giới hạn output mặc định **100 GB** không có nghĩa code sẽ chiếm 100 GB hoặc server có sẵn quota đó; kiểm tra thêm dung lượng trống với biên 25%. Thời gian 24 giờ là thời gian xin job, không phải thời gian hoàn tất đã đo trên HPC. Không có benchmark full thật trên server trong repo.

## 6. Thuật toán giữ nguyên từ notebook cũ

`src/textvqa_degradation/operators.py` chứa nguyên code cell toán tử của notebook `04_Final.ipynb`, bao gồm `DEFAULT_PARAMS`. Hash code cell gốc:

```text
86e80ed9e98d4607852dfdbe0ff9f5381aba1c81849251b009cb54834333c12e
```

Các phép biến đổi:

- Motion/camera shake: tích chập PSF đường thẳng hoặc random walk.
- Defocus: tích chập kernel đĩa/ellipse.
- Low light: giảm phơi sáng và gamma; nhiễu Poisson + Gaussian đọc cảm biến theo thứ tự gốc.
- Overexposure/glare: tăng phơi sáng, clipping và vùng glare Gaussian khi được chọn.
- Low resolution: downsample AREA rồi upsample CUBIC.
- Perspective: homography với kiểm tra diện tích biên không quá 5%, tối đa 50 lần thử.
- JPEG: encode/decode ở quality ngẫu nhiên theo mức.

Main L2 phối hợp **2–3 phép**, không lặp cùng loại; không đồng thời chọn cả motion và defocus, không đồng thời chọn low light và overexposure. Thứ tự các bước được giữ nguyên.

Ba JSON QA v0.5.1 không cung cấp OCR boxes. Vì vậy dùng nguyên nhánh `image_relative_fallback` của thuật toán cũ; không chạy OCR mới hoặc giả kích thước chữ. Không EXIF-transpose ảnh nguồn. Ảnh kết quả là RGB PNG cùng kích thước ảnh clean đã đọc.

Cùng input/config/thư viện/code thì việc chạy tuần tự hay nhiều tiến trình cho cùng output. Bộ full **không khớp byte với dataset 1k/10k cũ**, vì manifest và hash cấu hình mới làm seed mỗi ảnh thay đổi. Không đổi mặc định tham số dựa trên kết quả VQA.

## 7. Cây output

```text
full_textvqa_degraded/
├── main/
│   ├── images/realistic_mix/L2/
│   │   ├── train/<image_id>.png       # kèm <image_id>.png.json
│   │   ├── val/<image_id>.png
│   │   └── test/<image_id>.png
│   ├── final_main_manifest.csv       # 45.336 dòng câu hỏi
│   ├── question_manifest.csv
│   ├── image_manifest.csv            # 28.408 dòng ảnh
│   ├── manifests/train.csv
│   ├── manifests/val.csv
│   ├── manifests/test.csv
│   ├── diagnostics_summary.json
│   └── failures.json
├── calibration/
│   ├── images/<condition>/<L1|L2|L3>/train/...
│   ├── calibration_manifest.csv
│   ├── image_manifest.csv
│   └── manual_review/
│       ├── review_gallery.html
│       ├── previews/...
│       ├── review_samples.csv
│       ├── reviewed_samples.csv
│       └── reviewed_samples.random_label_provenance.json
├── ablation/
│   ├── images/<condition>/L2/train/...
│   ├── ablation_manifest.csv          # 400 ảnh nguồn cố định
│   ├── image_manifest.csv            # 2.800 output
│   └── question_manifest.csv
├── config/
│   ├── draft_degradation_config.json
│   └── frozen_degradation_config.json
├── metadata/
│   ├── source_image_manifest.csv
│   ├── locked_eval_manifest.csv      # tên tương thích; chứa cả train/val/test
│   ├── source_provenance.json
│   ├── split_policy.json
│   └── degradation_params.jsonl
├── preflight/
├── environment.json
├── generation_summary.json
└── pipeline_failure.json
```

Manifest/config còn có file `.sha256`. Manifest chính chứa `dataset`, `split`, `image_id`, `source_image_id`, `question_id`, `question`, `answers`, `gt_available`, các đường dẫn clean/degraded, kích thước hai ảnh, `condition/level`, `degradation_type/severity`, `seed`, `params`, hash nguồn/kết quả/cấu hình. `params` mới là danh sách lỗi thực tế trong một ảnh realistic_mix.

`test` không có đáp án công khai: giữ các giá trị từ JSON hoặc `[]` nếu trường thiếu và `gt_available=False`; không bịa đáp án. Có thể suy luận trên ảnh test, nhưng không dùng các giá trị thiếu này làm ground truth khi chấm VQA.

## 8. Khi nào đã hoàn tất?

Log có:

```text
main outputs: 28408 failures: 0
ablation outputs: 2800 failures: 0
READY FOR DOWNSTREAM TEXTVQA / VQA EVALUATION: main, ablation
```

Đồng thời `generation_summary.json` phải có `status: complete`, `main_images: 28408`, `main_questions: 45336`. Xác nhận thêm Slurm `State=COMPLETED`, `ExitCode=0:0`; lỗi sau đó hoặc job bị kill không được coi là lượt chạy thành công.

Để sử dụng đánh giá chính thức, chọn manifest `main/manifests/val.csv`. Không chấm gộp train/val/test thành một điểm. Một ảnh nhiều câu hỏi dùng cùng đường ảnh suy thoái; mọi biến thể của cùng nguồn phải ở cùng split nếu xây dựng thêm split thực nghiệm. Output không sao chép ảnh clean: vẫn cần giữ dataset clean nguồn khi chạy paired comparison.

## 9. Resume và lỗi

Nếu mất kết nối SSH sau submit: job vẫn chạy. Nếu hết giờ/bị hủy/lỗi: chạy lại **cùng lệnh và cấu hình**. Code kiểm tra lại nguồn và PNG/JSON trước khi tái sử dụng; không tạo lại ảnh đã hợp lệ. Hai job không được cùng ghi một output; có khóa hệ điều hành để ngăn trường hợp đó.

- PNG mồ côi do job ngắt giữa hai lần ghi PNG/JSON: tạo lại xác định bằng cùng seed.
- PNG và JSON cùng tồn tại nhưng hash/seed/config sai: dừng, không ghi đè lặng lẽ.
- Annotation/source bytes thay đổi: dừng ở artifact khóa hoặc đối chiếu hash.
- Đổi thuật toán/seed/nguồn/tham số: cần output mới; không `git pull` đổi code giữa một lượt đang dở vì code hash được khóa.
- Có thể đổi `workers`, cấu hình Slurm, thời gian và trần dung lượng để resume; đây là tham số vận hành, không đổi seed/cấu hình suy thoái.
- `failure.json` trong từng stage và `pipeline_failure.json` lưu lỗi gần nhất. Sau một lượt kiểm chứng thành công, lỗi cũ được đánh dấu resolved.

Resume vẫn đọc/hash ảnh để xác minh, có thể mất thời gian. Chương trình không tự giả định một ảnh hợp lệ chỉ vì tên file tồn tại.

## 10. Kiểm thử dành cho người phát triển

Sau khi có môi trường chứa `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

Tests chạy trên ảnh giả nhỏ có nội dung pixel thật, kiểm tra mọi split/câu hỏi, giữ nguyên toán tử và tham số, deterministic mix, tạo PNG và metadata, resume tuần tự/song song, phục hồi PNG mồ côi, chặn sidecar/annotation thay đổi và controller chỉ submit một job. Không cần GPU cho tests. Các fixtures chỉ dùng trong thư mục tạm; không có kết quả dataset full được tạo sẵn trong GitHub. Chi tiết kiểm chứng và giới hạn nằm trong `VERIFICATION.md`.

## 11. Bố trí repo

```text
run_all.py                       # file duy nhất người dùng khởi chạy
configs/full_textvqa.json         # path + tài nguyên Slurm
requirements.txt                 # thư viện xử lý ảnh đã pin
src/textvqa_degradation/          # thuật toán gốc, audit, pipeline
tests/                          # kiểm thử tự động
operator_provenance.json          # hash toán tử notebook cũ
```

Không commit dataset, output, môi trường Python hoặc thông tin đăng nhập lên GitHub.
